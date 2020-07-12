from typing import Dict, List
import torch
import torch.nn.functional as F
from torch import nn

import fvcore.nn.weight_init as weight_init
from detectron2.layers import ShapeSpec
from detectron2.utils.registry import Registry

from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY

from .anchor_generator import build_anchor_generator
from .horpn_outputs import HORPNOutputs, find_top_horpn_proposals


HORPN_HEAD_REGISTRY = Registry("RPN_HEAD")
HORPN_HEAD_REGISTRY.__doc__ = """
Registry for HORPN heads, which take feature maps and perform
object interactiveness classification and bounding box regression for anchors.

The registered object will be called with `obj(cfg, input_shape)`.
The call should return a `nn.Module` object.
"""

def build_horpn_head(cfg, input_shape):
    """
    Build a HORPN head defined by `cfg.MODEL.HORPN.HEAD_NAME`
    """
    name = cfg.MODEL.RPN.HEAD_NAME
    return HORPN_HEAD_REGISTRY.get(name)(cfg, input_shape)


@HORPN_HEAD_REGISTRY.register()
class StandardHORPNHead(nn.Module):
    """
    HORPN classification and regression heads. It contains the following components:
        * person branch:
            1. Hidden feature extraction for each sliding window position.
            2. Objectness logits (if including person)
            3. Pesron bounding-box deltas.
            4. find top person boxes to reason object interactness logits
        * object branch:
            1. Hidden feature extraction for each sliding windown position.
            2. Relational network to reason interactness logits, together with top person boxes
            3. Object bounding-box deltas.
    """
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()

        # fmt: off
        num_conv   = cfg.MODEL.HORPN.NUM_CONV
        conv_dim   = cfg.MODEL.HORPN.CONV_DIM
        num_rn_fc  = cfg.MODEL.HORPN.NUM_RN_FC
        rn_fc_dim  = cfg.MODEL.HORPN.RN_FC_DIM
        self.topk  = cfg.MODEL.HORPN.TOPK_PERSON_CELLS
        # fmt: on

        anchor_generator = build_anchor_generator(cfg, input_shape)
        box_dim = self.box_dim = anchor_generator.box_dim

        # standard HORPN head is shared across levels:
        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        # HORPN head should take the same input as anchor generator
        num_cell_anchors = anchor_generator.num_cell_anchors        
        assert len(set(num_cell_anchors)) == 1, "Each level must have the same number of anchors"
        num_cell_anchors = self.num_cell_anchors = num_cell_anchors[0]

        # 3x3 conv for the person hidden features
        _p_out_dim = in_channels
        self.person_convs = []
        for k in range(num_conv):
            conv = nn.Conv2d(_p_out_dim, conv_dim, kernel_size=3, stride=1, padding=1)
            self.add_module("person_conv{}".format(k + 1), conv)
            self.person_convs.append(conv)
            _p_out_dim = conv_dim

        # 3x3 conv for the object hidden features
        _o_out_dim = in_channels
        self.object_convs = []
        for k in range(num_conv):
            conv = nn.Conv2d(_o_out_dim, conv_dim, kernel_size=3, stride=1, padding=1)
            self.add_module("object_conv{}".format(k + 1), conv)
            self.object_convs.append(conv)
            _o_out_dim = conv_dim
        
        # Relational networks for interactness logits prediction
        _out_dim = _o_out_dim + _p_out_dim
        self.rn_fcs = []
        for k in range(num_rn_fc):
            fc = nn.Linear(_out_dim, rn_fc_dim)
            self.add_module("rn_fc{}".format(k + 1), fc)
            self.rn_fcs.append(fc)
            _out_dim = rn_fc_dim

        # Proposal predictor
        self.person_logits = nn.Conv2d(_p_out_dim, num_cell_anchors, kernel_size=1)
        self.person_deltas = nn.Conv2d(_p_out_dim, num_cell_anchors * box_dim, kernel_size=1)
        self.object_logits = nn.Linear(_out_dim, num_cell_anchors)
        self.object_deltas = nn.Conv2d(_o_out_dim, num_cell_anchors * box_dim, kernel_size=1)

        # Weights initialization
        for layer in self.person_convs:
            weight_init.c2_msra_fill(layer)
        for layer in self.object_convs:
            weight_init.c2_msra_fill(layer)
        for layer in self.rn_fcs:
            weight_init.c2_xavier_fill(layer)
        for layer in [self.person_logits, self.person_deltas]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)
        for layer in [self.object_logits, self.object_deltas]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)

    def forward(self, features):
        """
        Shape shorthand:
            N: number of images in the minibatch
            L: number of feature maps per image on which RPN is run
            A: number of cell anchors (must be the same for all feature maps)
            Hi, Wi: height and width of the i-th feature map
            C: number of channels
            K: top K person cell anchors
                
        Args:
            features (list[Tensor]): A list of `N` tensors.
                Element i represents the feature maps of i-th images in the input data;

        Returns:
            horpn_returns (dict[str: list[Tensor]]) contains:
                
                pred_person_logits (list[Tensor]): A list of L elements.
                    Element i is a tensor of shape (N, A, Hi, Wi) representing
                    the predicted person objectness logits for anchors.
                pred_person_deltas (list[Tensor]): A list of L elements.
                    Element i is a tensor of shape (N, A*4, Hi, Wi) representing
                    the predicted "deltas" used to transform anchors to proposals.
                pred_object_logits (list[Tensor]): A list of L elements.
                    Element i is a tensor of shape (N, A, Hi, Wi) representing
                    the predicted object interactness logits for anchors.
                pred_object_deltas (list[Tensor]): A list of L elements.
                    Element i is a tensor of shape (N, A*4, Hi, Wi) representing
                    the predicted "deltas" used to transform anchors to proposals.
        """
        # Person branch
        pred_person_hidden = []
        pred_person_logits = []
        pred_person_deltas = []

        for x in features:
            for layer in self.person_convs:
                x = F.relu(layer(x))
            p_logits = self.person_logits(x)
            p_deltas = self.person_deltas(x)
            pred_person_hidden.append(x)
            pred_person_logits.append(p_logits)
            pred_person_deltas.append(p_deltas)

        # Find the cells with maximal person objectivess logits in the
        # feature maps and obtain the corresponding hidden features.
        topk_person_hidden = find_top_cells(pred_person_hidden, pred_person_logits, self.topk)

        # Object branch
        pred_object_logits = []
        pred_object_deltas = []
        for x in features:
            n, d, h, w = x.size()
            for layer in self.object_convs:
                x = F.relu(layer(x))
            o_deltas = self.object_deltas(x)
            # Object hidden features `x` is tensor with shape (N, C, Hi, Wi),
            # Person hidden features `topk_person_hidden` is a tensor with shape (N, C, K),
            # `matching_and_reshaping` reshapes them as a tensor with shape (N, Hi*Wi, K, C)
            matched_x, matched_person_hidden = matching_and_reshaping(x, topk_person_hidden)
            # Concatenate reshaped object and person hidden features
            x = torch.cat((matched_x, matched_person_hidden), -1)
            # The predicted interactness logits `o_logits` has shape (N, Hi * Wi, A)
            for i, layer in enumerate(self.rn_fcs):
                x = F.relu(layer(x)).sum(2).squeeze(2) if i == 0 else F.relu(layer(x))
            o_logits = self.object_logits(x)
            # Unreshape interactness logits from (N, Hi * Wi, A) to (N, A, Hi, Wi)
            o_logits = o_logits.permute(0, 2, 1).view(n, self.num_cell_anchors, h, w)
            
            pred_object_logits.append(o_logits)
            pred_object_deltas.append(o_deltas)

        horpn_returns = {
            "pred_person_logits": pred_person_logits,
            "pred_person_deltas": pred_person_deltas,
            "pred_object_logits": pred_object_logits,
            "pred_object_deltas": pred_object_deltas,
        }
        return horpn_returns


def find_top_cells(hidden, logits, topk):
    """
    Find the top k cell anchors with the maximal logits across all levels in the
    feature pyramid and return the corresponding hidden feature representation.

    Args:
            hidden (list[Tensor]): list of hidden feature maps
            logits (list[Tensor]): list of predicted logits

    Returns:
            topk_hidden (list[Tensor]): list of hidden features of topk cells
    """
    assert len(hidden) == len(logits), "Must have the same feature pyramid levels!"

    # 1. Find the top k cell anchors at each feature pyramid level
    topk_hidden_per_lvl = []
    topk_scores_per_lvl = []

    for hidden_ix, logits_ix in zip(hidden, logits):
        n, d, h, w = hidden_ix.size()
        cell_scores = torch.max(logits_ix, 1)[0].view(n, -1)
        topk_scores, topk_cells = torch.topk(cell_scores, topk)

        # The corresponding hidden feature of topk cell anchors
        hidden_view = hidden_ix.view(n, d, -1)
        topk_hidden = [torch.index_select(hidden_view[bx], 1, topk_cells[bx]) for bx in range(n)]
        # Issue: torch.stack() do not create new axis if n == 1 (e.g., only one image in batch).
        # Check the number of images in batch. If n==1, we manually create the new axis.
        topk_hidden = torch.stack(topk_hidden, 0) if n > 1 else torch.unsqueeze(topk_hidden[0], 0)

        topk_hidden_per_lvl.append(topk_hidden)
        topk_scores_per_lvl.append(topk_scores)

    # 2. Pick up the top k cell anchors across all levels
    topk_scores_all_lvls = torch.cat(topk_scores_per_lvl, dim=1)
    topk_hidden_all_lvls = torch.cat(topk_hidden_per_lvl, dim=2)

    _, keep = torch.topk(topk_scores_all_lvls, topk)

    return torch.stack([
                torch.index_select(topk_hidden_all_lvls[bx], 1, keep[bx])
                for bx in range(n)
            ], 0)
    

def matching_and_reshaping(o_tensor, p_tensor):
    """
    Matching the shape of object hidden feature and person hidden feature.
    
    Args:
        o_tensor (Tensor): object hidden feature has shape (N, C, Hi, Wi),
            where axes 0-3 are the number of images in batch, channels, height, and width.
        p_tensor (Tensor): person hidden feature has shape (N, C, K),
            where k is number of person anchors.

    This function contains the following operations:
        1. Reshape `o_tensor` from (N, C, Hi, Wi) to (N, C, Hi * Wi)
        2. Repeat `o_tensor` to shape (N, C, Hi * Wi, K) to match with number of person anchors.
        3. Repeat `p_tensor` to shape (N, C, Hi * Wi, K) to match with `o_tensor`.
        4. Move 1-th axis to the last. This step is for the subsequent fc layers, where
           the last axis should be channels.
    """
    n1, c1, h, w = o_tensor.size()
    n2, _, k = p_tensor.size()
    assert n1 == n2, "Must have the same number of images"

    o_tensor = o_tensor.view(n1, c1, -1).unsqueeze(3).repeat(1, 1, 1, k)
    p_tensor = p_tensor.unsqueeze(2).repeat(1, 1, w * h, 1)

    o_tensor = o_tensor.permute(0, 2, 3, 1)
    p_tensor = p_tensor.permute(0, 2, 3, 1)

    return o_tensor, p_tensor


@PROPOSAL_GENERATOR_REGISTRY.register()
class HORPN(nn.Module):
    """
    Human-Object Region Proposal Network (HORPN).
    HORPN generates region proposals for humans and their interacting objects.
    """
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()

        # fmt: off
        self.min_box_side_len     = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
        self.in_features          = cfg.MODEL.HORPN.IN_FEATURES
        self.nms_thresh           = cfg.MODEL.HORPN.NMS_THRESH
        self.batch_size_per_image = cfg.MODEL.HORPN.BATCH_SIZE_PER_IMAGE
        self.positive_fraction    = cfg.MODEL.HORPN.POSITIVE_FRACTION
        self.smooth_l1_beta       = cfg.MODEL.HORPN.SMOOTH_L1_BETA
        self.loss_weight          = cfg.MODEL.HORPN.LOSS_WEIGHT
        self.ratio_objects        = cfg.MODEL.HORPN.RATIO_OBJECTS
        # fmt: on

        # Map from self.training state to train/test settings
        self.pre_nms_topk = {
            True: cfg.MODEL.HORPN.PRE_NMS_TOPK_TRAIN,
            False: cfg.MODEL.HORPN.PRE_NMS_TOPK_TEST,
        }
        self.post_nms_topk = {
            True: cfg.MODEL.HORPN.POST_NMS_TOPK_TRAIN,
            False: cfg.MODEL.HORPN.POST_NMS_TOPK_TEST,
        }
        self.boundary_threshold = cfg.MODEL.HORPN.BOUNDARY_THRESH

        self.anchor_generator = build_anchor_generator(
            cfg, [input_shape[f] for f in self.in_features]
        )

        self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.HORPN.BBOX_REG_WEIGHTS)

        self.anchor_matcher = Matcher(
            cfg.MODEL.HORPN.IOU_THRESHOLDS,
            cfg.MODEL.HORPN.IOU_LABELS,
            allow_low_quality_matches=True
        )

        self.rpn_head = build_horpn_head(cfg, [input_shape[f] for f in self.in_features])

    def forward(self, images, features, gt_instances=None):
        """
        Args:
            images (ImageList): input images of length `N`
            features (dict[str: Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            gt_instances (list[Instances], optional): a length `N` list of `Instances`s.
                Each `Instances` stores ground-truth instances for the corresponding image.

        Returns:
            proposals: list[Instances]: contains fields "proposal_boxes", "interactness_logits"
            loss: dict[Tensor] or None
        """
        gt_boxes = [x.gt_boxes for x in gt_instances] if gt_instances is not None else None
        gt_classes = [x.gt_classes for x in gt_instances] if gt_instances is not None else None
        gt_actions = [x.gt_actions for x in gt_instances] if gt_instances is not None else None
        gt_isactive = [x.gt_isactive for x in gt_instances] if gt_instances is not None else None
        del gt_instances

        features = [features[f] for f in self.in_features]
        
        horpn_returns = self.rpn_head(features)

        anchors = self.anchor_generator(features)

        gt_anns = {
            "gt_boxes": gt_boxes,
            "gt_classes": gt_classes,
            "gt_actions": gt_actions,
            "gt_isactive": gt_isactive,
        }

        outputs = HORPNOutputs(
            self.box2box_transform,
            self.anchor_matcher,
            self.batch_size_per_image,
            self.positive_fraction,
            images,
            horpn_returns,
            anchors,
            self.boundary_threshold,
            gt_anns,
            self.smooth_l1_beta,
        )

        if self.training:
            losses = {k: v * self.loss_weight for k, v in outputs.losses().items()}
        else:
            losses = {}

        with torch.no_grad():
            # Find the top proposals by applying NMS and removing boxes that are too small.
            # The proposals are treated as fixed for approximate joint training with roi heads.
            # This approach ignores the derivative w.r.t. the proposal boxes’ coordinates that
            # are also network responses, so is approximate.
            proposals = find_top_horpn_proposals(
                outputs.predict_proposals(),
                outputs.predict_proposal_logits(),
                images,
                self.nms_thresh,
                self.pre_nms_topk[self.training],
                self.post_nms_topk[self.training],
                self.ratio_objects,
                self.min_box_side_len,
                self.training,
            )
        return proposals, losses