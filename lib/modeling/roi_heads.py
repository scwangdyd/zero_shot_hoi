import logging
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from typing import Dict, List, Optional, Tuple, Union

from detectron2.layers import ShapeSpec
from detectron2.structures import Boxes, BoxMode, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.roi_heads import ROIHeads, ROI_HEADS_REGISTRY
from detectron2.utils.logger import setup_logger

from lib.utils.interactions import Interactions
from .sampling import subsample_labels, subsample_labels_with_must_include
from .fast_rcnn import BoxOutputLayers, HoiOutputLayers
from .box_head import build_box_head, build_hoi_head

logger = setup_logger(name=__name__)

@ROI_HEADS_REGISTRY.register()
class StandardHOROIHeads(nn.Module):
    """
    It's "standard" in a sense that there is no ROI transform sharing or feature sharing
    between tasks. The cropped rois go to separate branches (boxes and HOI) directly.
    This way, it is easier to make separate abstractions for different branches.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super(StandardHOROIHeads, self).__init__()
        # fmt: off
        self.in_features                  = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.proposal_append_gt           = cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT
        self.num_classes                  = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.num_actions                  = cfg.MODEL.ROI_HEADS.NUM_ACTIONS
        self.box_batch_size_per_image     = cfg.MODEL.ROI_HEADS.BOX_BATCH_SIZE_PER_IMAGE
        self.hoi_batch_size_per_image     = cfg.MODEL.ROI_HEADS.HOI_BATCH_SIZE_PER_IMAGE
        self.box_positive_sample_fraction = cfg.MODEL.ROI_HEADS.BOX_POSITIVE_FRACTION
        self.hoi_positive_sample_fraction = cfg.MODEL.ROI_HEADS.HOI_POSITIVE_FRACTION
        # fmt: on

        # Matcher to assign box proposals to gt boxes
        self.proposal_matcher = Matcher(
            cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
            cfg.MODEL.ROI_HEADS.IOU_LABELS,
            allow_low_quality_matches=False,
        )

        self._init_box_head(cfg, input_shape)
        self._init_hoi_head(cfg, input_shape)
    
    def _init_box_head(self, cfg, input_shape):
        # fmt: off
        pooler_resolution        = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales            = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        sampling_ratio           = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type              = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        self.train_on_pred_boxes = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        # fmt: on

        # If StandardHOROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        self.box_predictor = BoxOutputLayers(cfg, self.box_head.output_shape)

    def _init_hoi_head(self, cfg, input_shape):
        self.hoi_on = cfg.MODEL.HOI_ON
        if not self.hoi_on:
            return
        # fmt: off
        pooler_resolution      = cfg.MODEL.HOI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales          = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        sampling_ratio         = cfg.MODEL.HOI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type            = cfg.MODEL.HOI_BOX_HEAD.POOLER_TYPE
        allow_person_to_person = cfg.MODEL.HOI_BOX_HEAD.ALLOW_PERSON_TO_PERSON
        # fmt: on
        self.allow_person_to_person = allow_person_to_person

        # If StandardHOROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.hoi_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.hoi_head = build_hoi_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        self.hoi_predictor = HoiOutputLayers(cfg, self.hoi_head.output_shape)

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[Tuple[List[Instances]], Dict[str, torch.Tensor]]:
        """
        Args:
            images (ImageList):
            features (dict[str,Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            proposals (list[Instances]): length `N` list of `Instances`. The i-th
                `Instances` contains object proposals for the i-th input image,
                with fields "proposal_boxes" and "interactness_logits".
            targets (list[Instances], optional): length `N` list of `Instances`. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.
                It may have the following fields:

                - gt_boxes: the bounding box of each instance.
                - gt_classes: the label for each instance with a category ranging in [0, #class].
                - gt_actions: the ground-truth binary matrix for each instance, indicating the
                    interactions with other instances within the image.

        Returns:
            Tuple[Instances]: (box instances, hoi instances). Each `instances` is length `N`
                list containing the detected instances.
                Returned during inference only. May be [] during training.

            dict[str->Tensor]: mapping from a named loss to a tensor storing the loss.
                Used during training only.
        """
        del images
        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            losses = self._forward_box(features, proposals)
            losses.update(self._forward_hoi(features, proposals))
            return proposals, losses
        else:
            pred_box_instances = self._forward_box(features, proposals)
            pred_hoi_instances = self._forward_hoi(features, pred_box_instances)
            pred_instances = (pred_box_instances, pred_hoi_instances)
            return pred_instances, {}

    def _forward_box(
        self, features: Dict[str, torch.Tensor], proposals: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with their matching
                ground truth. Each has fields "proposal_boxes", and "interactness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances. Each has fields
                "pred_boxes", "pred_classes", "scores"
        """
        features = [features[f] for f in self.in_features]
        # boxes forward
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        box_predictions = self.box_predictor(box_features)
        # Reweight box predictions with the interactness score from HORPN
        box_predictions = self._reweight_box_given_proposal_scores(box_predictions, proposals)
        del box_features
        
        if self.training:
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        box_predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            
            losses = self.box_predictor.losses(box_predictions, proposals)
            return losses
        else:
            pred_instances, kept_idxs = self.box_predictor.inference(box_predictions, proposals)
            return pred_instances

    def _forward_hoi(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the interaction prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): 
                At training, the per-image object proposals with matching ground truth. Each has
                    fields "proposal_boxes", and "interactness_logits", "gt_classes", "gt_actions".
                At inference, the per-image predicted box instances from box head. Each has fields
                    "pred_boxes", "pred_classes", "scores"

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted hoi instances. Each has fields
                "person_boxes", "object_boxes", "object_classes", "action_classes", "scores"
        """
        if not self.hoi_on:
            return {} if self.training else []

        hopairs = self.construct_hopairs(instances)
        
        features = [features[f] for f in self.in_features]
        union_features  = self.hoi_pooler(features, [x.union_boxes  for x in hopairs])
        person_features = self.hoi_pooler(features, [x.person_boxes for x in hopairs])
        object_features = self.hoi_pooler(features, [x.object_boxes for x in hopairs])

        union_features  = self.hoi_head(union_features)
        person_features = self.hoi_head(person_features)
        object_features = self.hoi_head(object_features)
        
        hoi_predictions = self.hoi_predictor(union_features, person_features, object_features)

        del union_features, person_features, object_features, features

        if self.training:
            losses = self.hoi_predictor.losses(hoi_predictions, hopairs)
            return losses
        else:
            pred_interactions = self.hoi_predictor.inference(hoi_predictions, hopairs)
            return pred_interactions

    def _reweight_box_given_proposal_scores(
        self, predictions: Tuple[torch.Tensor], proposals: List[Instances]
    ) -> Tuple[torch.Tensor]:
        """
        Reweight the box prediction scores with their proposal interactness scores.

        Args:
            predictions (tuple[Tensor]): the per-image box predictions.
            proposals (list[Instances]): the per-image object proposals with "interactness scores".

        Returns:
            predictions (tuple[Tensor]): the per-image box predictions.
        """
        proposal_scores = torch.cat([torch.sigmoid(x.interactness_logits) for x in proposals])
        foreground_reweight = proposal_scores.unsqueeze(1).repeat(1, self.num_classes + 1)
        background_reweight = 1 - proposal_scores

        box_scores = predictions[0]
        box_deltas = predictions[1]

        box_scores = F.softmax(box_scores, dim=-1)
        box_scores = box_scores * foreground_reweight
        box_scores[:, self.num_classes] = box_scores[:, self.num_classes] + background_reweight

        return (box_scores, box_deltas)

    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances]
    ) -> List[Instances]:
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.box_batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.box_positive_sample_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_actions", that's included in `targets`.
        """
        gt_boxes = [x.gt_boxes for x in targets]
        gt_classes = [x.gt_classes for x in targets]
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts the proposals will be
        # low quality due to random initialization. It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used as positive examples
        # for the second stage components (box head, cls head). Adding the gt boxes to the set of
        # proposals ensures that the second stage components will have some positive examples from
        # the start of training. For RPN, this augmentation improves convergence and empirically
        # improves box AP on COCO by about 0.5 points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, gt_classes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs,
                matched_labels,
                targets_per_image.gt_classes,
                proposals_per_image.is_person
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                proposals_per_image.set("matched_idxs", sampled_targets)
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt


    def _sample_proposals(
        self,
        matched_idxs: torch.Tensor,
        matched_labels: torch.Tensor,
        gt_classes: torch.Tensor,
        must_include_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.
            must_include_mask (Tensor): a mask with values:
                * 1: the proposal is from person branch in HORPN.
                * 0: the proposal is from object branch in HORPN.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        # Because there are many more objects than person in generated proposals
        # (ratio_object = 0.95), the original `subsample_labels` used in Detectron2 may miss pesron
        # instances, causing empty bugs in the subsequent HOI head since it cannot find valid
        # person-object pairs. Here we have to avoid this case.
        sampled_fg_idxs, sampled_bg_idxs = subsample_labels_with_must_include(
            gt_classes,
            self.box_batch_size_per_image,
            self.box_positive_sample_fraction,
            self.num_classes,
            must_include_mask=must_include_mask,
            num_must_include=1
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]


    @torch.no_grad()
    def construct_hopairs(self, instances: List[Instances]) -> List[Instances]:
        """
        Prepare person-object pairs to be used to train HOI heads.
        At training, it returns union regions of person-object proposals and assigns
            training labels. It returns ``self.hoi_batch_size_per_image`` random samples
            from pesron-object pairs, with a fraction of positives that is no larger than
            ``self.hoi_positive_sample_fraction``.
        At inference, it returns union regions of predicted person boxes and object boxes.

        Args:
            instances (list[Instances]):
                At training, proposals_with_gt. See ``self.label_and_sample_proposals``
                At inference, predicted box instances. See ``self._forward_box``

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the human-object pairs.
                Each `Instances` has the following fields:

                - union_boxes: the union region of person boxes and object boxes
                - person_boxes: person boxes in a matched sequences with union_boxes
                - object_boxes: object boxes in a matched sequences with union_boxes
                - gt_actions: the ground-truth actions that the pair is assigned.
                    Used for training HOI head.
                - person_box_scores: person box scores from box instances. Used at inference.
                - object_box_scores: object box scores from box instances. Used at inference.
                - object_box_classes: predicted box classes from box instances. Used at inference.
        """
        hopairs = []
        for instances_per_image in instances:
            if self.training:
                # Proposals generated from person branch in HORPN will be seen as person boxes;
                # Proposals generated from object branch in HORPN will be object boxes.
                boxes = instances_per_image.proposal_boxes
                person_idxs = (instances_per_image.is_person == 1).nonzero().squeeze(1)
                object_idxs = (instances_per_image.is_person == 0).nonzero().squeeze(1)
            else:
                # At inference, split person/object boxes based on predicted classes by box head
                boxes = instances_per_image.pred_boxes
                person_idxs = torch.nonzero(instances_per_image.pred_classes == 0).squeeze(1)
                object_idxs = torch.nonzero(instances_per_image.pred_classes >  0).squeeze(1)
            
            if self.allow_person_to_person:
                # Allow person to person interactions. Then all boxes will be used.
                object_idxs = torch.arange(len(instances_per_image), device=object_idxs.device)

            num_pboxes, num_oboxes = person_idxs.numel(), object_idxs.numel()

            union_boxes = _pairwise_union_regions(boxes[person_idxs], boxes[object_idxs])
            # Indexing person/object boxes in a matched order.
            person_idxs = person_idxs[:, None].repeat(1, num_oboxes).flatten()
            object_idxs = object_idxs[None, :].repeat(num_pboxes, 1).flatten()
            # Remove self-to-self interaction.
            keep = (person_idxs != object_idxs).nonzero().squeeze(1)
            union_boxes = union_boxes[keep]
            person_idxs = person_idxs[keep]
            object_idxs = object_idxs[keep]

            hopairs_per_image = Instances(instances_per_image.image_size)
            hopairs_per_image.union_boxes = union_boxes
            hopairs_per_image.person_boxes = boxes[person_idxs]
            hopairs_per_image.object_boxes = boxes[object_idxs]

            if self.training:
                # `person_idxs` and `object_idxs` are used in self.label_and_sample_hopairs()
                hopairs_per_image.person_idxs = person_idxs
                hopairs_per_image.object_idxs = object_idxs
            else:
                hopairs_per_image.person_box_scores = instances_per_image.scores[person_idxs]
                hopairs_per_image.object_box_scores = instances_per_image.scores[object_idxs]
                hopairs_per_image.object_box_classes = instances_per_image.pred_classes[object_idxs]
            
            hopairs.append(hopairs_per_image)

        if self.training:
            hopairs = self.label_and_sample_hopairs(hopairs, instances)

        return hopairs

    @torch.no_grad()
    def label_and_sample_hopairs(
        self, hopairs: List[Instances], proposals: List[Instances]
    ) -> List[Instances]:
        """
        Sample person-object pairs to be used to train HOI heads and assign labels.

        Args:
            See `StandardHOROIHeads.construct_hopairs`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the sampled person-object pairs
                and corresponding action labels.
        """
        hopairs_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for hopairs_per_image, proposals_per_image in zip(hopairs, proposals):
            # `gt_actions` is a tensor with shape (N, M, K), where axes 0-2 represents person
            # proposals, object proposals, and actions. We index `gt_actions` using person
            # indices and object indices to get the labels for each person-object pair.
            gt_actions = proposals_per_image.gt_actions[hopairs_per_image.person_idxs]
            matched_idxs = proposals_per_image.matched_idxs[hopairs_per_image.object_idxs]
            gt_actions = torch.stack([gt_actions.take(i, j) for i, j in enumerate(matched_idxs)])

            gt_classes = torch.stack([
                proposals_per_image.gt_classes[hopairs_per_image.person_idxs],
                proposals_per_image.gt_classes[hopairs_per_image.object_idxs]
            ], dim=1)

            gt_isactive = torch.stack([
                proposals_per_image.gt_isactive[hopairs_per_image.person_idxs],
                proposals_per_image.gt_isactive[hopairs_per_image.object_idxs]
            ], dim=1)

            # Valid pairs are defined as interactions among foreground proposals. Pairs among
            # background-background proposals or foreground-background proposals will be treated
            # as invalid in the subsampling. Note that, if two proposals in pair are both
            # foreground but they are no interacting with each other, we still see them as valid
            # samples, since those hard-negative samples can improve the action prediction.
            labels = torch.sum(gt_actions, dim=1)
            valids = (gt_classes[:, 0] < self.num_classes) & (gt_classes[:, 1] < self.num_classes)
            # label == 1: interacting;
            # label == 0: non-interacting;
            # label == self.num_actions: background (invalid)
            labels[~valids] = self.num_actions
            gt_actions[~valids] = 0
            
            sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
                labels=labels,
                num_samples=self.hoi_batch_size_per_image,
                positive_fraction=self.hoi_positive_sample_fraction,
                bg_label=self.num_actions
            )
            sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)

            hopairs_per_image = hopairs_per_image[sampled_idxs]
            hopairs_per_image.gt_actions = gt_actions[sampled_idxs]
            hopairs_per_image.gt_classes = gt_classes[sampled_idxs]
            hopairs_per_image.gt_isactive = gt_isactive[sampled_idxs]
            
            num_fg_samples.append(sampled_fg_idxs.numel())
            num_bg_samples.append(sampled_bg_idxs.numel())
            hopairs_with_gt.append(hopairs_per_image)
    
        # Log the number of fg/bg samples that are selected for training HOI heads
        storage = get_event_storage()
        storage.put_scalar("hoi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("hoi_head/num_bg_samples", np.mean(num_bg_samples))

        return hopairs_with_gt


def _pairwise_union_regions(boxes1: Boxes, boxes2: Boxes) -> Boxes:
    """
    Given two lists of boxes of size N and M, compute the union regions between
    all N x M pairs of boxes. The box order must be (xmin, ymin, xmax, ymax).

    Args:
        boxes1, boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
        unions: Boxes
    """
    boxes1 = boxes1.tensor
    boxes2 = boxes2.tensor
    
    X1 = torch.min(boxes1[:, None, 0], boxes2[:, 0]).flatten()
    Y1 = torch.min(boxes1[:, None, 1], boxes2[:, 1]).flatten()
    X2 = torch.max(boxes1[:, None, 2], boxes2[:, 2]).flatten()
    Y2 = torch.max(boxes1[:, None, 3], boxes2[:, 3]).flatten()

    unions = torch.stack([X1, Y1, X2, Y2], dim=1)
    unions = Boxes(unions) # BoxMode.XYXY_ABS

    return unions


def add_ground_truth_to_proposals(gt_boxes, gt_classes, proposals):
    """
    Call `add_ground_truth_to_proposals_single_image` for all images.

    Args:
        gt_boxes(list[Boxes]): list of N elements. Element i is a Boxes
            representing the gound-truth for image i.
        proposals (list[Instances]): list of N elements. Element i is a Instances
            representing the proposals for image i.

    Returns:
        list[Instances]: list of N Instances. Each is the proposals for the image,
            with field "proposal_boxes" and "interactness_logits".
    """
    assert gt_boxes is not None

    assert len(proposals) == len(gt_boxes)
    if len(proposals) == 0:
        return proposals

    return [
        add_ground_truth_to_proposals_single_image(gt_boxes_i, gt_classes_i, proposals_i)
        for gt_boxes_i, gt_classes_i, proposals_i in zip(gt_boxes, gt_classes, proposals)
    ]


def add_ground_truth_to_proposals_single_image(gt_boxes, gt_classes, proposals):
    """
    Augment `proposals` with ground-truth boxes from `gt_boxes`.

    Args:
        Same as `add_ground_truth_to_proposals`, but with gt_boxes and proposals
        per image.

    Returns:
        Same as `add_ground_truth_to_proposals`, but for only one image.
    """
    device = proposals.interactness_logits.device
    # Concatenating gt_boxes with proposals requires them to have the same fields
    # Assign all ground-truth boxes an interactness logit corresponding to P(object) \approx 1.
    gt_logit_value = math.log((1.0 - 1e-10) / (1 - (1.0 - 1e-10)))
    gt_logits = gt_logit_value * torch.ones(len(gt_boxes), device=device)
    
    is_person = torch.full((len(gt_boxes), ), 0., device=device)
    is_person[(gt_classes == 0).nonzero().squeeze(1)] = 1.

    gt_proposal = Instances(proposals.image_size)
    gt_proposal.proposal_boxes = gt_boxes
    gt_proposal.interactness_logits = gt_logits
    gt_proposal.is_person = is_person
    new_proposals = Instances.cat([proposals, gt_proposal])

    return new_proposals