import itertools
import logging
import numpy as np
import torch
import torch.nn.functional as F
from fvcore.nn import smooth_l1_loss

from detectron2.layers import batched_nms, cat
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.memory import retry_if_cuda_oom

from detectron2.modeling.sampling import subsample_labels

logger = logging.getLogger(__name__)


"""
Shape shorthand in this module:

    N: number of images in the minibatch
    L: number of feature maps per image on which RPN is run
    A: number of cell anchors (must be the same for all feature maps)
    Hi, Wi: height and width of the i-th feature map
    4: size of the box parameterization

Naming convention:

    interactness: refers to the binary classification of an anchor as object vs. not
    object.

    deltas: refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box2box
    transform (see :class:`box_regression.Box2BoxTransform`).

    pred_interactness_logits: predicted interactness scores in [-inf, +inf]; use
        sigmoid(pred_interactness_logits) to estimate P(object).

    gt_interactness_logits: ground-truth binary classification labels for interactness

    pred_anchor_deltas: predicted box2box transform deltas

    gt_anchor_deltas: ground-truth box2box transform deltas
"""


def find_top_horpn_proposals(
    proposals,
    pred_proposal_logits,
    images,
    nms_thresh,
    pre_nms_topk,
    post_nms_topk,
    ratio_objects,
    min_box_side_len,
    training,
):
    """
    # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
    For each feature map, select the `pre_nms_topk` highest scoring proposals, apply NMS,
    clip proposals, and remove small boxes. Return the `post_nms_topk` highest scoring
    proposals among all the feature maps if `training` is True, otherwise, returns the
    highest `post_nms_topk` scoring proposals for each feature map.

    Args:
        proposals (dict[str: list[Tensor]]): All proposal predictions on the feature maps.
            "person": A list of L tensors. Tensor i has shape (N, Hi*Wi*A, 4).
            "object": A list of L tensors. Tensor i has shape (N, Hi*Wi*A, 4).
        pred_proposal_logits (dict[str: list[Tensor]]):
            "person": A list of L tensors. Tensor i has shape (N, Hi*Wi*A).
            "object": A list of L tensors. Tensor i has shape (N, Hi*Wi*A).
        images (ImageList): Input images as an :class:`ImageList`.
        nms_thresh (float): IoU threshold to use for NMS
        pre_nms_topk (int): number of top k scoring proposals to keep before applying NMS.
            When RPN is run on multiple feature maps (as in FPN) this number is per feature map.
        post_nms_topk (int): number of top k scoring proposals to keep after applying NMS.
            When RPN is run on multiple feature maps (as in FPN) this number is total over all maps.
        ratio_objects (float):
            ratio of object proposals and total proposals after nms over all feature maps.
        min_box_side_len (float):
            minimum proposal box side length in pixels (absolute units wrt input images).
        training (bool): True if proposals are to be used in training, otherwise False.
            This arg exists only to support a legacy bug; look for the "NB: Legacy bug ..." comment.

    Returns:
        proposals (list[Instances]): list of N Instances. The i-th Instances stores post_nms_topk
            object proposals for image i, sorted by their interactness score in descending order.
    """
    person_proposals = proposals["person"]
    object_proposals = proposals["object"]
    pred_person_proposal_logits = pred_proposal_logits["person"]
    pred_object_proposal_logits = pred_proposal_logits["object"]

    image_sizes = images.image_sizes  # in (h, w) order
    num_images = len(image_sizes)
    device = person_proposals[0].device

    object_post_nms_topk = int(post_nms_topk * ratio_objects)
    person_post_nms_topk = post_nms_topk - object_post_nms_topk

    # 1. Select top-k anchor for every level and every image
    person_topk_scores = []  # #lvl Tensor, each of shape N x topk
    person_topk_proposals = []
    person_level_ids = []  # #lvl Tensor, each of shape (topk,)
    batch_idx = torch.arange(num_images, device=device)
    for level_id, proposals_i, logits_i in zip(
        itertools.count(), person_proposals, pred_person_proposal_logits
    ):
        Hi_Wi_A = logits_i.shape[1]
        num_proposals_i = min(pre_nms_topk, Hi_Wi_A)

        # sort is faster than topk (https://github.com/pytorch/pytorch/issues/22812)
        # topk_scores_i, topk_idx = logits_i.topk(num_proposals_i, dim=1)
        logits_i, idx = logits_i.sort(descending=True, dim=1)
        topk_scores_i = logits_i[batch_idx, :num_proposals_i]
        topk_idx = idx[batch_idx, :num_proposals_i]

        # each is N x topk
        topk_proposals_i = proposals_i[batch_idx[:, None], topk_idx]  # N x topk x 4

        person_topk_proposals.append(topk_proposals_i)
        person_topk_scores.append(topk_scores_i)
        person_level_ids.append(
            torch.full((num_proposals_i,), level_id, dtype=torch.int64, device=device)
        )

    object_topk_scores = []  # #lvl Tensor, each of shape N x topk
    object_topk_proposals = []
    object_level_ids = []  # #lvl Tensor, each of shape (topk,)
    batch_idx = torch.arange(num_images, device=device)
    for level_id, proposals_i, logits_i in zip(
        itertools.count(), object_proposals, pred_object_proposal_logits
    ):
        Hi_Wi_A = logits_i.shape[1]
        num_proposals_i = min(pre_nms_topk, Hi_Wi_A)

        # sort is faster than topk (https://github.com/pytorch/pytorch/issues/22812)
        # topk_scores_i, topk_idx = logits_i.topk(num_proposals_i, dim=1)
        logits_i, idx = logits_i.sort(descending=True, dim=1)
        topk_scores_i = logits_i[batch_idx, :num_proposals_i]
        topk_idx = idx[batch_idx, :num_proposals_i]

        # each is N x topk
        topk_proposals_i = proposals_i[batch_idx[:, None], topk_idx]  # N x topk x 4

        object_topk_proposals.append(topk_proposals_i)
        object_topk_scores.append(topk_scores_i)
        object_level_ids.append(
            torch.full((num_proposals_i,), level_id, dtype=torch.int64, device=device)
        )

    # 2. Concat all levels together
    person_topk_scores = cat(person_topk_scores, dim=1)
    object_topk_scores = cat(object_topk_scores, dim=1)
    person_topk_proposals = cat(person_topk_proposals, dim=1)
    object_topk_proposals = cat(object_topk_proposals, dim=1)
    person_level_ids = cat(person_level_ids, dim=0)
    object_level_ids = cat(object_level_ids, dim=0)

    # 3. For each image, run a per-level NMS, and choose topk results.
    results = []
    for n, image_size in enumerate(image_sizes):
        person_boxes = Boxes(person_topk_proposals[n])
        person_scores_per_img = person_topk_scores[n]
        person_lvl = person_level_ids

        object_boxes = Boxes(object_topk_proposals[n])
        object_scores_per_img = object_topk_scores[n]
        object_lvl = object_level_ids

        valid_mask = torch.isfinite(person_boxes.tensor).all(dim=1) & torch.isfinite(person_scores_per_img)
        if not valid_mask.all():
            if training:
                raise FloatingPointError(
                    "Predicted person boxes or scores contain Inf/NaN. Training has diverged."
                )
            person_boxes = person_boxes[valid_mask]
            person_scores_per_img = person_scores_per_img[valid_mask]
            person_lvl = person_lvl[valid_mask]

        valid_mask = torch.isfinite(object_boxes.tensor).all(dim=1) & torch.isfinite(object_scores_per_img)
        if not valid_mask.all():
            if training:
                raise FloatingPointError(
                    "Predicted object boxes or scores contain Inf/NaN. Training has diverged."
                )
            object_boxes = object_boxes[valid_mask]
            object_scores_per_img = object_scores_per_img[valid_mask]
            object_lvl = object_lvl[valid_mask]

        person_boxes.clip(image_size)
        object_boxes.clip(image_size)

        # filter empty boxes
        keep = person_boxes.nonempty(threshold=min_box_side_len)
        if keep.sum().item() != len(person_boxes):
            person_boxes = person_boxes[keep]
            person_scores_per_img = person_scores_per_img[keep]
            person_lvl = person_lvl[keep]

        keep = object_boxes.nonempty(threshold=min_box_side_len)
        if keep.sum().item() != len(object_boxes):
            object_boxes = object_boxes[keep]
            object_scores_per_img = object_scores_per_img[keep]
            object_lvl = object_lvl[keep]

        person_keep = batched_nms(person_boxes.tensor, person_scores_per_img.clone(), person_lvl, nms_thresh)
        object_keep = batched_nms(object_boxes.tensor, object_scores_per_img.clone(), object_lvl, nms_thresh)
        # In Detectron1, there was different behavior during training vs. testing.
        # (https://github.com/facebookresearch/Detectron/issues/459)
        # During training, topk is over the proposals from *all* images in the training batch.
        # During testing, it is over the proposals for each image separately.
        # As a result, the training behavior becomes batch-dependent,
        # and the configuration "POST_NMS_TOPK_TRAIN" end up relying on the batch size.
        # This bug is addressed in Detectron2 to make the behavior independent of batch size.
        person_keep = person_keep[:person_post_nms_topk]  # keep is already sorted
        object_keep = object_keep[:object_post_nms_topk]

        person_boxes = person_boxes[person_keep]
        object_boxes = object_boxes[object_keep]

        person_scores_per_img = person_scores_per_img[person_keep]
        object_scores_per_img = object_scores_per_img[object_keep]

        is_person = torch.zeros(len(person_keep) + len(object_keep), device=device)
        is_person[:len(person_keep)] = 1.

        res = Instances(image_size)
        res.proposal_boxes = Boxes.cat([person_boxes, object_boxes])
        res.interactness_logits = torch.cat([person_scores_per_img, object_scores_per_img])
        res.is_person = is_person
        results.append(res)

    return results


def horpn_losses(
    gt_person_logits,
    gt_object_logits,
    gt_person_deltas,
    gt_object_deltas,
    pred_person_logits,
    pred_object_logits,
    pred_person_deltas,
    pred_object_deltas,
    smooth_l1_beta,
):
    """
    Args:
        gt_*_logits (Tensor): shape (N,), each element in {-1, 0, 1} representing
            ground-truth interactness labels with: -1 = ignore; 0 = not object; 1 = object.
        gt_*_deltas (Tensor): shape (N, box_dim), row i represents ground-truth
            box2box transform targets (dx, dy, dw, dh) or (dx, dy, dw, dh, da) that
            map anchor i to its matched ground-truth box.
        pred_*_logits (Tensor): shape (N,),
            each element is a predicted interactness logit.
        pred_*_deltas (Tensor): shape (N, box_dim), each row is a predicted box2box
            transform (dx, dy, dw, dh) or (dx, dy, dw, dh, da)
        smooth_l1_beta (float): The transition point between L1 and L2 loss in
            the smooth L1 loss function. When set to 0, the loss becomes L1. When
            set to +inf, the loss becomes constant 0.

    Returns:
        interactness_loss, localization_loss, both unnormalized (summed over samples).
    """

    person_pos_masks = gt_person_logits == 1
    object_pos_masks = gt_object_logits == 1

    person_localization_loss = smooth_l1_loss(
        pred_person_deltas[person_pos_masks],
        gt_person_deltas[person_pos_masks],
        smooth_l1_beta,
        reduction="sum"
    )
    object_localization_loss = smooth_l1_loss(
        pred_object_deltas[object_pos_masks],
        gt_object_deltas[object_pos_masks],
        smooth_l1_beta,
        reduction="sum"
    )

    person_valid_masks = gt_person_logits >= 0
    object_valid_masks = gt_object_logits >= 0

    person_interactness_loss = F.binary_cross_entropy_with_logits(
        pred_person_logits[person_valid_masks],
        gt_person_logits[person_valid_masks].to(torch.float32),
        reduction="sum",
    )
    object_interactness_loss = F.binary_cross_entropy_with_logits(
        pred_object_logits[object_valid_masks],
        gt_object_logits[object_valid_masks].to(torch.float32),
        reduction="sum",
    )

    return {
        "person_localization_loss": person_localization_loss,
        "object_localization_loss": object_localization_loss,
        "person_interactness_loss": person_interactness_loss,
        "object_interactness_loss": object_interactness_loss,
    }


class HORPNOutputs(object):
    def __init__(
        self,
        box2box_transform,
        anchor_matcher,
        batch_size_per_image,
        positive_fraction,
        images,
        horpn_returns,
        anchors,
        boundary_threshold=0,
        gt_anns=None,
        smooth_l1_beta=0.0,
    ):
        """
        Args:
            box2box_transform (Box2BoxTransform):
                :class:`Box2BoxTransform` instance for anchor-proposal transformations.
            anchor_matcher (Matcher):
                :class:`Matcher` instance for matching anchors to ground-truth boxes;
                used to determine training labels.
            batch_size_per_image (int):
                number of proposals to sample when training
            positive_fraction (float):
                target fraction of sampled proposals that should be positive
            images (ImageList):
                :class:`ImageList` instance representing N input images
            horpn_returns (dict[str: list[Tensor]]): returns from HORPN head
                see :class:`StandardHORPNHead.forword()`
            anchors (list[list[Boxes]]): A list of N elements. Each element is a list of L Boxes.
                The Boxes at (n, l) stores the entire anchor array for feature map l in image n
                (i.e. the cell anchors repeated over all locations in feature map (n, l)).
            boundary_threshold (int): if >= 0, then anchors that extend beyond the image
                boundary by more than boundary_thresh are not used in training. Set to a very large
                number or < 0 to disable this behavior. Only needed in training.
            gt_anns (dict, optional): Dictionary of boxes annotations.
                the ground-truth ("gt") boxes for image i.
            smooth_l1_beta (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.
        """
        self.box2box_transform = box2box_transform
        self.anchor_matcher = anchor_matcher
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

        self.pred_person_logits = horpn_returns["pred_person_logits"]
        self.pred_person_deltas = horpn_returns["pred_person_deltas"]
        self.pred_object_logits = horpn_returns["pred_object_logits"]
        self.pred_object_deltas = horpn_returns["pred_object_deltas"]

        self.anchors = anchors
        self.gt_anns = gt_anns
        self.num_feature_maps = len(self.pred_object_logits)
        self.num_images = len(images)
        self.image_sizes = images.image_sizes
        self.boundary_threshold = boundary_threshold
        self.smooth_l1_beta = smooth_l1_beta

    def _get_ground_truth(self):
        """
        Returns:
            gt_person_logits: list of N tensors. Tensor i is a vector whose length is the
                total number of anchors in image i (i.e., len(anchors[i])). Label values are
                in {-1, 0, 1}, with meanings: -1 = ignore; 0 = negative class; 1 = positive class.
            gt_object_logits: list of N tensors. Tensor i is a vector whose length is the
                total number of anchors in image i (i.e., len(anchors[i])). Label values are
                in {-1, 0, 1}, with meanings: -1 = ignore; 0 = negative class; 1 = positive class.
            gt_person_deltas: list of N tensors. Tensor i has shape (len(anchors[i]), 4).
            gt_object_deltas: list of N tensors. Tensor i has shape (len(anchors[i]), 4).
        """
        gt_person_logits = []
        gt_object_logits = []
        gt_person_deltas = []
        gt_object_deltas = []

        # Split gt_anns based on person/object and if it is active
        gt_boxes = self.gt_anns["gt_boxes"]
        gt_classes = self.gt_anns["gt_classes"]
        gt_isactive = self.gt_anns["gt_isactive"]

        # Concatenate anchors from all feature maps into a single Boxes per image
        anchors = [Boxes.cat(anchors_i) for anchors_i in self.anchors]

        assert len(self.image_sizes) == len(anchors) == len(anchors) == len(gt_boxes)

        for img_i in range(len(self.image_sizes)):
            # image_size_i: (h, w) for the i-th image
            # anchors_i: anchors for i-th image
            # gt_boxes_i: ground-truth boxes for i-th image
            # gt_isactive_i: if objects are interacting with persons in i-th image
            # gt_classes_i: ground-truth class annotation for i-th image
            image_size_i  = self.image_sizes[img_i]
            anchors_i     = anchors[img_i]
            gt_boxes_i    = gt_boxes[img_i]
            gt_classes_i  = gt_classes[img_i]
            gt_isactive_i = gt_isactive[img_i]
            n_boxes_i     = len(gt_boxes_i)
            assert n_boxes_i == len(gt_isactive_i) == len(gt_classes_i)

            gt_person_boxes_i = [
                gt_boxes_i[j] for j in range(n_boxes_i) \
                if gt_isactive_i[j] > 0 and gt_classes_i[j] == 0
            ]
            gt_object_boxes_i = [
                gt_boxes_i[j] for j in range(n_boxes_i) \
                if gt_isactive_i[j] > 0 and gt_classes_i[j] > 0
            ]

            num_person_gts, num_object_gts = len(gt_person_boxes_i), len(gt_object_boxes_i)

            if num_person_gts == 0:
                # These values won't be used anyway since the anchor is labeled as background
                gt_person_deltas_i = torch.zeros_like(anchors_i.tensor)
                gt_person_logits_i = torch.zeros(
                    (len(anchors_i.tensor),), dtype=torch.int8, device=gt_boxes_i.device
                )
            else:
                gt_person_boxes_i = Boxes.cat(gt_person_boxes_i)
                person_match_quality = retry_if_cuda_oom(pairwise_iou)(gt_person_boxes_i, anchors_i)
                matched_person_idxs, gt_person_logits_i = retry_if_cuda_oom(self.anchor_matcher)(person_match_quality)
                # Matching is memory-expensive and may result in CPU tensors. But the result is small
                gt_person_logits_i = gt_person_logits_i.to(device=gt_boxes_i.device)
                del person_match_quality

                if self.boundary_threshold >= 0:
                    # Discard anchors that go out of the boundaries of the image
                    anchors_inside_image = anchors_i.inside_box(image_size_i, self.boundary_threshold)
                    gt_person_logits_i[~anchors_inside_image] = -1

                matched_gt_person_boxes = gt_person_boxes_i[matched_person_idxs]
                gt_person_deltas_i = self.box2box_transform.get_deltas(
                    anchors_i.tensor, matched_gt_person_boxes.tensor
                )

            gt_person_logits.append(gt_person_logits_i)
            gt_person_deltas.append(gt_person_deltas_i)


            if num_object_gts == 0:
                # These values won't be used anyway since the anchor is labeled as background
                gt_object_deltas_i = torch.zeros_like(anchors_i.tensor)
                gt_object_logits_i = torch.zeros(
                    (len(anchors_i.tensor),), dtype=torch.int8, device=gt_boxes_i.device
                )
            else:
                gt_object_boxes_i = Boxes.cat(gt_object_boxes_i)
                object_match_quality = retry_if_cuda_oom(pairwise_iou)(gt_object_boxes_i, anchors_i)
                matched_object_idxs, gt_object_logits_i = retry_if_cuda_oom(self.anchor_matcher)(object_match_quality)
                # Matching is memory-expensive and may result in CPU tensors. But the result is small
                gt_object_logits_i = gt_object_logits_i.to(device=gt_boxes_i.device)
                del object_match_quality
                
                if self.boundary_threshold >= 0:
                    # Discard anchors that go out of the boundaries of the image
                    anchors_inside_image = anchors_i.inside_box(image_size_i, self.boundary_threshold)
                    t_object_logits_i[~anchors_inside_image] = -1
                
                matched_gt_object_boxes = gt_object_boxes_i[matched_object_idxs]
                gt_object_deltas_i = self.box2box_transform.get_deltas(
                    anchors_i.tensor, matched_gt_object_boxes.tensor
                )

            gt_object_logits.append(gt_object_logits_i)
            gt_object_deltas.append(gt_object_deltas_i)

        gts = {
            "gt_person_logits": gt_person_logits,
            "gt_person_deltas": gt_person_deltas,
            "gt_object_logits": gt_object_logits,
            "gt_object_deltas": gt_object_deltas
        }

        return gts

    def losses(self):
        """
        Return the losses from a set of HORPN predictions and their associated ground-truth.

        Returns:
            dict[loss name -> loss value]: A dict mapping from loss name to loss value.
                Loss names are: `loss_rpn_cls` for interactness classification and
                `loss_rpn_loc` for proposal localization.
        """

        def resample(label):
            """
            Randomly sample a subset of positive and negative examples by overwriting
            the label vector to the ignore value (-1) for all elements that are not
            included in the sample.
            """
            pos_idx, neg_idx = subsample_labels(
                label, self.batch_size_per_image, self.positive_fraction, 0
            )
            # Fill with the ignore label (-1), then set positive and negative labels
            label.fill_(-1)
            label.scatter_(0, pos_idx, 1)
            label.scatter_(0, neg_idx, 0)
            return label

        gts = self._get_ground_truth()
        """
        gt_person_logits: list of N tensors corresponding to person.
            Tensor i is a vector whose length is the total number of anchors
            in image i (i.e., len(anchors[i]))
        gt_person_deltas: list of N tensors corresponding to person.
            Tensor i has shape (len(anchors[i]), B), where B is the box dimension
        gt_object_logits: list of N tensors corresponding to objects.
            Tensor i is a vector whose length is the total number of anchors
            in image i (i.e., len(anchors[i]))
        gt_object_deltas: list of N tensors corresponding to objects.
            Tensor i has shape (len(anchors[i]), B), where B is the box dimension
        """
        # Collect all interactness (person)/interactiveness (object) labels and
        # delta targets over feature maps and images.
        # The final ordering is L, N, H, W, A from slowest to fastest axis.
        num_anchors_per_map = [np.prod(x.shape[1:]) for x in self.pred_person_logits]
        num_anchors_per_image = sum(num_anchors_per_map)

        # Stack to: (N, num_anchors_per_image)
        gt_person_logits = torch.stack([resample(label) for label in gts["gt_person_logits"]], dim=0)
        gt_object_logits = torch.stack([resample(label) for label in gts["gt_object_logits"]], dim=0)

        # Log the number of positive/negative anchors per-image that's used in training
        num_pos_person_anchors = (gt_person_logits == 1).sum().item()
        num_neg_person_anchors = (gt_person_logits == 0).sum().item()

        num_pos_object_anchors = (gt_object_logits == 1).sum().item()
        num_neg_object_anchors = (gt_object_logits == 0).sum().item()

        storage = get_event_storage()
        storage.put_scalar("rpn/num_pos_person_anchors", num_pos_person_anchors / self.num_images)
        storage.put_scalar("rpn/num_neg_person_anchors", num_neg_person_anchors / self.num_images)
        storage.put_scalar("rpn/num_pos_object_anchors", num_pos_object_anchors / self.num_images)
        storage.put_scalar("rpn/num_neg_object_anchors", num_neg_object_anchors / self.num_images)

        assert gt_person_logits.shape[1] == num_anchors_per_image
        assert gt_object_logits.shape[1] == num_anchors_per_image

        # Split to tuple of L tensors, each with shape (N, num_anchors_per_map)
        gt_person_logits = torch.split(gt_person_logits, num_anchors_per_map, dim=1)
        gt_object_logits = torch.split(gt_object_logits, num_anchors_per_map, dim=1)

        # Concat from all feature maps
        gt_person_logits = cat([x.flatten() for x in gt_person_logits], dim=0)
        gt_object_logits = cat([x.flatten() for x in gt_object_logits], dim=0)

        # Stack to: (N, num_anchors_per_image, B)
        gt_person_deltas = torch.stack(gts["gt_person_deltas"], dim=0)
        gt_object_deltas = torch.stack(gts["gt_object_deltas"], dim=0)
        assert gt_person_deltas.shape[1] == num_anchors_per_image
        assert gt_object_deltas.shape[1] == num_anchors_per_image

        B = gt_person_deltas.shape[2]  # box dimension (4 or 5)
        assert B == gt_object_deltas.shape[2]

        # Split to tuple of L tensors, each with shape (N, num_anchors_per_image)
        gt_person_deltas = torch.split(gt_person_deltas, num_anchors_per_map, dim=1)
        gt_object_deltas = torch.split(gt_object_deltas, num_anchors_per_map, dim=1)

        # Concat from all feature maps
        gt_person_deltas = cat([x.reshape(-1, B) for x in gt_person_deltas], dim=0)
        gt_object_deltas = cat([x.reshape(-1, B) for x in gt_object_deltas], dim=0)

        # Collect all interactness logits and delta predictions over feature maps
        # and images to arrive at the same shape as the labels and targets
        # The final ordering is L, N, H, W, A from slowest to fastest axis.
        pred_person_logits = cat(
            [
                # Reshape: (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N*Hi*Wi*A, )
                x.permute(0, 2, 3, 1).flatten()
                for x in self.pred_person_logits
            ],
            dim=0,
        )
        pred_person_deltas = cat(
            [
                # Reshape: (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B)
                #          -> (N*Hi*Wi*A, B)
                x.view(x.shape[0], -1, B, x.shape[-2], x.shape[-1])
                .permute(0, 3, 4, 1, 2)
                .reshape(-1, B)
                for x in self.pred_person_deltas
            ],
            dim=0,
        )
        pred_object_logits = cat(
            [
                # Reshape: (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N*Hi*Wi*A, )
                x.permute(0, 2, 3, 1).flatten()
                for x in self.pred_object_logits
            ],
            dim=0,
        )
        pred_object_deltas = cat(
            [
                # Reshape: (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B)
                #          -> (N*Hi*Wi*A, B)
                x.view(x.shape[0], -1, B, x.shape[-2], x.shape[-1])
                .permute(0, 3, 4, 1, 2)
                .reshape(-1, B)
                for x in self.pred_object_deltas
            ],
            dim=0,
        )

        loss_returns = horpn_losses(
            gt_person_logits,
            gt_object_logits,
            gt_person_deltas,
            gt_object_deltas,
            pred_person_logits,
            pred_object_logits,
            pred_person_deltas,
            pred_object_deltas,
            self.smooth_l1_beta,
        )
        normalizer = 1.0 / (self.batch_size_per_image * self.num_images)
        # cls: classification loss
        loss_p_cls = loss_returns["person_interactness_loss"] * normalizer
        loss_o_cls = loss_returns["object_interactness_loss"] * normalizer
        # loc: localization loss
        loss_p_loc = loss_returns["person_localization_loss"] * normalizer
        loss_o_loc = loss_returns["object_localization_loss"] * normalizer

        return {
            "loss_rpn_pcls": loss_p_cls,
            "loss_rpn_ocls": loss_o_cls,
            "loss_rpn_ploc": loss_p_loc,
            "loss_rpn_oloc": loss_o_loc,
        }

    def predict_proposals(self):
        """
        Transform anchors into proposals by applying the predicted anchor deltas.

        Returns:
            proposals (list[Tensor]): A list of L tensors. Tensor i has shape
                (N, Hi*Wi*A, B), where B is box dimension (4 or 5).
        """
        person_proposals = []
        # Transpose anchors from images-by-feature-maps (N, L) to feature-maps-by-images (L, N)
        anchors = list(zip(*self.anchors))
        # Person proposals: for each feature map
        for anchors_i, pred_person_deltas_i in zip(anchors, self.pred_person_deltas):
            B = anchors_i[0].tensor.size(1)
            N, _, Hi, Wi = pred_person_deltas_i.shape
            # Reshape: (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N*Hi*Wi*A, B)
            pred_person_deltas_i = (
                pred_person_deltas_i.view(N, -1, B, Hi, Wi).permute(0, 3, 4, 1, 2).reshape(-1, B)
            )
            # Concatenate all anchors to shape (N*Hi*Wi*A, B)
            # type(anchors_i[0]) is Boxes (B = 4) or RotatedBoxes (B = 5)
            anchors_i = type(anchors_i[0]).cat(anchors_i)
            proposals_i = self.box2box_transform.apply_deltas(
                pred_person_deltas_i, anchors_i.tensor
            )
            # Append feature map proposals with shape (N, Hi*Wi*A, B)
            person_proposals.append(proposals_i.view(N, -1, B))

        object_proposals = []
        for anchors_i, pred_object_deltas_i in zip(anchors, self.pred_object_deltas):
            B = anchors_i[0].tensor.size(1)
            N, _, Hi, Wi = pred_object_deltas_i.shape
            # Reshape: (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N*Hi*Wi*A, B)
            pred_object_deltas_i = (
                pred_object_deltas_i.view(N, -1, B, Hi, Wi).permute(0, 3, 4, 1, 2).reshape(-1, B)
            )
            # Concatenate all anchors to shape (N*Hi*Wi*A, B)
            # type(anchors_i[0]) is Boxes (B = 4) or RotatedBoxes (B = 5)
            anchors_i = type(anchors_i[0]).cat(anchors_i)
            proposals_i = self.box2box_transform.apply_deltas(
                pred_object_deltas_i, anchors_i.tensor
            )
            # Append feature map proposals with shape (N, Hi*Wi*A, B)
            object_proposals.append(proposals_i.view(N, -1, B))

        return {
            "person": person_proposals,
            "object": object_proposals,
        }

    def predict_proposal_logits(self):
        """
        Return interactness logits in the same format as the proposals returned by
        :meth:`predict_proposals`.

        Returns:
            pred_interactness_logits (list[Tensor]): A list of L tensors. Tensor i has shape
                (N, Hi*Wi*A).
        """
        pred_person_proposal_logits = [
            # Reshape: (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            person_score.permute(0, 2, 3, 1).reshape(self.num_images, -1)
            for person_score in self.pred_person_logits
        ]
        pred_object_proposal_logits = [
            # Reshape: (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            object_score.permute(0, 2, 3, 1).reshape(self.num_images, -1)
            for object_score in self.pred_object_logits
        ]
        return {
            "person": pred_person_proposal_logits,
            "object": pred_object_proposal_logits,
        }