# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import torch
import pycocotools.mask as mask_util

from detectron2.utils.visualizer import (
    ColorMode,
    Visualizer,
    _PanopticPrediction,
)
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.colormap import random_color
from detectron2.structures import Boxes, RotatedBoxes

from lib.utils.visualizer import InteractionVisualizer, _create_text_labels

class _DetectedInstance:
    """
    Used to store data about detected objects in video frame,
    in order to transfer color to objects in the future frames.
    Attributes:
        label (int):
        bbox (tuple[float]):
        mask_rle (dict):
        color (tuple[float]): RGB colors in range (0, 1)
        ttl (int): time-to-live for the instance. For example, if ttl=2,
            the instance color can be transferred to objects in the next two frames.
    """

    __slots__ = ["label", "bbox", "color", "ttl"]

    def __init__(self, label, bbox, color, ttl):
        self.label = label
        self.bbox = bbox
        self.color = color
        self.ttl = ttl


class VideoVisualizer:
    def __init__(self, metadata, instance_mode=ColorMode.IMAGE):
        """
        Args:
            metadata (MetadataCatalog): image metadata.
        """
        self.metadata = metadata
        self._old_instances = []
        assert instance_mode in [
            ColorMode.IMAGE,
            ColorMode.IMAGE_BW,
        ], "Other mode not supported yet."
        self._instance_mode = instance_mode

    def draw_interaction_predictions(self, frame, predictions):
        """
        Draw interaction prediction results on an image.
        Args:
            frame (ndarray): an RGB image of shape (H, W, C), in the range [0, 255].
            predictions (Instances): the output of an interaction detection model.
                Following fields will be used to draw: "person_boxes", "object_boxes",
                "object_classes", "action_classes", "scores".
        Returns:
            output (VisImage): image object with visualizations.
        """
        frame_visualizer = InteractionVisualizer(frame, self.metadata)
        thing_colors = self.metadata.get("thing_colors", "None")
        if thing_colors:
            thing_colors = [color for name, color in thing_colors.items()]

        num_instances = len(predictions)
        if num_instances == 0:
            return frame_visualizer.output
        
        person_boxes = self._convert_boxes(predictions.person_boxes)
        object_boxes = self._convert_boxes(predictions.object_boxes)
        object_classes = predictions.object_classes
        classes = predictions.pred_classes
        scores = predictions.scores
        
        # Take the unique person and object boxes.
        unique_person_boxes = np.asarray([list(x) for x in set(tuple(x) for x in person_boxes)])
        unique_object_boxes = np.asarray([list(x) for x in set(tuple(x) for x in object_boxes)])
        unique_object_classes = {tuple(x): -1 for x in unique_object_boxes}
        for box, c in zip(object_boxes, object_classes):
            unique_object_classes[tuple(box)] = c
        unique_object_colors = {tuple(x): None for x in unique_object_boxes}
        if thing_colors:
            for box, c in unique_object_classes.items():
                unique_object_colors[box] = thing_colors[c]
        unique_object_colors = [color for _, color in unique_object_colors.items()]

        # Assign colors to person boxes and object boxes.
        object_detected = [
            _DetectedInstance(unique_object_classes[tuple(box)], box, color=color, ttl=8)
            for box, color in zip(unique_object_boxes, unique_object_colors)
        ]
        object_colors = self._assign_colors(object_detected)
        
        assigned_person_colors = {tuple(x): 'w' for x in unique_person_boxes}
        assigned_object_colors = {tuple(x.bbox): x.color for x in object_detected}
        
        # Take all interaction associated with each unique person box
        # classes_to_contiguous_id = self.metadata.get("interaction_classes_to_contiguous_id", None)
        # contiguous_id_to_classes = {v: k for k, v in classes_to_contiguous_id.items()} \
        #     if classes_to_contiguous_id else None
        labels = _create_text_labels(classes, scores)
    
        interactions_to_draw = {tuple(x): [] for x in unique_person_boxes}
        labels_to_draw = {tuple(x): [] for x in unique_person_boxes}
        for i in range(num_instances):
            x = tuple(person_boxes[i])
            interactions_to_draw[x].append(object_boxes[i])
            if labels is not None:
                labels_to_draw[x].append(
                    {
                        "label": labels[i],
                        "color": assigned_object_colors[tuple(object_boxes[i])]
                    }
                )
        
        if self._instance_mode == ColorMode.IMAGE_BW:
            # any() returns uint8 tensor
            frame_visualizer.output.img = frame_visualizer._create_grayscale_image(
                (masks.any(dim=0) > 0).numpy() if masks is not None else None
            )
            alpha = 0.3
        else:
            alpha = 0.5

        frame_visualizer.overlay_interactions(
            unique_person_boxes=unique_person_boxes,
            unique_object_boxes=unique_object_boxes,
            interactions=interactions_to_draw,
            interaction_labels=labels_to_draw,
            assigned_person_colors=assigned_person_colors,
            assigned_object_colors=assigned_object_colors,
            alpha=0.5,
        )

        return frame_visualizer.output
        

    def draw_instance_predictions(self, frame, predictions):
        """
        Draw instance-level prediction results on an image.
        Args:
            frame (ndarray): an RGB image of shape (H, W, C), in the range [0, 255].
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").
        Returns:
            output (VisImage): image object with visualizations.
        """
        frame_visualizer = Visualizer(frame, self.metadata)
        num_instances = len(predictions)
        if num_instances == 0:
            return frame_visualizer.output

        boxes = predictions.pred_boxes.tensor.numpy() if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes.numpy() if predictions.has("pred_classes") else None

        detected = [
            _DetectedInstance(classes[i], boxes[i], color=None, ttl=8)
            for i in range(num_instances)
        ]
        colors = self._assign_colors(detected)

        labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))

        if self._instance_mode == ColorMode.IMAGE_BW:
            # any() returns uint8 tensor
            frame_visualizer.output.img = frame_visualizer._create_grayscale_image(
                (masks.any(dim=0) > 0).numpy() if masks is not None else None
            )
            alpha = 0.3
        else:
            alpha = 0.5

        frame_visualizer.overlay_instances(
            boxes=boxes,
            labels=labels,
            assigned_colors=colors,
            alpha=alpha,
        )

        return frame_visualizer.output

    def _assign_colors(self, instances):
        """
        Naive tracking heuristics to assign same color to the same instance,
        will update the internal state of tracked instances.
        Returns:
            list[tuple[float]]: list of colors.
        """

        # Compute iou with either boxes or masks:
        is_crowd = np.zeros((len(instances),), dtype=np.bool)

        boxes_old = [x.bbox for x in self._old_instances]
        boxes_new = [x.bbox for x in instances]
        ious = mask_util.iou(boxes_old, boxes_new, is_crowd)
        threshold = 0.6

        if len(ious) == 0:
            ious = np.zeros((len(self._old_instances), len(instances)), dtype="float32")

        # Only allow matching instances of the same label:
        for old_idx, old in enumerate(self._old_instances):
            for new_idx, new in enumerate(instances):
                if old.label != new.label:
                    ious[old_idx, new_idx] = 0

        matched_new_per_old = np.asarray(ious).argmax(axis=1)
        max_iou_per_old = np.asarray(ious).max(axis=1)

        # Try to find match for each old instance:
        extra_instances = []
        for idx, inst in enumerate(self._old_instances):
            if max_iou_per_old[idx] > threshold:
                newidx = matched_new_per_old[idx]
                if instances[newidx].color is None:
                    instances[newidx].color = inst.color
                    continue
            # If an old instance does not match any new instances,
            # keep it for the next frame in case it is just missed by the detector
            inst.ttl -= 1
            if inst.ttl > 0:
                extra_instances.append(inst)

        # Assign random color to newly-detected instances:
        for inst in instances:
            if inst.color is None:
                inst.color = random_color(rgb=True, maximum=1)
        self._old_instances = instances[:] + extra_instances
        return [d.color for d in instances]
    
    def _convert_boxes(self, boxes):
        """
        Convert different format of boxes to an NxB array, where B = 4 or 5 is the box dimension.
        """
        if isinstance(boxes, Boxes) or isinstance(boxes, RotatedBoxes):
            return boxes.tensor.numpy()
        else:
            return np.asarray(boxes)
        
    def draw_proposals(self, frame, proposals, thresh):
        """
        Draw interaction prediction results on an image.

        Args:
            predictions (Instances): the output of an interaction detection model.
            Following fields will be used to draw:
                "person_boxes", "object_boxes", "pred_classes", "scores"

        Returns:
            output (VisImage): image object with visualizations.
        """
        _MAX_OBJECT_AREA = 60000

        frame_visualizer = InteractionVisualizer(frame, self.metadata)
        num_instances = len(proposals)
        if num_instances == 0:
            return frame_visualizer.output 
        
        proposal_boxes = self._convert_boxes(proposals.proposal_boxes)
        scores = np.asarray(proposals.interactness_logits)
        is_person = np.asarray(proposals.is_person)

        topn_person = 5
        topn_object = 10
        # Boxes to draw
        person_boxes = proposal_boxes[is_person == 1][0:topn_person, :]
        person_scores = scores[is_person == 1][0:topn_person]
        
        object_boxes = proposal_boxes[is_person == 0]
        object_scores = scores[is_person == 0]     
        
        areas = np.prod(object_boxes[:, 2:] - object_boxes[:, :2], axis=1)
        keep = areas < _MAX_OBJECT_AREA
        object_boxes = object_boxes[keep]
        object_scores = object_scores[keep]
        
        object_boxes = object_boxes[0:topn_object, :]   
        object_scores = object_scores[0:topn_object]

        boxes_to_draw = object_boxes
        scores_to_draw = object_scores
        labels_to_draw = np.asarray(["{:.1f}".format(x * 100) for x in scores_to_draw])
        assigned_colors = np.asarray(["g"] * topn_object)
        
        keep = scores_to_draw > thresh
        
        frame_visualizer.overlay_instances(
            boxes=boxes_to_draw[keep],
            labels=labels_to_draw[keep],
            assigned_colors=assigned_colors[keep],
            alpha=1
        )

        return frame_visualizer.output