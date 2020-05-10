import colorsys
import logging
import math
import numpy as np
from enum import Enum, unique
import cv2
import matplotlib as mpl
import matplotlib.colors as mplc
import matplotlib.figure as mplfigure
import pycocotools.mask as mask_util
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg

from detectron2.utils.visualizer import Visualizer, ColorMode, VisImage
from detectron2.structures import BitMasks, Boxes, BoxMode, Keypoints, PolygonMasks, RotatedBoxes
from detectron2.utils.colormap import random_color

logger = logging.getLogger(__name__)

__all__ = ["InteractionVisualizer"]

_SMALL_OBJECT_AREA_THRESH = 1000

def _create_text_labels(classes, scores, class_names=None):
    """
    Args:
        classes (list[int] or None):
        scores (list[float] or None):
        class_names (Dict[int: str] or None):

    Returns:
        list[str] or None
    """
    labels = None
    if classes is not None and class_names is not None:
        labels = [class_names[i] for i in classes]
    if classes is not None and isinstance(classes[0], str):
        labels = classes.tolist()
    if scores is not None:
        if labels is None:
            labels = ["{:.0f}%".format(s * 100) for s in scores]
        else:
            labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]
    return labels


class InteractionVisualizer(Visualizer):
    def __init__(self, img_rgb, metadata, scale=1.0, instance_mode=ColorMode.IMAGE):
        """
        Args:
            img_rgb: a numpy array of shape (H, W, C), where H and W correspond to
                the height and width of the image respectively. C is the number of
                color channels. The image is required to be in RGB format since that
                is a requirement of the Matplotlib library. The image is also expected
                to be in the range [0, 255].
            metadata (MetadataCatalog): image metadata.
        """
        self.img = np.asarray(img_rgb).clip(0, 255).astype(np.uint8)
        self.metadata = metadata
        self.output = VisImage(self.img, scale=scale)
        self.cpu_device = torch.device("cpu")

        # too small texts are useless, therefore clamp to 9
        self._default_font_size = max(
            np.sqrt(self.output.height * self.output.width) // 90, 8 // scale
        )
        self._instance_mode = instance_mode

    def draw_interaction_predictions(self, predictions):
        """
        Draw interaction prediction results on an image.

        Args:
            predictions (Instances): the output of an interaction detection model.
            Following fields will be used to draw:
                "person_boxes", "object_boxes", "pred_classes", "scores"

        Returns:
            output (VisImage): image object with visualizations.
        """
        num_instances = len(predictions)
        if num_instances == 0:
            return self.output 
        
        person_boxes = self._convert_boxes(predictions.person_boxes)
        object_boxes = self._convert_boxes(predictions.object_boxes)
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes if predictions.has("pred_classes") else None

        # convert labels
        # contiguous_id_to_classes = self.metadata.get("contiguous_id_to_classes", None)
        labels = _create_text_labels(classes, scores)

        # Take unique person and object boxes and assign colors.
        unique_person_boxes = np.asarray([list(x) for x in set(tuple(x) for x in person_boxes)])
        unique_object_boxes = np.asarray([list(x) for x in set(tuple(x) for x in object_boxes)])
        
        # If labels and meta data are available, use assigned colors. Otherwise, use random colors.
        thing_colors = self.metadata.get("thing_colors", None)
        assigned_person_colors = {tuple(x): 'w' for x in unique_person_boxes}
        assigned_object_colors = {tuple(x): random_color(True, 1) for x in unique_object_boxes}
        # if labels is not None and thing_colors is not None:
        #     for label_ix, box_ix in zip(labels, object_boxes):
        #         class_name = " ".join(label_ix.split(" ")[1:-1])
        #         color = thing_colors[class_name] if class_name in thing_colors else None
        #         if color:
        #             assigned_object_colors[tuple(box_ix)] = np.asarray(color) / 255.

        # Take all interaction associated with each unique person box
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
    
        self.overlay_interactions(
            unique_person_boxes=unique_person_boxes,
            unique_object_boxes=unique_object_boxes,
            interactions=interactions_to_draw,
            interaction_labels=labels_to_draw,
            assigned_person_colors=assigned_person_colors,
            assigned_object_colors=assigned_object_colors,
            alpha=0.5,
        )
        return self.output

    def overlay_interactions(
        self,
        *,
        unique_person_boxes=None,
        unique_object_boxes=None,
        interactions=None,
        interaction_labels=None,
        assigned_person_colors=None,
        assigned_object_colors=None,
        alpha=0.5
    ):
        """
        Args:
            person_boxes (Boxes or ndarray): either a :class:`Boxes`,
                or an Nx4 numpy array of XYXY_ABS format for the N person in a single image,
            object_boxes (Boxes or ndarray): either a :class:`Boxes`,
                or an Nx4 numpy array of XYXY_ABS format for the N object in a single image,
            labels (list[str]): the text to be displayed for each instance.
            assigned_colors (list[matplotlib.colors]): a list of colors, where each color
                corresponds to each box in the image. Refer to 'matplotlib.colors'
                for full list of formats that the colors are accepted in.

        Returns:
            output (VisImage): image object with visualizations.
        """
        if assigned_person_colors is None:
            assigned_person_colors = ["b"] * len(person_boxes)
        if assigned_object_colors is None:
            assigned_object_colors = ["g"] * len(object_boxes)

        # 1. Draw unique person and object boxes. 
        for box in unique_person_boxes:
            self.draw_box(box, edge_color=assigned_person_colors[tuple(box)], alpha=1.0)
        for box in unique_object_boxes:
            self.draw_box(box, edge_color=assigned_object_colors[tuple(box)], alpha=alpha)

        # 2. Draw interactions.
        for person_box, object_list in interactions.items():
            for object_box in object_list:
                
                p_x0, p_y0, p_x1, p_y1 = person_box
                p_xc, p_yc = (p_x0 + p_x1) / 2, (p_y0 + p_y1) / 2
                      
                o_x0, o_y0, o_x1, o_y1 = object_box
                o_xc, o_yc = (o_x0 + o_x1) / 2, (o_y0 + o_y1) / 2
            
                self.draw_circle((p_xc, p_yc), color="w")
                self.draw_circle((o_xc, o_yc), color="w")
                self.draw_line([p_xc, o_xc], [p_yc, o_yc], color="w", linewidth=2)

        # 3. Display in largest to smallest order based on person areas to reduce occlusion.
        areas = np.prod(unique_person_boxes[:, 2:] - unique_person_boxes[:, :2], axis=1)
        sorted_idxs = np.argsort(-areas).tolist()
        # Re-order overlapped instances in descending order.
        unique_person_boxes = unique_person_boxes[sorted_idxs]

        for person_box, texts in interaction_labels.items():
            x0, y0, x1, y1 = person_box
            text_pos = (x0, y0)
            horiz_align = "left"
            direction = "down"
            # for small objects, draw text at the side to avoid occlusion
            instance_area = (y1 - y0) * (x1 - x0)
            if (
                instance_area < _SMALL_OBJECT_AREA_THRESH * self.output.scale
                or y1 - y0 < 40 * self.output.scale
            ):
                if y1 >= self.output.height - 5:
                    text_pos = (x1, y0)
                    direction = "down"
                elif y0 < 5:
                    text_pos = (x0, y1)
                    direction = "up"
                else:
                    text_pos = (x0, y1)
                    direction = "up"
            
            height_ratio = (y1 - y0) / np.sqrt(self.output.height * self.output.width)
            font_size = (
                np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2)
                * 0.5
                * self._default_font_size
            )
            
            label_texts = [x["label"] for x in texts]
            text_colors = [x["color"] for x in texts]
            
            self.draw_text_multilines(
                text_pos,
                texts=label_texts,
                colors=text_colors,
                horizontal_alignment=horiz_align,
                font_size=font_size,
                alpha=alpha,
                direction=direction
            )
            
        return self.output


    def draw_text_multilines(
        self,
        position,
        texts,
        colors,
        *,
        font_size=None,
        horizontal_alignment="left",
        rotation=0,
        alpha=0.5,
        direction="down"
    ):
        """
        Args:
            position (tuple): a tuple of the x and y coordinates to place text on image.
            texts (list): A list of interaction labels
            font_size (int, optional): font of the text. If not provided, a font size
                proportional to the image width is calculated and used.
            colors (list): A list of colors of the text. Refer to `matplotlib.colors`
                for full list of formats that are accepted.
            horizontal_alignment (str): see `matplotlib.text.Text`
            rotation: rotation angle in degrees CCW

        Returns:
            output (VisImage): image object with text drawn.
        """
        if not font_size:
            font_size = self._default_font_size

        # since the text background is dark, we don't want the text to be dark
        bright_colors = []
        for x in colors:
            color = np.maximum(list(mplc.to_rgb(x)), 0.2)
            color[np.argmax(color)] = max(0.8, np.max(color))
            bright_colors.append(color)
        

        canvas = self.output.ax.figure.canvas
        t = self.output.ax.transData
        
        x, y = position
        for text, color in zip(texts, colors):
            f = self.output.ax.text(
                x,
                y,
                text,
                size=font_size * self.output.scale,
                family="sans-serif",
                #bbox={"facecolor": "black", "alpha": 0.3, "pad": 0.7, "edgecolor": "none"},
                bbox={"facecolor": color, "alpha": alpha, "pad": 0.7, "edgecolor": "none"},
                verticalalignment="top",
                horizontalalignment=horizontal_alignment,
                color='w',
                zorder=10,
                rotation=rotation,
                transform=t
            )

            # Need to draw to update the text position.
            f.draw(canvas.get_renderer())
            ex = f.get_window_extent()
            t = mpl.transforms.offset_copy(
                f.get_transform(),
                y=ex.height if direction == "up" else -ex.height,
                units="dots"
            )

        return self.output
    
    
    def draw_proposals(self, proposals):
        """
        Draw interaction prediction results on an image.

        Args:
            predictions (Instances): the output of an interaction detection model.
            Following fields will be used to draw:
                "person_boxes", "object_boxes", "pred_classes", "scores"

        Returns:
            output (VisImage): image object with visualizations.
        """
        num_instances = len(proposals)
        if num_instances == 0:
            return self.output 
        
        proposal_boxes = self._convert_boxes(proposals.proposal_boxes)
        scores = np.asarray(proposals.interactness_logits)
        is_person = np.asarray(proposals.is_person)

        # Boxes to draw
        person_boxes = proposal_boxes[is_person == 1]
        object_boxes = proposal_boxes[is_person == 0]

        person_scores = scores[is_person == 1]
        object_scores = scores[is_person == 0]
        
        topn_person = 5
        topn_object = 10
        boxes_to_draw = np.append(
            person_boxes[0:topn_person, :], object_boxes[0:topn_object, :], axis=0
        )
        scores_to_draw = np.append(
            person_scores[0:topn_person], object_scores[0:topn_object], axis=0
        )
        labels_to_draw = ["{:.1f}".format(x * 100) for x in scores_to_draw]
        assigned_colors = ["b"] * topn_person + ["g"] * topn_object
        
        self.overlay_instances(
            boxes=boxes_to_draw,
            labels=labels_to_draw,
            assigned_colors=assigned_colors,
            alpha=1
        )

        return self.output