#!/usr/bin/env python

import argparse
import json
import numpy as np
import os
from collections import defaultdict
import cv2
import tqdm
import torch
from fvcore.common.file_io import PathManager


from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import Boxes, BoxMode, Instances
from detectron2.utils.logger import setup_logger
from lib.utils.visualizer import InteractionVisualizer
from lib.data.datasets import builtin 


def create_instances(predictions, image_size):
    """
    Args:
        predictions (List[List]): a list of interaction prediction, which is saved in format
            [[interaction_id, person_box, object_box, score], ..., []]
    """
    ret = Instances(image_size)

    score = np.asarray([x[-1] for x in predictions])
    chosen = (score > args.conf_threshold).nonzero()[0]
    if len(chosen) == 0:
        return None
    score = score[chosen]

    # Note that HICO-DET official evaluation uses BoxMode=XYXY_ABS
    person_bbox = np.asarray([predictions[i][1:5] for i in chosen])
    object_bbox = np.asarray([predictions[i][5:9] for i in chosen])
    
    labels = np.asarray([predictions[i][0] for i in chosen])

    ret.scores = score
    ret.person_boxes = Boxes(person_bbox)
    ret.object_boxes = Boxes(object_bbox)
    ret.pred_classes = labels
    
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script that visualizes the interaction predictions from HICO-DET dataset."
    )
    parser.add_argument("--input", required=True, help=".pth file saved for the evaluation")
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--dataset", help="name of the dataset", default="hico-det_test")
    parser.add_argument("--conf-threshold", default=0.5, type=float, help="confidence threshold")
    args = parser.parse_args()

    logger = setup_logger()

    #with PathManager.open(args.input, "r") as f:
    predictions = torch.load(args.input)

    pred_by_image = defaultdict(list)
    for p in predictions:
        pred_by_image[p["image_id"]].append(p["hoi_instances"])

    dicts = list(DatasetCatalog.get(args.dataset))
    metadata = MetadataCatalog.get(args.dataset)
    if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):

        def dataset_id_map(ds_id):
            return metadata.thing_dataset_id_to_contiguous_id[ds_id]

    elif "lvis" in args.dataset:
        # LVIS results are in the same format as COCO results, but have a different
        # mapping from dataset category id to contiguous category id in [0, #categories - 1]
        def dataset_id_map(ds_id):
            return ds_id - 1

    else:
        raise ValueError("Unsupported dataset: {}".format(args.dataset))
    
    os.makedirs(args.output, exist_ok=True)

    for dic in tqdm.tqdm(dicts):
        img = cv2.imread(dic["file_name"], cv2.IMREAD_COLOR)[:, :, ::-1]
        basename = os.path.basename(dic["file_name"])

        predictions = create_instances(pred_by_image[dic["image_id"]][0], img.shape[:2])
        if predictions is None:
            continue
        vis = InteractionVisualizer(img, metadata)
        vis_pred = vis.draw_interaction_predictions(predictions).get_image()
        cv2.imwrite(os.path.join(args.output, basename), vis_pred[:, :, ::-1])

        #vis = InteractionVisualizer(img, metadata)
        #vis_gt = vis.draw_dataset_dict(dic).get_image()

        #concat = np.concatenate((vis_pred, vis_gt), axis=1)
        #cv2.imwrite(os.path.join(args.output, basename), concat[:, :, ::-1])
