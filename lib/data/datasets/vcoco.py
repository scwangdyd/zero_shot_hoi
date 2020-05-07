
import contextlib
import datetime
import io
import json
import logging
import numpy as np
import os
import pycocotools.mask as mask_util
from fvcore.common.file_io import PathManager, file_lock
from fvcore.common.timer import Timer
from PIL import Image

from detectron2.structures import Boxes, BoxMode, PolygonMasks
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog

"""
This file contains functions to parse HOI annotations into dicts in "Detectron2 format".
"""


logger = logging.getLogger(__name__)

__all__ = ["load_vcoco_json"]


def load_vcoco_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None):
    """
    Load a json file with HOI's instances annotation format.

    Args:
        json_file (str): full path to the json file in HOI instances annotation format.
        image_root (str or path-like): the directory where the images in this json file exists.
        dataset_name (str): the name of the dataset (e.g., `vcoco_train`).
            If provided, this function will also put "thing_classes" into
            the metadata associated with this dataset.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "iscrowd", "bbox", "keypoints",
            "category_id", "segmentation"). The values for these keys will be returned as-is.
            For example, the densepose annotations are loaded in this way.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard dataset dicts format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    from pycocotools.coco import COCO

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    id_map = None
    action_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        cat_ids = sorted(coco_api.getCatIds())
        cats = coco_api.loadCats(cat_ids)
        # The categories in a custom json file may not be sorted.
        thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
        meta.thing_classes = thing_classes

        # In COCO, certain category ids are artificially removed,
        # and by convention they are always ignored.
        # We deal with COCO's id issue and translate
        # the category ids to contiguous ids in [0, 80).

        # It works by looking at the "categories" field in the json, therefore
        # if users' own json also have incontiguous ids, we'll
        # apply this mapping as well but print a warning.
        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            if "coco" not in dataset_name:
                logger.warning(
                    """
Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
"""
                )
        id_map = {v: i for i, v in enumerate(cat_ids)}
        meta.thing_dataset_id_to_contiguous_id = id_map

        person_cls_id = meta.person_cls_id
        action_classes = meta.action_classes

    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())
    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    imgs = coco_api.loadImgs(img_ids)
    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [{'segmentation': [[192.81,
    #     247.09,
    #     ...
    #     219.03,
    #     249.06]],
    #   'area': 1035.749,
    #   'iscrowd': 0,
    #   'image_id': 1268,
    #   'bbox': [192.81, 224.8, 74.73, 33.43],
    #   'category_id': 16,
    #   'id': 42986,
    #   'hoi_isactive': 1,
    #   'hoi_triplets': [{person_id: 42984, object_id: 42986, action_id: 4}, ...],
    #  },
    #  ...]
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]

    imgs_anns = list(zip(imgs, anns))

    logger.info("Loaded {} images in HOI format from {}".format(len(imgs_anns), json_file))

    dataset_dicts = []

    ann_keys = ["iscrowd", "bbox", "category_id"]
    
    ann_keys += (extra_annotation_keys or [])

    num_instances_without_hoi_annotations = 0

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        num_instances = len(anno_dict_list)
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.

            # The original COCO valminusminival2014 & minival2014 annotation files
            # actually contains bugs that, together with certain ways of using COCO API,
            # can trigger this assertion.
            assert anno["image_id"] == image_id

            obj = {key: anno[key] for key in ann_keys if key in anno}

            # "hoi_triplets" in the annotation is a list[dict], where each dict is an
            # annotation record for an interaction. Example of anno["hoi_triplet"][0]:
            # [{
            #       person_id: 42984,
            #       object_id: 42986,
            #       action_id: 4
            #   },
            # ... ]
            # Here "person_id" ("object_id") is the *anno id* of the person (object) instance.
            # For each instance, we record its interactions with other instances in the given
            # image in an binary matrix named `actions` with shape (N, K), where N is the number
            # of instances and K is the number of actions. If this instance is interacting with
            # j-th instance with k-th action, then (i, j) entry of `actions` will be 1.
            actions = np.zeros((num_instances, len(action_classes)))
            hoi_triplets = anno["hoi_triplets"]
            if len(hoi_triplets) > 0:
                # Mapping *anno id* of instances to contiguous indices in this image
                map_to_contiguous_id_within_image(hoi_triplets, anno_dict_list)
                for triplet in hoi_triplets:
                    action_id = triplet["action_id"]
                    is_person = (anno["category_id"] == person_cls_id)
                    target_id = triplet["object_id"] if is_person else triplet["person_id"]
                    actions[target_id, action_id] = 1
            else:
                num_instances_without_hoi_annotations += 1

            obj["actions"] = actions
            obj["isactive"] = 1 if len(hoi_triplets) > 0 else 0

            obj["bbox_mode"] = BoxMode.XYWH_ABS

            if id_map:
                obj["category_id"] = id_map[obj["category_id"]]
            
            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)

    if num_instances_without_hoi_annotations > 0:
        logger.warning(
            "There are {} instances without hoi annotation.".format(
                num_instances_without_hoi_annotations
            )
        )
    return dataset_dicts


def map_to_contiguous_id_within_image(hoi_triplets, anno_dict_list):
    """
    Map annotation id in HOI triplets to contiguous index within the given image.
    For example, map {"person_id": 2001, "object_id": 2003, "action_id": 1} to
                     {"person_id": 0,    "object_id": 2,    "action_id": 1}) if
    the annotation ids in this image start from 2001.

    Args:
        hoi_triplets (list[dict]): HOI annotations of an instance.
        anno_dict_list (list[dict]): annotations of all instances in the image.

    Returns:
        list[dict]: HOI annotations with contiguous id within the image.
    """
    anno_id_to_contiguous_id = {ann['id']: ix for ix, ann in enumerate(anno_dict_list)}
    # This fails when annotation file is buggy. The dataset may contain person alone interactions,
    # (e.g., without interacting objects). The object index in this case is denoted as -1.
    anno_id_to_contiguous_id.update({-1: -1})

    for triplet in hoi_triplets:
        triplet['person_id'] = anno_id_to_contiguous_id[triplet['person_id']]
        triplet['object_id'] = anno_id_to_contiguous_id[triplet['object_id']]
