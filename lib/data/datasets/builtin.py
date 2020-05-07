# -*- coding: utf-8 -*-
"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations

We hard-code some paths to the dataset that's assumed to exist in "./datasets/".

Refer to the tutorial "detectron2/docs/DATASETS.md" to add new dataset.
"""

import os
from detectron2.data.datasets import register_coco_instances
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog

from .builtin_meta import _get_builtin_metadata
from .hico import load_hico_json
from .vcoco import load_vcoco_json


_PREDEFINED_SPLITS_COCO = {}
_PREDEFINED_SPLITS_COCO["coco_minus_vcoco"] = {
    "coco2014train_minus_vcocoval": (
        "coco/images/train2014", # path to images
        "coco/annotations/instances_train2014minusvcocoval.json" # path to the annotation file
        ),
    "coco2014val_minus_vcocotest": (
        "coco/images/val2014", 
        "coco/annotations/instances_val2014minusvcocotest.json"
    ),
}
_PREDEFINED_SPLITS_COCO["coco_minus_vcoco_known"] = {
    "coco2014train_minus_vcocoval_known": (
        "coco/images/train2014",
        "coco/annotations/instances_train2014minusvcocoval_known.json"
    ),
    "coco2014val_minus_vcocotest_known": (
        "coco/images/val2014",
        "coco/annotations/instances_val2014minusvcocotest_known.json"
    ),
}


_PREDEFINED_SPLITS_VCOCO = {}
_PREDEFINED_SPLITS_VCOCO["vcoco"] = {
    "vcoco_train": (
        "coco/images/train2014",
        "coco/annotations/instances_vcocotrain_hoi_github.json"
    ),
    "vcoco_val": (
        "coco/images/train2014",
        "coco/annotations/instances_vcocoval_hoi_github.json"
    ),
    "vcoco_test": (
        "coco/images/val2014",
        "coco/annotations/instances_vcocotest_hoi_github.json"
    ),
    "vcoco_val_only_interaction": (
        "coco/images/train2014",
        "coco/annotations/instances_vcocoval_hoi_only_active_github.json"
    ),
    "vcoco_test_only_interaction": (
        "coco/images/val2014",
        "coco/annotations/instances_vcocotest_hoi_only_active_github.json"
    ),
}
_PREDEFINED_SPLITS_VCOCO["vcoco_known"] = {
    "vcoco_train_known": (
        "coco/images/train2014",
        "coco/annotations/instances_vcocotrain_hoi_known_github.json"
    ),
    "vcoco_val_known": (
        "coco/images/train2014",
        "coco/annotations/instances_vcocoval_hoi_known_github.json"
    ),
}


_PREDEFINED_SPLITS_HICO = {}
_PREDEFINED_SPLITS_HICO["hico-det"] = {
    "hico-det_train": (
        "hico_20160224_det/images/train2015",
        "hico_20160224_det/annotations/instances_hico_train_hoi_github_may1.json",
    ),
    "hico-det_test": (
        "hico_20160224_det/images/test2015",
        "hico_20160224_det/annotations/instances_hico_test_hoi_github_may1.json",
    ),
    "hico-det_train_seen": (
        "hico_20160224_det/images/train2015",
        "hico_20160224_det/annotations/instances_hico_train_hoi_seen_github.json"
    ),
}


def register_hico_instances(name, metadata, json_file, image_root, evaluator_type):
    """
    Register a hico-det dataset in COCO's json annotation format for human-object
    interaction detection (i.e., `instances_hico_*.json` in the dataset).

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        name (str): the name that identifies a dataset, e.g. "hico-det".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_hico_json(json_file, image_root, name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file,
        image_root=image_root,
        evaluator_type=evaluator_type,
        **metadata
    )


def register_vcoco_instances(name, metadata, json_file, image_root, evaluator_type):
    """
    Register a vcoco dataset in COCO's json annotation format for human-object
    interaction detection (i.e., `instances_hico_*.json` in the dataset).

    Args:
        see `register_hico_instances`
    """
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_vcoco_json(json_file, image_root, name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file,
        image_root=image_root,
        evaluator_type=evaluator_type,
        **metadata
    )


def register_all_hico(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_HICO.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            register_hico_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
                evaluator_type=dataset_name
            )


def register_all_coco(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_COCO.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )


def register_all_vcoco(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_VCOCO.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            register_vcoco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
                evaluator_type=dataset_name
            )


# Register them all under _root (path to datasets)
_root = os.getenv("DETECTRON2_DATASETS", "/raid1/suchen/dataset/")
register_all_hico(_root)
register_all_vcoco(_root)
#register_all_coco(_root)
