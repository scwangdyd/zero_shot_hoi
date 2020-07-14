# -*- coding: utf-8 -*-
from .hico_meta import HICO_OBJECTS, HICO_ACTIONS, HICO_INTERACTIONS
from .vcoco_meta import VCOCO_OBJECTS, VCOCO_ACTIONS

def _get_coco_instances_meta():
    """
    Returns metadata for COCO dataset.
    """
    thing_ids = [k["id"] for k in HICO_OBJECTS if k["isthing"] == 1]
    thing_colors = [k["color"] for k in HICO_OBJECTS if k["isthing"] == 1]
    assert len(thing_ids) == 80, len(thing_ids)
    # Mapping from the incontiguous COCO category id to an id in [0, 79]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in HICO_OBJECTS if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


def _get_vcoco_instances_meta():
    """
    Returns metadata for VCOCO dataset.
    """
    thing_ids = [k["id"] for k in VCOCO_OBJECTS if k["isthing"] == 1]
    thing_colors = [k["color"] for k in VCOCO_OBJECTS if k["isthing"] == 1]
    assert len(thing_ids) == 80, len(thing_ids)
    # Mapping from the incontiguous COCO category id to an id in [0, 79]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in VCOCO_OBJECTS if k["isthing"] == 1]
    # Splitting object categories using our known/novel splits. Here all objects are known.
    known_classes = [k["name"] for k in VCOCO_OBJECTS if k["isthing"] == 1]
    novel_classes = []
    # Category id of `person`
    person_cls_id = [k["id"] for k in VCOCO_OBJECTS if k["name"] == 'person'][0]
    # VCOCO actions
    action_classes = [k["name"] for k in VCOCO_ACTIONS]

    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes":  thing_classes,
        "thing_colors":   thing_colors,
        "known_classes":  known_classes,
        "novel_classes":  novel_classes,
        "person_cls_id":  person_cls_id,
        "action_classes": action_classes,
    }
    return ret


def _get_hico_instances_meta():
    """
    Returns metadata for HICO-DET dataset.
    """
    thing_ids = [k["id"] for k in HICO_OBJECTS if k["isthing"] == 1]
    thing_colors = {k["name"]: k["color"] for k in HICO_OBJECTS if k["isthing"] == 1}
    assert len(thing_ids) == 80, len(thing_ids)
    # Mapping from the incontiguous category id to an id in [0, 79]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in HICO_OBJECTS if k["isthing"] == 1]
    # Splitting object categories using our known/novel splits. Here all objects are known.
    known_classes = [k["name"] for k in HICO_OBJECTS if k["isthing"] == 1]
    novel_classes = []
    # HICO-DET actions
    action_classes = [k["name"] for k in HICO_ACTIONS]
    action_priors  = [k["prior"] for k in HICO_ACTIONS]
    # Category id of `person`
    person_cls_id = [k["id"] for k in HICO_OBJECTS if k["name"] == 'person'][0]
    # Mapping interactions (action name + object name) to contiguous id 
    interaction_classes_to_contiguous_id = {
            x["action"] + " " + x["object"]: x["interaction_id"] for x in HICO_INTERACTIONS
        }

    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes":  thing_classes,
        "thing_colors":   thing_colors,
        "known_classes":  known_classes,
        "novel_classes":  novel_classes,
        "action_classes": action_classes,
        "action_priors":  action_priors,
        "person_cls_id":  person_cls_id,
        "interaction_classes_to_contiguous_id": interaction_classes_to_contiguous_id
    }
    return ret


def _get_vcoco_known_instances_meta():
    """
    Returns metadata for HOI dataset.
    """
    thing_ids = [k["id"] for k in HICO_OBJECTS if k["isthing"] == 1]
    thing_colors = [k["color"] for k in HICO_OBJECTS if k["isthing"] == 1]
    assert len(thing_ids) == 80, len(thing_ids)

    known_ids = [k["id"] for k in HICO_OBJECTS if k["isknown"] == 1]
    assert len(known_ids) == 43, len(known_ids)
    # Mapping from the incontiguous known category id to an id in [0, 42]
    known_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(known_ids)}
    known_classes = [k["name"] for k in HICO_OBJECTS if k["isknown"] == 1]
    novel_classes = []
    # Category id of `person`
    person_cls_id = [k["id"] for k in VCOCO_OBJECTS if k["name"] == 'person'][0]
    # VCOCO actions
    action_classes = [k["name"] for k in VCOCO_ACTIONS]

    ret = {
        "thing_dataset_id_to_contiguous_id": known_dataset_id_to_contiguous_id,
        "thing_classes":  known_classes,
        "thing_colors":   thing_colors,
        "known_classes":  known_classes,
        "novel_classes":  novel_classes,
        "person_cls_id":  person_cls_id,
        "action_classes": action_classes,
    }
    return ret


def _get_builtin_metadata(dataset_name):
    if dataset_name == 'vcoco':
        return _get_vcoco_instances_meta()
    elif dataset_name == 'hico-det':
        return _get_hico_instances_meta()
    elif dataset_name == 'vcoco_known':
        return _get_vcoco_known_instances_meta()
    # elif dataset_name == 'coco_minus_vcoco':
    #     return _get_coco_instances_meta()
    # elif dataset_name == "coco_minus_vcoco_known":
    #     return _get_known_instances_meta()
    raise KeyError("No built-in metadata for dataset {}".format(dataset_name))