# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import contextlib
import copy
import io
import itertools
import json
import numpy as np
import os
import logging
import pickle
import scipy.io as sio
from collections import OrderedDict
import pycocotools.mask as mask_util
import torch
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.logger import create_small_table, setup_logger
from detectron2.evaluation.evaluator import DatasetEvaluator

logger = setup_logger(name=__name__)

class HICOEvaluator(DatasetEvaluator):
    """
    Evaluate object proposal, instance detection, using COCO's metrics and APIs.
    Evaluate human-object interaction detection using HICO-DET's metrics and APIs.
    """

    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:

                    "json_file": the path to the COCO format annotation;
                    "matlab_file": the original matlab annotation files,

                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
            cfg (CfgNode): config instance
            distributed (True): if True, will collect results from all ranks and run evaluation
                in the main process. Otherwise, will evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:

                1. "instance_predictions.pth" a file in torch serialization
                    format that contains all the raw original predictions.
                2. "coco_instances_results.json" a json file in COCO's result format.
                3. "hico_interaction_results.mat" a matlab file
                    used for HICO-DET official evaluation.
        """
        self._tasks = self._tasks_from_config(cfg)
        self._distributed = distributed
        self._output_dir = output_dir
        self._cpu_device = torch.device("cpu")

        self._matlab = cfg.TEST.MATLAB
        self._hico_official_matlab_path = cfg.TEST.HICO_OFFICIAL_MATLAB_PATH
        self._hico_official_anno_file = cfg.TEST.HICO_OFFICIAL_ANNO_FILE
        self._hico_official_bbox_file = cfg.TEST.HICO_OFFICIAL_BBOX_FILE

        self._metadata = MetadataCatalog.get(dataset_name)
        if not hasattr(self._metadata, "json_file"):
            logger.warning(
                f"json_file was not found in MetaDataCatalog for '{dataset_name}'."
                " Trying to convert it to COCO format ..."
            )

            cache_path = os.path.join(output_dir, f"{dataset_name}_coco_format.json")
            self._metadata.json_file = cache_path
            convert_to_coco_json(dataset_name, cache_path)

        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)

        # Test set json files do not contain annotations (evaluation must be
        # performed using the COCO evaluation server).
        self._do_evaluation = "annotations" in self._coco_api.dataset

    def reset(self):
        self._predictions = []

    def _tasks_from_config(self, cfg):
        """
        Returns:
            tuple[str]: tasks that can be evaluated under the given configuration.
        """
        tasks = ("bbox",)
        if cfg.MODEL.HOI_ON:
            tasks = tasks + ("hoi",)
        return tasks

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model (e.g., HOIRCNN). It is a list of dict.
                Each dict corresponds to an image and contains keys
                like "height", "width", "file_name", "image_id".
            outputs: the outputs of a model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            if "box_instances" in output:
                instances = output["box_instances"].to(self._cpu_device)
                prediction["box_instances"] = instances_to_coco_json(instances, input["image_id"])

            if "hoi_instances" in output:
                instances = output["hoi_instances"].to(self._cpu_device)
                prediction["hoi_instances"] = instances_to_hico_matlab(self._metadata, instances)

            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)

            self._predictions.append(prediction)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            logger.warning("[HICOEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(predictions, f)

        self._results = OrderedDict()
        if "proposals" in predictions[0]:
            self._eval_box_proposals(predictions)
        if "box_instances" in predictions[0]:
            self._eval_box_predictions(predictions)
        if "hoi_instances" in predictions[0]:
            self._eval_interactions(predictions)
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_box_proposals(self, predictions):
        """
        Evaluate the box proposals in predictions.
        Fill self._results with the metrics for "box_proposals" task.
        """
        if self._output_dir:
            # Saving generated box proposals to file.
            # Predicted box_proposals are in XYXY_ABS mode.
            bbox_mode = BoxMode.XYXY_ABS.value
            ids, boxes, interactness_logits = [], [], []
            for prediction in predictions:
                ids.append(prediction["image_id"])
                boxes.append(prediction["proposals"].proposal_boxes.tensor.numpy())
                interactness_logits.append(prediction["proposals"].interactness_logits.numpy())

            proposal_data = {
                "boxes": boxes,
                "interactness_logits": interactness_logits,
                "ids": ids,
                "bbox_mode": bbox_mode,
            }
            with PathManager.open(os.path.join(self._output_dir, "box_proposals.pkl"), "wb") as f:
                pickle.dump(proposal_data, f)

        if not self._do_evaluation:
            logger.info("Annotations are not available for evaluation.")
            return

        logger.info("Evaluating bbox proposals ...")
        res = {}
        areas = {"all": "", "small": "s", "medium": "m", "large": "l"}
        for limit in [100, 500]:
            for area, suffix in areas.items():
                stats = _evaluate_box_proposals(predictions, self._coco_api, area=area, limit=limit)
                key = "AR{}@{:d}".format(suffix, limit)
                res[key] = float(stats["ar"].item() * 100)
                for sub_key in ["", "_known", "_novel"]:
                    key = "R{}{}@{:d}+IoU=0.5".format(suffix, sub_key, limit)
                    res[key] = float(stats["recalls{}".format(sub_key)][0].item() * 100)
                    print(" R{}{}@{:d}+IoU@0.5 = {:.3f}".format(suffix, sub_key, limit, res[key]))

        logger.info("Proposal metrics: \n" + create_small_table(res))
        self._results["box_proposals"] = res

    def _eval_box_predictions(self, predictions):
        """
        Evaluate box predictions.
        Fill self._results with the metrics of the tasks.
        """
        logger.info("Preparing results for COCO format ...")
        coco_results = list(itertools.chain(*[x["box_instances"] for x in predictions]))

        # unmap the category ids for objects
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
            for result in coco_results:
                category_id = result["category_id"]
                assert (
                    category_id in reverse_id_mapping
                ), "A prediction has category_id={}, which is not available in the dataset.".format(
                    category_id
                )
                result["category_id"] = reverse_id_mapping[category_id]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(coco_results))
                f.flush()

        if not self._do_evaluation:
            logger.info("Annotations are not available for evaluation.")
            return

        logger.info("Evaluating predictions ...")
        coco_eval = (
            _evaluate_predictions_on_coco(
                self._coco_api, coco_results, "bbox"
            )
            if len(coco_results) > 0
            else None  # cocoapi does not handle empty results very well
        )

        res = self._derive_coco_results(
            coco_eval,
            "bbox",
            iouThr=0.5,
            class_names=self._metadata.get("thing_classes"),
            known_classes=self._metadata.get("known_classes"),
            novel_classes=self._metadata.get("novel_classes"),
        )
        self._results["bbox"] = res

    def _eval_interactions(self, predictions):
        """
        Evaluate predictions on the human-object interactions.
        Fill self._results with the metrics of the tasks.
        """
        logger.info("Preparing results for HICO-DET matlab format ...")
        images = [x["image_id"] for x in predictions]
        results = [x["hoi_instances"] for x in predictions]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "hico_interaction_results.mat")
            file_path = os.path.abspath(file_path)
            logger.info("Saving results to {}".format(file_path))
            write_results_hico_format(images, results, file_path)

        if not self._do_evaluation:
            logger.info("Annotations are not available for evaluation.")
            return

        logger.info("Evaluating interaction using HICO-DET official MATLAB code ...")
        self._evaluate_hico_on_matlab(
            file_path, self._hico_official_anno_file, self._hico_official_bbox_file
        )

    def _evaluate_hico_on_matlab(self, dets_file, anno_file, bbox_file):
        import subprocess
        logger.info('-----------------------------------------------------')
        logger.info('Computing results with the official MATLAB eval code.')
        logger.info('-----------------------------------------------------')
        cmd = 'cd {} && '.format(self._hico_official_matlab_path)
        cmd += '{:s} -nodisplay -nodesktop '.format(self._matlab)
        cmd += '-r "dbstop if error; '
        cmd += 'hico_eval_wrapper(\'{:s}\', \'{:s}\', \'{:s}\'); quit;"'.format(
            dets_file, anno_file, bbox_file
        )
        logger.info('Running:\n{}'.format(cmd))
        subprocess.call(cmd, shell=True)

    def _derive_coco_results(
        self,
        coco_eval,
        iou_type,
        iouThr=None,
        class_names=None, 
        known_classes=None,
        novel_classes=None,
    ):
        """
        Derive the desired score numbers from summarized COCOeval.

        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.

        Returns:
            a dict of {metric name: score}
        """

        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        }[iou_type]

        if coco_eval is None:
            logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # the standard metrics
        results = {
            metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }
        logger.info(
            "Evaluation results for {}: \n".format(iou_type) + create_small_table(results)
        )
        if not np.isfinite(sum(results.values())):
            logger.info("Note that some metrics cannot be computed.")

        if class_names is None or len(class_names) <= 1:
            return results
        if "person" in known_classes:
            known_classes.remove("person")
        # Compute per-category AP
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        results_per_category = []
        results_known_category = [] # Exclude "person" category
        results_novel_category = []
        for idx, name in enumerate(class_names):
            # iou threshold index t: 0.5:0.05:0.9
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            if iouThr is not None:
                t = np.where(iouThr == coco_eval.params.iouThrs)[0]
                precision = precisions[t, :, idx, 0, -1]
            else:
                precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))
            if name in known_classes:
                results_known_category.append(ap * 100)
            if name in novel_classes:
                results_novel_category.append(ap * 100)

        str_suffix = "{:d}".format(int(iouThr*100)) if iouThr else ""
        results_known_novel_split = {
            "AP{}-total".format(str_suffix): np.mean(results_known_category+results_novel_category),
            "AP{}-known".format(str_suffix): np.mean(results_known_category),
            "AP{}-novel".format(str_suffix): np.mean(results_novel_category) \
                if len(results_novel_category) else "nan"
        }

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        logger.info("Per-category {} AP: \n".format(iou_type) + table)
        logger.info(
            "Evaluation results for {}: \n".format(iou_type) + create_small_table(results)
        )
        logger.info(
            "Evaluation results for {} known/novel splits: \n".format(iou_type) + \
            create_small_table(results_known_novel_split)
        )
        results.update({"AP-" + name: ap for name, ap in results_per_category})
        return results


def instances_to_coco_json(instances, img_id):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.

    Args:
        instances (Instances):
        img_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()

    results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": boxes[k],
            "score": scores[k],
        }
        results.append(result)
    return results


def instances_to_hico_matlab(metadata, instances):
    """
    Dump an "Instances" object to a HICO-DET matlab format that's used for evaluation.
    Format: [[hoi_id, person box, object box, score], ...]

    Args:
        metadata ()
        instances (Instances):
        img_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []
    # Meta data
    INTERACTION_CLASSES_TO_ID_MAP = metadata.interaction_classes_to_contiguous_id
    ACTION_CLASSES_META = metadata.action_classes
    THING_CLASSES_META = metadata.thing_classes

    # Note that HICO-DET official evaluation uses BoxMode=XYXY_ABS
    person_boxes = instances.person_boxes.tensor.numpy()
    object_boxes = instances.object_boxes.tensor.numpy()

    person_boxes = person_boxes.tolist()
    object_boxes = object_boxes.tolist()

    scores = instances.scores.tolist()
    object_classes = instances.object_classes.tolist()
    action_classes = instances.action_classes.tolist()

    results = []
    for person_box, object_box, object_id, action_id, score in zip(
        person_boxes, object_boxes, object_classes, action_classes, scores
    ):
        # append detection results
        action_class_name = ACTION_CLASSES_META[action_id]
        object_class_name = THING_CLASSES_META[object_id]
        interaction_name = action_class_name + " " + object_class_name
        if interaction_name in INTERACTION_CLASSES_TO_ID_MAP:
            interaction_id = INTERACTION_CLASSES_TO_ID_MAP[interaction_name]
        else:
            # invalid human-object combinations
            continue

        result = [
            interaction_id,
            person_box[0], person_box[1], person_box[2], person_box[3],
            object_box[0], object_box[1], object_box[2], object_box[3],
            score
        ]
        results.append(result)

    return results


def _evaluate_box_proposals(dataset_predictions, coco_api, thresholds=None, area="all", limit=None):
    """
    Evaluate detection proposal recall metrics. This function is a much
    faster alternative to the official COCO API recall evaluation code. However,
    it produces slightly different results.
    """
    # Record max overlap value for each gt box
    # Return vector of overlap values
    areas = {
        "all": 0,
        "small": 1,
        "medium": 2,
        "large": 3,
        "96-128": 4,
        "128-256": 5,
        "256-512": 6,
        "512-inf": 7,
    }
    area_ranges = [
        [0 ** 2, 1e5 ** 2],  # all
        [0 ** 2, 32 ** 2],  # small
        [32 ** 2, 96 ** 2],  # medium
        [96 ** 2, 1e5 ** 2],  # large
        [96 ** 2, 128 ** 2],  # 96-128
        [128 ** 2, 256 ** 2],  # 128-256
        [256 ** 2, 512 ** 2],  # 256-512
        [512 ** 2, 1e5 ** 2],
    ]  # 512-inf
    assert area in areas, "Unknown area range: {}".format(area)
    area_range = area_ranges[areas[area]]
    gt_overlaps = []
    isknown_obj = []
    num_pos = 0
    num_known = 0
    num_novel = 0

    for prediction_dict in dataset_predictions:
        predictions = prediction_dict["proposals"]

        # sort predictions in descending order
        # TODO maybe remove this and make it explicit in the documentation
        inds = predictions.interactness_logits.sort(descending=True)[1]
        predictions = predictions[inds]

        ann_ids = coco_api.getAnnIds(imgIds=prediction_dict["image_id"])
        anno = coco_api.loadAnns(ann_ids)
        gt_boxes = [
            BoxMode.convert(obj["bbox"], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
            for obj in anno
            if obj["iscrowd"] == 0 and obj["isactive"] == 1
        ]
        gt_boxes = torch.as_tensor(gt_boxes).reshape(-1, 4)  # guard against no boxes
        gt_boxes = Boxes(gt_boxes)
        gt_areas = torch.as_tensor(
            [obj["area"] for obj in anno if obj["iscrowd"] == 0 and obj["isactive"] == 1]
        )
        _isknown = torch.as_tensor(
            [obj["isknown"] for obj in anno if obj["iscrowd"] == 0 and obj["isactive"] == 1]
        )

        if len(gt_boxes) == 0 or len(predictions) == 0:
            continue

        valid_gt_inds = (gt_areas >= area_range[0]) & (gt_areas <= area_range[1])
        gt_boxes = gt_boxes[valid_gt_inds]
        _isknown = _isknown[valid_gt_inds]

        num_pos += len(gt_boxes)

        if len(gt_boxes) == 0:
            continue

        if limit is not None and len(predictions) > limit:
            predictions = predictions[:limit]

        overlaps = pairwise_iou(predictions.proposal_boxes, gt_boxes)

        _gt_overlaps = torch.zeros(len(gt_boxes))
        for j in range(min(len(predictions), len(gt_boxes))):
            # find which proposal box maximally covers each gt box
            # and get the iou amount of coverage for each gt box
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)

            # find which gt box is 'best' covered (i.e. 'best' = most iou)
            gt_ovr, gt_ind = max_overlaps.max(dim=0)
            assert gt_ovr >= 0
            # find the proposal box that covers the best covered gt box
            box_ind = argmax_overlaps[gt_ind]
            # record the iou coverage of this gt box
            _gt_overlaps[j] = overlaps[box_ind, gt_ind]
            assert _gt_overlaps[j] == gt_ovr
            # mark the proposal box and the gt box as used
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1

        # append recorded iou coverage level
        gt_overlaps.append(_gt_overlaps)
        isknown_obj.append(_isknown)

    gt_overlaps = (
        torch.cat(gt_overlaps, dim=0) if len(gt_overlaps) else torch.zeros(0, dtype=torch.float32)
    )
    #gt_overlaps, sort_ids = torch.sort(gt_overlaps)

    isknown_obj = (
        torch.cat(isknown_obj, dim=0) if len(isknown_obj) else torch.zeros(0, dtype=torch.float32)
    )
    #isknown_obj = isknown_obj[sort_ids]

    if thresholds is None:
        step = 0.05
        thresholds = torch.arange(0.5, 0.95 + 1e-5, step, dtype=torch.float32)
    recalls = torch.zeros_like(thresholds)
    # compute recall for each iou threshold
    for i, t in enumerate(thresholds):
        recalls[i] = (gt_overlaps >= t).float().sum() / float(num_pos)
    # ar = 2 * np.trapz(recalls, thresholds)
    ar = recalls.mean()

    # compute recall for known classes
    recalls_known = torch.zeros_like(thresholds)
    known_ids = isknown_obj.nonzero().squeeze()
    for i, t in enumerate(thresholds):
        recalls_known[i] = (gt_overlaps[known_ids] >= t).float().sum() / float(len(known_ids))
    ar_known = recalls_known.mean()

    # compute recall for novel classes
    recalls_novel = torch.zeros_like(thresholds)
    novel_ids = (isknown_obj == 0).nonzero().squeeze()
    for i, t in enumerate(thresholds):
        recalls_novel[i] = (gt_overlaps[novel_ids] >= t).float().sum() / float(len(novel_ids))
    ar_novel = recalls_novel.mean()

    return {
        "ar": ar, "ar_known": ar_known, "ar_novel": ar_novel,
        "recalls": recalls, "recalls_known": recalls_known, "recalls_novel": recalls_novel,
        "thresholds": thresholds,
        "gt_overlaps": gt_overlaps,
        "num_pos": num_pos,
    }


def _evaluate_predictions_on_coco(coco_gt, coco_results, iou_type):
    """
    Evaluate the coco results using COCOEval API.
    """
    assert len(coco_results) > 0

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval


def write_results_hico_format(images, results, file_path):
    """
    Write HICO detection results into .mat file.

    Args:
        images (List[int]): A list of image id
        results (List[List]): A list of detection results, which is saved in a list containing
            [interaction_id, person box, object box, score]
        file_path (String): savefile name
    """
    assert len(images) == len(results)
    dets = [[] for _ in range(len(images))]

    for i, results_per_image in enumerate(results):
        dets[i] = results_per_image

    sio.savemat(file_path, mdict={'dets': dets})


