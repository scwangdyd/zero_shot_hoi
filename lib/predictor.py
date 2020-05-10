# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import atexit
import bisect
import numpy as np
import multiprocessing as mp
from collections import deque
import cv2
import torch

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode

from lib.data import datasets # ensure the builtin datasets are registered
from lib.utils.visualizer import InteractionVisualizer
from lib.utils.video_visualizer import VideoVisualizer


def create_visualization_metadata(cfg):
    """
    Create a "MetadataCatelog" for visualization.
    Args:
        cfg (CfgNode)
    Returns:
        visualization_metadata (MetadataCatalog): it contains
            *
            *
            *
    """
    metadata = MetadataCatalog.get(
        cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
    )
    known_classes = metadata.get("known_classes", [])
    thing_classes = metadata.get("thing_classes", [])
    thing_colors = metadata.get("thing_colors", [])
    if thing_colors:
        for name, color in thing_colors.items():
            thing_colors[name] = np.asarray(color) / 255 if max(color) > 1 else np.asarray(color)

    visualization_metadata = MetadataCatalog.get("visualization")
    visualization_metadata.set(
        known_classes=known_classes, thing_classes=thing_classes, thing_colors=thing_colors
    )

    if cfg.MODEL.HOI_ON:
        action_classes = metadata.get("action_classes", None)
        interaction_to_contiguous_id = metadata.get("interaction_classes_to_contiguous_id", None)
        contiguous_id_to_interaction = None
        if interaction_to_contiguous_id:
            contiguous_id_to_interaction = {v: k for k, v in interaction_to_contiguous_id.items()}
        visualization_metadata.set(
            action_classes=action_classes,
            interaction_to_contiguous_id=interaction_to_contiguous_id,
            contiguous_id_to_interaction=contiguous_id_to_interaction
        )
        
    if cfg.ZERO_SHOT.ZERO_SHOT_ON:
        novel_classes_from_dataset = metadata.get("novel_classes", [])
        novel_classes_from_args = cfg.ZERO_SHOT.NOVEL_CLASSES
        novel_classes = novel_classes_from_args + novel_classes_from_dataset
        # draw novel objects in red boxes
        novel_colors = [np.asarray([1.0, 0, 0]) for _ in novel_classes]

        thing_classes += novel_classes
        thing_colors.update({cls: color for cls, color in zip(novel_classes, novel_colors)})
        visualization_metadata.set(thing_classes=thing_classes, thing_colors=thing_colors)
    return visualization_metadata


class VisualizationDemo(object):
    def __init__(self, cfg, args, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode
        self.draw_proposals = args.draw_proposals
        self.thresh = args.confidence_threshold
        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)
        
        self._init_visualization_metadata(cfg, args)
        
    def _init_visualization_metadata(self, cfg, args):
        """
        Initilize visualizer.
        Args:
            cfg (CfgNode)
        """
        self.metadata = create_visualization_metadata(cfg)

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = InteractionVisualizer(image, self.metadata, instance_mode=self.instance_mode)
        if self.draw_proposals:
            instances = predictions["proposals"].to(self.cpu_device)
            vis_output = visualizer.draw_proposals(proposals=instances)
        elif "hoi_instances" in predictions:
            instances = predictions["hoi_instances"].to(self.cpu_device)
            instances = self._convert_hoi_instances(instances)
            vis_output = visualizer.draw_interaction_predictions(predictions=instances)
        elif "box_instances" in predictions:
            instances = predictions["box_instances"].to(self.cpu_device)
            vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def run_on_video(self, video):
        """
        Visualizes predictions on frames of the input video.
        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.
        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictions):
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if self.draw_proposals:
                instances = predictions["proposals"].to(self.cpu_device)
                vis_frame = video_visualizer.draw_proposals(frame, instances, self.thresh)
            elif "hoi_instances" in predictions:
                instances = predictions["hoi_instances"].to(self.cpu_device)
                instances = self._convert_hoi_instances(instances)               
                vis_frame = video_visualizer.draw_interaction_predictions(frame, instances)
            elif "box_instances" in predictions:
                instances = predictions["box_instances"].to(self.cpu_device)
                instances = self._convert_hoi_instances(instances)               
                vis_frame = video_visualizer.draw_instance_predictions(frame, instances)

            # Converts Matplotlib RGB format to OpenCV BGR format
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            return vis_frame

        frame_gen = self._frame_from_video(video)
        if self.parallel:
            buffer_size = self.predictor.default_buffer_size

            frame_data = deque()

            for cnt, frame in enumerate(frame_gen):
                frame_data.append(frame)
                self.predictor.put(frame)

                if cnt >= buffer_size:
                    frame = frame_data.popleft()
                    predictions = self.predictor.get()
                    yield process_predictions(frame, predictions)

            while len(frame_data):
                frame = frame_data.popleft()
                predictions = self.predictor.get()
                yield process_predictions(frame, predictions)
        else:
            for frame in frame_gen:
                yield process_predictions(frame, self.predictor(frame))
                
    def _convert_hoi_instances(self, instances):
        """
        Convert an "Instances" object to a HOI "Instances" by merging the predicted
        object class and action class to an interaction class.
        For example, object ("bench") + action ("sit on") -> interaction ("sit on bench")
        """
        num_instance = len(instances)
        if num_instance == 0:
            return instances
        # Meta data
        interaction_to_contiguous_id = self.metadata.get("interaction_to_contiguous_id", None)
        
        if interaction_to_contiguous_id:
            action_classes = self.metadata.get("action_classes", None)
            thing_classes = self.metadata.get("thing_classes", None)
            known_classes = self.metadata.get("known_classes", None)
            novel_classes = np.setdiff1d(thing_classes, known_classes).tolist()
        
            pred_object_classes = instances.object_classes.tolist()
            pred_action_classes = instances.action_classes.tolist()

            interaction_classes = []
            keep = []
            for ix in range(num_instance):
                object_id = pred_object_classes[ix]
                action_id = pred_action_classes[ix]
                # append detection results
                pred_action_name = action_classes[action_id]
                pred_object_name = thing_classes[object_id]
                pred_interaction_name = pred_action_name + " " + pred_object_name
                if pred_interaction_name in interaction_to_contiguous_id:
                    #interaction_id = interaction_to_contiguous_id[pred_interaction_name]
                    interaction_classes.append(pred_interaction_name)
                    keep.append(ix)
                elif pred_object_name in novel_classes:
                    # TODO: mine valid interaction with novel objects using external source.
                    # Interactions with novel object categories
                    interaction_classes.append(pred_interaction_name)
                    keep.append(ix)

            instances = instances[keep]
            instances.pred_classes = np.asarray(interaction_classes)
        return instances


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5