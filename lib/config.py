from detectron2.config import CfgNode as CN


def add_hoircnn_default_config(cfg):
    """
    Add default config for our HOI detection model.
    """
    cfg.MODEL.HOI_ON = True
    # ------------------------------------------------------------------------ #
    # HORPN options
    # ------------------------------------------------------------------------ #
    cfg.MODEL.HORPN = CN()

    # Anchor parameters
    cfg.MODEL.HORPN.IN_FEATURES = ["p3", "p4", "p5", "p6"]

    # Remove RPN anchors that go outside the image by BOUNDARY_THRESH pixels
    # Set to -1 or a large value, e.g. 100000, to disable pruning anchors
    cfg.MODEL.HORPN.BOUNDARY_THRESH = -1

    # IOU overlap ratios [BG_IOU_THRESHOLD, FG_IOU_THRESHOLD]
    # Minimum overlap required between an anchor and ground-truth box for the
    # (anchor, gt box) pair to be a positive example (IoU >= FG_IOU_THRESHOLD
    # ==> positive RPN example: 1)
    # Maximum overlap allowed between an anchor and ground-truth box for the
    # (anchor, gt box) pair to be a negative examples (IoU < BG_IOU_THRESHOLD
    # ==> negative RPN example: 0)
    # Anchors with overlap in between (BG_IOU_THRESHOLD <= IoU < FG_IOU_THRESHOLD)
    # are ignored (-1)
    cfg.MODEL.HORPN.IOU_THRESHOLDS = [0.5, 0.7]
    cfg.MODEL.HORPN.IOU_LABELS = [0, -1, 1]

    # Total number of RPN examples per image
    cfg.MODEL.HORPN.BATCH_SIZE_PER_IMAGE = 512

    # Target fraction of foreground (positive) examples per RPN minibatch
    cfg.MODEL.HORPN.POSITIVE_FRACTION = 0.5

    # Weights on (dx, dy, dw, dh) for normalizing RPN anchor regression targets
    cfg.MODEL.HORPN.BBOX_REG_WEIGHTS = (1.0, 1.0, 1.0, 1.0)

    # The transition point from L1 to L2 loss. Set to 0.0 to make the loss simply L1.
    cfg.MODEL.HORPN.SMOOTH_L1_BETA = 0.0
    cfg.MODEL.HORPN.LOSS_WEIGHT = 1.0

    # Number of top scoring RPN proposals to keep before applying NMS
    # When FPN is used, this is *per FPN level* (not total)
    cfg.MODEL.HORPN.PRE_NMS_TOPK_TRAIN = 2000
    cfg.MODEL.HORPN.PRE_NMS_TOPK_TEST = 1000
    # Number of top scoring RPN proposals to keep after applying NMS
    # When FPN is used, this limit is applied per level and then again to the union
    # of proposals from all levels
    # NOTE: When FPN is used, the meaning of this config is different from Detectron1.
    # It means per-batch topk in Detectron1, but per-image topk here.
    # See "modeling/horpn_outputs.py" for details.
    cfg.MODEL.HORPN.POST_NMS_TOPK_TRAIN = 200
    cfg.MODEL.HORPN.POST_NMS_TOPK_TEST = 200

    # NMS threshold used on RPN proposals
    cfg.MODEL.HORPN.NMS_THRESH = 0.7

    # Ratio of object proposals to the total proposals. If the `post_nms_topk_train` = 200,
    # then there will be 190 object proposals and 10 person proposals.
    cfg.MODEL.HORPN.RATIO_OBJECTS = 0.95

    # Number of top scoring person proposals to reason interactness scores of objects.
    cfg.MODEL.HORPN.TOPK_PERSON_CELLS = 8

    # Number of convs in HORPN and its channels.
    cfg.MODEL.HORPN.NUM_CONV = 1
    cfg.MODEL.HORPN.CONV_DIM = 256
    
    # Number of fully connected layers in HORPN and its channels.
    cfg.MODEL.HORPN.NUM_RN_FC = 2
    cfg.MODEL.HORPN.RN_FC_DIM = 256

    # ---------------------------------------------------------------------------- #
    # Additional ANCHOR_GENERATOR options
    # ---------------------------------------------------------------------------- #
    # One size for each in feature map.
    # This is to match the HORPN in features ["p3", "p4", "p5", "p6"].
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[64], [128], [256], [512]]
    # Three aspect ratios (same for all in feature maps)
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]

    # ---------------------------------------------------------------------------- #
    # Additional ROI HEADS options
    # ---------------------------------------------------------------------------- #
    # Number of action classes
    cfg.MODEL.ROI_HEADS.NUM_ACTIONS = 117
    # RoI minibatch size *per image* (number of regions of interest [ROIs])
    # Total number of RoIs per training minibatch =
    #   ROI_HEADS.BATCH_SIZE_PER_IMAGE * SOLVER.IMS_PER_BATCH
    # *Note*: We replace original `BATCH_SIZE_PER_IMAGE` with `BOX_BATCH_SIZE_PER_IMAGE`
    # to differentiate the batch size used for box head and hoi head.
    cfg.MODEL.ROI_HEADS.BOX_BATCH_SIZE_PER_IMAGE = 256
    # Target fraction of RoI minibatch that is labeled foreground (i.e. class > 0)
    # *Note*: We replace original `POSITIVE_FRACTION` with `BOX_POSITIVE_FRACTION`
    # to differentiate the `POSITIVE_FRACTION` used for box head and hoi head.
    cfg.MODEL.ROI_HEADS.BOX_POSITIVE_FRACTION = 0.25
    # Minibatch size *per image* (number of human-object pairs)
    # Given the RoIs per training minibatch (`BOX_BATCH_SIZE_PER_IMAGE`), we split RoIs
    # into N `person`s and M` object`s (N + M = `BOX_BATCH_SIZE_PER_IMAGE`), and contruct
    # N * M person-object pairs. `HOI_BATCH_SIZE_PER_IMAGE` = The number of pairs per image.
    # Total number of pairs per training minibatch =
    #   ROI_HEADS.HOI_BATCH_SIZE_PER_IMAGE * SOLVER.IMS_PER_BATCH
    cfg.MODEL.ROI_HEADS.HOI_BATCH_SIZE_PER_IMAGE = 128
    # Target fraction of person-object pairs that is labeled foreground (i.e. interaction > 0)
    cfg.MODEL.ROI_HEADS.HOI_POSITIVE_FRACTION = 1.0
    # Minimum score threshold (assuming scores in a [0, 1] range); a value chosen to
    # balance obtaining high recall with not having too many low precision detections
    # that will slow down inference post processing steps (like NMS)
    # A default threshold of 0.0 increases AP by ~0.2-0.3 but significantly slows down inference.
    cfg.MODEL.ROI_HEADS.HOI_SCORE_THRESH_TEST = 0.001

    # ------------------------------------------------------------------------ #
    # HOI HEADS options
    # ------------------------------------------------------------------------ #
    cfg.MODEL.HOI_BOX_HEAD = CN()
    # Options for HOI_BOX_HEAD models: HOIRCNNConvFCHead,
    cfg.MODEL.HOI_BOX_HEAD.NAME = ""
    cfg.MODEL.HOI_BOX_HEAD.NUM_FC = 2
    # Hidden layer dimension for FC layers in the RoI box head
    cfg.MODEL.HOI_BOX_HEAD.FC_DIM = 512 #1024

    # Type of pooling operation applied to the incoming feature map for each RoI
    cfg.MODEL.HOI_BOX_HEAD.POOLER_TYPE = "ROIAlignV2"
    cfg.MODEL.HOI_BOX_HEAD.POOLER_RESOLUTION = 7
    cfg.MODEL.HOI_BOX_HEAD.POOLER_SAMPLING_RATIO = 0

    # Default weights on positive interactions for interaction classification
    # These are empirically chosen to approximately lead to balanced
    # positive v.s. negative samples. 
    cfg.MODEL.HOI_BOX_HEAD.ACTION_CLS_WEIGHTS = [1., 10.]

    # In some datasets, there are person to person interactions.
    # This option allows the model to handle this case.
    cfg.MODEL.HOI_BOX_HEAD.ALLOW_PERSON_TO_PERSON = False

    # ---------------------------------------------------------------------------- #
    # ZERO-SHOT INFERENCE options
    # ---------------------------------------------------------------------------- #
    cfg.ZERO_SHOT = CN()
    # Enable the zero-shot inference.
    cfg.ZERO_SHOT.ZERO_SHOT_ON = False
    
    # Interested novel classes to detect.
    cfg.ZERO_SHOT.NOVEL_CLASSES = [""]
    
    # Threshold (assuming scores in a [0, 1] range) to activate zero-shot inference.
    # Only boxes whose score of known classes < PRE_INFERENCE_THRESH will be passed
    # to infer the score of novel classes.
    cfg.ZERO_SHOT.PRE_INFERENCE_THRESH = 0.3
    
    # Minimum score threshold (assuming scores in a [0, 1] range) to keep
    # a box prediction of novel classes
    cfg.ZERO_SHOT.POST_INFERENCE_THRESH = 0.25

    # The number of known classes used to infer the score of novel classes.
    cfg.ZERO_SHOT.TOPK_KNOWN_CLASSES = 4

    # Word corpus including semantic embeddings. Note: only *Glove* is supported to date.
    cfg.ZERO_SHOT.SEMANTIC_CORPUS = "./datasets/Glove/glove.6B.300d.txt"
    
    # Files to load/save pre-computed semantic embeddings. It contains a
    # dict[novel class name (str): semantic embedding (np.array)]
    cfg.ZERO_SHOT.PRECOMPUTED_SEMANTIC_EMBEDDINGS = ""
    
    # A value chosen to keep the maximum number of box detections for novel objects
    cfg.ZERO_SHOT.DETECTIONS_PER_IMAGE = 3

    # ---------------------------------------------------------------------------- #
    # Additional TEST options
    # ---------------------------------------------------------------------------- #
    # A value chosen to keep the maximum number of HOI interaction per image.
    # A large value may increase mAP but significantly slows down inference.
    cfg.TEST.INTERACTIONS_PER_IMAGE = 100
    
    # The path to official HICO-DET evaluation matlab code.
    cfg.TEST.HICO_OFFICIAL_MATLAB_PATH = "./lib/evaluation/HICO_MATLAB_EVAL"
    
    # The path to original HICO-DET annotation files. This is used to evaluate
    # the HOI detection performance.
    cfg.TEST.HICO_OFFICIAL_ANNO_FILE = "/raid1/suchen/dataset/hico_20160224_det/anno.mat"
    cfg.TEST.HICO_OFFICIAL_BBOX_FILE = "/raid1/suchen/dataset/hico_20160224_det/anno_bbox.mat"

    # Name (or path to) the matlab executable
    cfg.TEST.MATLAB = 'matlab'