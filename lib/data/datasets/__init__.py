from . import builtin  # ensure the builtin datasets are registered
from .hico import load_hico_json
from .hico_meta import HICO_OBJECTS
from .vcoco_meta import VCOCO_OBJECTS

__all__ = [k for k in globals().keys() if "builtin" not in k and not k.startswith("_")]
