from .hstrnet_method import HSTRNet_Method
from .reltr_method import RelTR_Method
from .egtr_method import EGTR_Method
from .flowsg_method import FlowSG_Method
from .motifs_method import Motifs_Method
from .vctree_method import VCTree_Method
from .tde_method import TDE_Method
from .cvc_method import CVC_Method
from .imp_method import IMP_Method
from .transformer_method import TransformerSGG_Method
from .gpsnet_method import GPSNet_Method
from .penet_method import PENet_Method
from .react_method import REACT_Method
from .squat_method import Squat_Method
from .shagcl_method import SHAGCL_Method
from .usg_method import USG_Method

# 规范method name 为小写
method_maps = {
    "hstrnet": HSTRNet_Method,
    "reltr": RelTR_Method,
    "egtr": EGTR_Method,
    "flowsg": FlowSG_Method,
    "usg": USG_Method,
    "motifs": Motifs_Method,
    "vctree": VCTree_Method,
    "tde": TDE_Method,
    "cvc": CVC_Method,
    "imp": IMP_Method,
    "transformer": TransformerSGG_Method,
    "gpsnet": GPSNet_Method,
    "penet": PENet_Method,
    "react": REACT_Method,
    "squat": Squat_Method,
    "shagcl": SHAGCL_Method,
}

__all__ = [
    "hstrnet_method",
    "reltr_method",
    "egtr_method",
    "flowsg_method",
    "motifs_method",
    "vctree_method",
    "tde_method",
    "cvc_method",
    "imp_method",
    "transformer_method",
    "gpsnet_method",
    "penet_method",
    "react_method",
    "squat_method",
    "shagcl_method",
    "usg_method",
]
