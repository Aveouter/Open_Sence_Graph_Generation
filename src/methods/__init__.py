
from .hstrnet_method import HSTRNet_Method
from .reltr_method import RelTR_Method
from .egtr_method import EGTR_Method
from .flowsg_method import FlowSG_Method
from .motifs_method import Motifs_Method
from .vctree_method import VCTree_Method
from .tde_method import TDE_Method
from .cvc_method import CVC_Method

# 规范method name 为小写
method_maps = {
    "hstrnet": HSTRNet_Method,
    "reltr": RelTR_Method,
    "egtr": EGTR_Method,
    "flowsg": FlowSG_Method,
    "motifs": Motifs_Method,
    "vctree": VCTree_Method,
    "tde": TDE_Method,
    "cvc": CVC_Method,
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
]