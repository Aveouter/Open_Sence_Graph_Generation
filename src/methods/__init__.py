
from .hstrnet_method import HSTRNet_Method
from .reltr_method import RelTR_Method
from .egtr_method import EGTR_Method

# 规范method name 为小写
method_maps = {
    "hstrnet": HSTRNet_Method,
    "reltr": RelTR_Method,
    "egtr": EGTR_Method,
}

__all__ = [
    "hstrnet_method",
    "reltr_method",
    "egtr_method",
]