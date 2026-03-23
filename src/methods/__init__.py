
from .hstrnet_method import HSTRNet_Method
from .reltr_method import RelTR_Method

# 规范method name 为小写
method_maps = {
    "hstrnet": HSTRNet_Method,
    "RelTR": RelTR_Method,
}

__all__ = [
    "hstrnet_method",
    "RelTR",
]