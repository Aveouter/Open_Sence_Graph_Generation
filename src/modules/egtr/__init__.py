# src/modules/egtr/__init__.py
"""
EGTR: Extracting Graph from Transformer for Scene Graph Generation.
CVPR 2024 Best Paper Award Candidate.

This module contains the EGTR model implementation, including:
- deformable_detr.py: Modified Deformable DETR backbone (returns attention byproducts)
- egtr.py: Scene graph generation model and loss
- util.py: Loss functions (dice, focal, giou) and NestedTensor
- load_custom.py: CUDA kernel loading for deformable attention
- custom_kernel/: CUDA/C++ kernels
"""
