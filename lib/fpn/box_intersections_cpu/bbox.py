"""Portable NumPy fallback for the optional Cython bbox extension.

The Cython extension built from ``bbox.pyx`` is faster, but source checkouts and
CI jobs should not require checked-in platform-specific ``.so`` files.
"""

import numpy as np


def bbox_overlaps(boxes, query_boxes):
    """Return IoU overlaps with Fast R-CNN's inclusive ``+1`` box convention."""
    boxes = np.ascontiguousarray(boxes, dtype=np.float64)
    query_boxes = np.ascontiguousarray(query_boxes, dtype=np.float64)
    n = boxes.shape[0]
    k = query_boxes.shape[0]
    overlaps = np.zeros((n, k), dtype=np.float64)

    for query_idx in range(k):
        query = query_boxes[query_idx]
        query_area = (query[2] - query[0] + 1.0) * (query[3] - query[1] + 1.0)
        for box_idx in range(n):
            box = boxes[box_idx]
            iw = min(box[2], query[2]) - max(box[0], query[0]) + 1.0
            if iw <= 0:
                continue
            ih = min(box[3], query[3]) - max(box[1], query[1]) + 1.0
            if ih <= 0:
                continue
            box_area = (box[2] - box[0] + 1.0) * (box[3] - box[1] + 1.0)
            union = box_area + query_area - iw * ih
            if union > 0:
                overlaps[box_idx, query_idx] = iw * ih / union
    return overlaps
