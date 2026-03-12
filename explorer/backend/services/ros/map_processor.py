"""
services/ros/map_processor.py
──────────────────────────────
LiDAR bag → occupancy map diff with IoU score.
Source: site_commander/backend/map_processor.py
"""
from __future__ import annotations

import base64
import math
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple

from core.logging import get_logger

logger = get_logger(__name__)

try:
    import cv2
    import numpy as np
    from PIL import Image
    from rosbags.rosbag1 import Reader
    from rosbags.serde import deserialize_cdr, ros1_to_cdr
    _DEPS_OK = True
except ImportError as e:
    logger.warning("map_processor: missing dependency (%s) — MapProcessor disabled.", e)
    _DEPS_OK = False


def process_bag_for_changes(
    bag_path: Path,
    original_map_b64: str,
    resolution: float = 0.05,
    origin: Optional[list] = None,
) -> Tuple[Optional[str], int]:
    """
    Compare a new `.bag` LiDAR scan against a stored occupancy map.

    Returns:
        (base64_png_diff, iou_score_0_to_100)
        diff image:  gray = matched, red = new obstacles, cyan = missing obstacles
    """
    if not _DEPS_OK:
        return None, 0

    if origin is None:
        origin = [0, 0, 0]

    # 1. Decode stored map
    try:
        if "," in original_map_b64:
            original_map_b64 = original_map_b64.split(",")[1]
        orig_bytes = base64.b64decode(original_map_b64)
        map_orig   = np.array(Image.open(BytesIO(orig_bytes)).convert("L"))
        h, w       = map_orig.shape
    except Exception as e:
        logger.error("map_processor: map decode error: %s", e)
        return None, 0

    # 2. New canvas (white = free space)
    map_new = np.full((h, w), 255, dtype=np.uint8)
    ox, oy  = origin[0], origin[1]

    # 3. Read bag, plot scan endpoints
    try:
        with Reader(bag_path) as reader:
            scan_conns = [c for c in reader.connections if "scan" in c.topic]
            if not scan_conns:
                logger.warning("map_processor: no /scan topic found in %s", bag_path)
            else:
                conn  = scan_conns[0]
                count = 0
                for _, _, rawdata in reader.messages(connections=[conn]):
                    msg = deserialize_cdr(ros1_to_cdr(rawdata, conn.msgtype), conn.msgtype)
                    count += 1
                    if count > 5000:
                        break
                    cx    = int((0 - ox) / resolution)
                    cy    = int(h - ((0 - oy) / resolution))
                    angle = msg.angle_min
                    for r in msg.ranges:
                        if msg.range_min < r < msg.range_max and not math.isinf(r):
                            dist_px = r / resolution
                            px = int(cx + dist_px * math.cos(angle))
                            py = int(cy - dist_px * math.sin(angle))
                            if 0 <= px < w and 0 <= py < h:
                                map_new[py, px] = 0
                        angle += msg.angle_increment
    except Exception as e:
        logger.error("map_processor: bag read error: %s", e)
        return None, 0

    # 4. Binary threshold + IoU
    _, bin_orig = cv2.threshold(map_orig, 128, 255, cv2.THRESH_BINARY_INV)
    _, bin_new  = cv2.threshold(map_new,  128, 255, cv2.THRESH_BINARY_INV)
    intersection = cv2.bitwise_and(bin_orig, bin_new)
    union        = cv2.bitwise_or(bin_orig,  bin_new)
    score        = 0
    if np.count_nonzero(union) > 0:
        score = int((np.count_nonzero(intersection) / np.count_nonzero(union)) * 100)

    # 5. RGBA diff image
    diff_new  = cv2.subtract(bin_new,  bin_orig)
    diff_gone = cv2.subtract(bin_orig, bin_new)
    result    = np.zeros((h, w, 4), dtype=np.uint8)
    result[bin_orig > 0] = [100, 100, 100, 255]   # gray = matched
    result[diff_new  > 0] = [255,   0,   0, 255]  # red  = new obstacles
    result[diff_gone > 0] = [  0, 255, 255, 255]  # cyan = missing obstacles

    buf = BytesIO()
    Image.fromarray(result).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8"), score
