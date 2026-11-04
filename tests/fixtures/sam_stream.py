"""Build SAM 3.1 Responses-API events in the real wire grammar (as in https://github.com/meta-models/meta-model-cookbook)."""
import json

import numpy as np
from meta_sam_parser._mask_codec import _encode_raster


def sam_line(frame, obj, left, top, mask: np.ndarray, frame_w=640, frame_h=480):
    h, w = mask.shape
    payload = _encode_raster([int(b) for b in mask.ravel()], w, h)
    box = f"x1={left};y1={top};x2={left + w - 1};y2={top + h - 1};w={frame_w};h={frame_h}"
    return f"<{frame}f>{obj}<|box;{box}|><|mask;x=0;y=0;data={h},{w},{payload}|>"


def sam_events(lines):
    text = "\n".join(lines) + "\n"
    lane = {"item_id": "msg_1", "output_index": 0, "content_index": 0}
    return [
        {"type": "response.created", "sequence_number": 0,
         "response": {"id": "resp_1", "object": "response", "created_at": 0, "status": "in_progress",
                      "model": "sam-3.1", "output": [], "parallel_tool_calls": False, "tool_choice": "auto", "tools": []}},
        {"type": "response.output_text.delta", "sequence_number": 1, "delta": text, "logprobs": [], **lane},
        {"type": "response.content_part.done", "sequence_number": 2,
         "part": {"type": "output_text", "text": text, "annotations": []}, **lane},
        {"type": "response.completed", "sequence_number": 3,
         "response": {"id": "resp_1", "object": "response", "created_at": 0, "status": "completed",
                      "model": "sam-3.1", "output": [], "parallel_tool_calls": False, "tool_choice": "auto", "tools": []}},
    ]


def sse(events) -> bytes:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()
