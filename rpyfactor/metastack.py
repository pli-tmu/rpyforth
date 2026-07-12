"""Stack-fragment constants for rpyfactor."""

import os

NTOP = 2
SPILL_SIZE = 16384

TAG_INT = 0
TAG_OBJ = 1


def _parse_frame_size():
    raw = os.environ.get("RPYFACTOR_FRAME_SIZE")
    if raw is None or raw == "" or raw == "8":
        return 8
    return 8


FRAME_SIZE = _parse_frame_size()
ACTIVE_MAX = NTOP + FRAME_SIZE

USE_STACK_FRAGMENT = bool(os.environ.get("RPYFACTOR_STACK_FRAGMENT"))
# The virtualizable cache is part of the fragment design; opt out only for
# ablation builds with RPYFACTOR_STACK_VABLE=0.
_vable_raw = os.environ.get("RPYFACTOR_STACK_VABLE")
USE_STACK_VABLE = USE_STACK_FRAGMENT and _vable_raw != "0"

# The Interpreter itself is the virtualizable (rpyforth pattern): the JIT only
# keeps the cache in registers when every access flows through the portal's
# red variable, so the cache fields live directly on the Interpreter -- never
# behind a separate stack object re-fetched via a field (that read escapes the
# virtualizable and forces it on every primitive). Object cells are
# virtualizable too, like PyPy's fastlocals_w[*]; cs_ptr rides along so call
# entry/return does not write memory.
STACK_FRAGMENT_VIRTUALIZABLES = [
    "t0i", "t1i", "t0t", "t1t", "t0o", "t1o", "d",
    "frame_i[*]", "frame_t[*]", "frame_o[*]",
    "frag_ptr", "spill_ptr",
    "cs_ptr",
]
