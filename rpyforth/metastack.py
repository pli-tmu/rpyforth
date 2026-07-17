"""Metastack constants and the three-tier data-stack layout:

  * the top NTOP cells in scalar fields (stack caching, Ertl 1995),
  * the next FRAME_SIZE cells in a small virtualizable frame array,
  * everything deeper in the shared spill area.
"""

import os

NTOP = 2


def _uint_env(name):
    """Value of a non-negative decimal env var, or -1 if it is unset or not a
    plain decimal integer. Evaluated once at import time to bake in a constant,
    so plain host-Python semantics are fine here."""
    raw = os.environ.get(name)
    if not raw:
        return -1
    n = 0
    for ch in raw:
        if not ("0" <= ch <= "9"):
            return -1
        n = n * 10 + (ord(ch) - ord("0"))
    return n


def _env_is(name, value):
    return os.environ.get(name) == value


def _env_set(name):
    return bool(os.environ.get(name))


# FRAME_SIZE: cells cached in the virtualizable frame array (default 8,
# clamped to [1, 64]).
_frame_size = _uint_env("RPYFORTH_FRAME_SIZE")
if _frame_size < 1:
    FRAME_SIZE = 8
elif _frame_size > 64:
    FRAME_SIZE = 64
else:
    FRAME_SIZE = _frame_size
ACTIVE_MAX = NTOP + FRAME_SIZE

# EFFECTIVE_NTOP: number of scalar tops actually in effect. The flagship uses
# NTOP=2; the parametric-NTOP variant overrides it to one of 2/4/8/16.
_requested_ntop = _uint_env("RPYFORTH_NTOP")
if _requested_ntop in (2, 4, 8, 16):
    EFFECTIVE_NTOP = _requested_ntop
else:
    EFFECTIVE_NTOP = NTOP

CALL_WINDOW = NTOP

# Metastack capacity
FRAG_DEPTH = 16384
SPILL_SIZE = 16384

# Legacy constants kept for import compatibility with the float/object stubs.
FRAGMENT_SIZE = 256
TOP_CACHE_SIZE = 4
STACK_SIZE = FRAGMENT_SIZE

# The stack fragment is the master switch; every cache variant below requires
# it. With it off, the interpreter uses the plain array-backed stacks.
USE_STACK_FRAGMENT = _env_set("RPYFORTH_STACK_FRAGMENT")
STACK_FRAGMENT_STRICT = _env_set("RPYFORTH_STACK_FRAGMENT_STRICT")

# The int cache has three mutually exclusive layouts, resolved here in priority
# order so no later fixups are needed:
#   * frame-only (RPYFORTH_FRAME_ONLY=1, NTOP=0): every cached cell lives in the
#     virtualizable frame[*] array, no scalar tops.
#   * parametric-NTOP (RPYFORTH_NTOP set): EFFECTIVE_NTOP scalar tops.
#   * default: the two scalar tops t0, t1 plus the frame array.
# The float fragment is an opt-in addon (measured a net loss) that rides only on
# the default layout.
USE_FRAME_ONLY = USE_STACK_FRAGMENT and _env_is("RPYFORTH_FRAME_ONLY", "1")
USE_NTOP_VARIANT = (USE_STACK_FRAGMENT and not USE_FRAME_ONLY
                    and _env_set("RPYFORTH_NTOP"))
USE_FLOAT_FRAGMENT = (USE_STACK_FRAGMENT and not USE_FRAME_ONLY
                      and not USE_NTOP_VARIANT
                      and _env_is("RPYFORTH_FLOAT_FRAGMENT", "1"))


class DataStackOverflow(Exception):
    pass


class DSFragment(object):
    pass


class DSMetaStack(object):
    pass

if USE_FRAME_ONLY:
    _INT_CACHE_VIRTUALIZABLES = ["cache_depth", "frame[*]", "frag_ptr", "spill_ptr"]
elif USE_NTOP_VARIANT:
    _INT_CACHE_VIRTUALIZABLES = (["t%d" % i for i in range(EFFECTIVE_NTOP)] +
                                 ["cache_depth", "frame[*]", "frag_ptr", "spill_ptr"])
else:
    _INT_CACHE_VIRTUALIZABLES = ["t0", "t1", "cache_depth", "frame[*]",
                                 "frag_ptr", "spill_ptr"]

_STACK_FRAGMENT_VIRTUALIZABLES_BASE = _INT_CACHE_VIRTUALIZABLES + [
    "rs_ptr",
    "cs_pcs", "cs_ptr", "cs_base",
    "li",
    "cell_size", "cell_size_bytes", "base",
    "ds_ptr_locals",
]

if USE_FLOAT_FRAGMENT:
    STACK_FRAGMENT_VIRTUALIZABLES = _STACK_FRAGMENT_VIRTUALIZABLES_BASE + [
        "ft0", "ft1", "fdep", "fframe[*]",
        "ffrag_ptr", "fspill_ptr",
    ]
else:
    STACK_FRAGMENT_VIRTUALIZABLES = _STACK_FRAGMENT_VIRTUALIZABLES_BASE + [
        "ds_floats", "ds_ptr_floats",
    ]


def push_ds_fragments(state):
    if USE_STACK_FRAGMENT:
        state.push_fragment_on()
        if USE_FLOAT_FRAGMENT:
            state.push_float_fragment_on()


def pop_ds_fragments_commit(state):
    if USE_STACK_FRAGMENT:
        state.pop_fragment_commit_on()
        if USE_FLOAT_FRAGMENT:
            state.pop_float_fragment_commit_on()


def reset_ds_fragments(state):
    if USE_STACK_FRAGMENT:
        state.reset_on()
        if USE_FLOAT_FRAGMENT:
            state.freset_on()
