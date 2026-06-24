import os

NTOP = 2
FRAME_SIZE = 8
ACTIVE_MAX = NTOP + FRAME_SIZE          # active-fragment capacity (== 11)

# Import window on a non-tail call: the callee inherits the caller's top NTOP
# cells (already in the scalar tops -- free), and the caller's deeper frame
# cells are parked in the metastack and restored on return. Kept as CALL_WINDOW
# for import compatibility with tests/stubs.
CALL_WINDOW = NTOP

# Metastack capacity: one parked record per nested word call (frag_ptr indexes
# it, mirroring the call stack), and a flat arena bounding total parked cells.
FRAG_DEPTH = 16384
SPILL_SIZE = 16384

# Legacy constants kept for import compatibility with the float/object stubs.
FRAGMENT_SIZE = 256
TOP_CACHE_SIZE = 4
STACK_SIZE = FRAGMENT_SIZE

USE_STACK_FRAGMENT = bool(os.environ.get("RPYFORTH_STACK_FRAGMENT"))
STACK_FRAGMENT_STRICT = bool(os.environ.get("RPYFORTH_STACK_FRAGMENT_STRICT"))


class DataStackOverflow(Exception):
    pass


class DSFragment(object):
    pass


class DSMetaStack(object):
    pass

STACK_FRAGMENT_VIRTUALIZABLES = [
    "t0", "t1", "d", "frame[*]",
    "frag_ptr", "spill_ptr",
    "rs", "rs_ptr",
    "cs_threads", "cs_ips", "cs_ptr",
    "li",
    "cell_size", "cell_size_bytes", "base",
    "ds_floats", "ds_ptr_floats",
    "ds_locals", "ds_ptr_locals",
]


def push_ds_fragments(state):
    if USE_STACK_FRAGMENT:
        state.push_fragment_on()


def pop_ds_fragments_commit(state):
    if USE_STACK_FRAGMENT:
        state.pop_fragment_commit_on()


def reset_ds_fragments(state):
    if USE_STACK_FRAGMENT:
        state.reset_on()
