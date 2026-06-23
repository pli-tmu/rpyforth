import os

# --- Stack-fragment layout (see docs/STACK_FRAGMENT_VIRTUALIZED_REDESIGN.md) ---
#
# The active fragment = the top of the data stack, held in host-resident,
# virtualizable fields:
#   * NTOP scalar "top" cells (t0, t1, t2)  -> the hottest cells, in registers
#   * a small virtualizable spill array frame[FRAME_SIZE] underneath them
# Together they hold the top ACTIVE_MAX cells of the running word. Anything
# deeper -- the caller frames parked on a call, plus the rare single-word
# overflow -- lives in the plain-heap metastack arena ("other places").
#
# Both NTOP and FRAME_SIZE must stay small: a virtualizable [*] array is tracked
# slot by slot by the trace optimizer, so a large one makes traces pathological
# (a 64-slot frame segfaulted the optimizer). 2 scalars + an 8-slot array is the
# sweet spot: the hot stack ops touch only the scalar tops with zero array
# access, while keeping push/pop shifts narrow -- a third scalar top measurably
# slowed push/pop-heavy code (DO-loop setup) for a gain only words reaching the
# 3rd cell (e.g. rot) would see.
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


# Virtualized: the active fragment (the NTOP scalar tops t0/t1/t2, the active
# depth d, the small spill array frame[*]) and the scalar metastack pointers
# (frag_ptr, spill_ptr). The metastack's backing arrays (frag_saved_n,
# frag_spill_base, spill) stay plain heap -- but their references are immutable
# (see InnerInterpreter._immutable_fields_) so each is loaded once at the loop
# header rather than reloaded per access.
#
# Why this split: a large virtualizable [*] array is pathological for the trace
# optimizer -- it is tracked slot by slot, so a 16384-slot virtualizable array
# segfaults the optimizer. The metastack must be sized to the recursion depth
# (large), so its arrays cannot be virtualized; but the scalar pointers into it
# are hot and cheap to virtualize. The hot stack ops touch only the scalar tops,
# so the frame array is barely accessed at all.
STACK_FRAGMENT_VIRTUALIZABLES = [
    "t0", "t1", "d", "frame[*]",
    "frag_ptr", "spill_ptr",
    "rs", "rs_ptr",
    "cs_threads", "cs_ips", "cs_ptr",
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
