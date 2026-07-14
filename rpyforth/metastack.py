"""Metastack constants and the three-tier data-stack layout.

Each data stack is cached in three tiers:
  * the top NTOP cells in scalar fields (t0, t1) -- in registers;
  * the next FRAME_SIZE cells in a small virtualizable frame array;
  * everything deeper in the shared spill area (the paper's "arena"), one
    plain heap array of SPILL_SIZE cells allocated once per VM and shared by
    every fragment.

A "fragment" is only the NTOP scalar tops plus the FRAME_SIZE frame plus two
pointers (frag_ptr, spill_ptr); a fragment owns no spill storage of its own. It
is a window [0, spill_ptr) onto the one shared spill area, so nest
(push_fragment_on) and unnest (pop_fragment_commit_on) allocate nothing.
"""

import os

NTOP = 2


def _parse_frame_size():
    raw = os.environ.get("RPYFORTH_FRAME_SIZE")
    if raw is None or raw == "":
        return 8
    n = 0
    ok = True
    for ch in raw:
        if "0" <= ch <= "9":
            n = n * 10 + (ord(ch) - ord("0"))
        else:
            ok = False
            break
    if not ok or n < 1:
        return 8
    if n > 64:
        return 64
    return n


FRAME_SIZE = _parse_frame_size()
ACTIVE_MAX = NTOP + FRAME_SIZE


def _parse_ntop_sweep():
    # Translation-time scalar-tops count for the parametric ablation variant
    # (metastack_int_ntop.py). Valid values are 4/8/16 (also 2, for validating
    # the parametric class against the NTOP=2 flagship). Anything else, empty,
    # or unset falls back to the flagship NTOP=2 so the constant-folded chains
    # stay well-formed even when the variant is not selected.
    raw = os.environ.get("RPYFORTH_NTOP")
    if raw is None or raw == "":
        return NTOP
    n = 0
    ok = True
    for ch in raw:
        if "0" <= ch <= "9":
            n = n * 10 + (ord(ch) - ord("0"))
        else:
            ok = False
            break
    if not ok:
        return NTOP
    if n == 2 or n == 4 or n == 8 or n == 16:
        return n
    return NTOP


SWEEP_NTOP = _parse_ntop_sweep()

# Import window on a non-tail call: the callee inherits the caller's top NTOP
# cells (already in the scalar tops -- free), and the caller's deeper frame
# cells are parked in the metastack and restored on return. Kept as CALL_WINDOW
# for import compatibility with tests/stubs.
CALL_WINDOW = NTOP

# Metastack capacity: one parked record per nested word call (frag_ptr indexes
# it, mirroring the call stack), and a flat spill bounding total parked cells.
FRAG_DEPTH = 16384
SPILL_SIZE = 16384

# Legacy constants kept for import compatibility with the float/object stubs.
FRAGMENT_SIZE = 256
TOP_CACHE_SIZE = 4
STACK_SIZE = FRAGMENT_SIZE

USE_STACK_FRAGMENT = bool(os.environ.get("RPYFORTH_STACK_FRAGMENT"))
STACK_FRAGMENT_STRICT = bool(os.environ.get("RPYFORTH_STACK_FRAGMENT_STRICT"))


def _float_fragment_enabled():
    # Opt-in ablation variant (RPYFORTH_FLOAT_FRAGMENT=1): mirroring the int
    # metastack for the float stack was measured as a net loss -- the second
    # virtualizable array (fframe[*]) taxes every trace boundary, regressing
    # even float-free kernels (nestedloop -31%) and the float-hot ones it was
    # meant to help (heap -33%). Kept behind this flag as a negative result.
    if not USE_STACK_FRAGMENT:
        return False
    return os.environ.get("RPYFORTH_FLOAT_FRAGMENT") == "1"


USE_FLOAT_FRAGMENT = _float_fragment_enabled()


def _frame_only_enabled():
    # Opt-in ablation variant (RPYFORTH_FRAME_ONLY=1, NTOP=0): every cached cell
    # lives in the virtualizable frame[*] array, no scalar tops, so push/pop
    # move no data. Only meaningful under the stack fragment. Mutually exclusive
    # with the float fragment: if both are set frame-only wins and the float
    # fragment stays off.
    if not USE_STACK_FRAGMENT:
        return False
    return os.environ.get("RPYFORTH_FRAME_ONLY") == "1"


USE_FRAME_ONLY = _frame_only_enabled()


def _ntop_sweep_enabled():
    # Opt-in ablation variant (RPYFORTH_NTOP set and non-empty): selects the
    # parametric-NTOP int metastack (DSIntMetaStackN) with SWEEP_NTOP scalar
    # tops. Only meaningful under the stack fragment. Precedence:
    # FRAME_ONLY > NTOP sweep > float fragment.
    if not USE_STACK_FRAGMENT:
        return False
    raw = os.environ.get("RPYFORTH_NTOP")
    if raw is None or raw == "":
        return False
    if USE_FRAME_ONLY:
        return False
    return True


USE_NTOP_SWEEP = _ntop_sweep_enabled()

if USE_FRAME_ONLY:
    USE_FLOAT_FRAGMENT = False
    USE_NTOP_SWEEP = False

if USE_NTOP_SWEEP:
    USE_FLOAT_FRAGMENT = False


class DataStackOverflow(Exception):
    pass


class DSFragment(object):
    pass


class DSMetaStack(object):
    pass

# The frame-only ablation (USE_FRAME_ONLY, NTOP=0) drops the scalar tops t0, t1
# and virtualizes only the frame[*] array; every other cached field is shared.
if USE_FRAME_ONLY:
    _INT_CACHE_VIRTUALIZABLES = ["d", "frame[*]", "frag_ptr", "spill_ptr"]
elif USE_NTOP_SWEEP:
    # The parametric variant virtualizes the first SWEEP_NTOP scalar tops (the
    # rest are dead after constant folding) plus the shared cache fields.
    _INT_CACHE_VIRTUALIZABLES = (["t%d" % i for i in range(SWEEP_NTOP)] +
                                 ["d", "frame[*]", "frag_ptr", "spill_ptr"])
else:
    _INT_CACHE_VIRTUALIZABLES = ["t0", "t1", "d", "frame[*]",
                                 "frag_ptr", "spill_ptr"]

_STACK_FRAGMENT_VIRTUALIZABLES_BASE = _INT_CACHE_VIRTUALIZABLES + [
    "rs_ptr",
    "cs_pcs", "cs_ptr", "cs_base",
    "li",
    "cell_size", "cell_size_bytes", "base",
    "ds_ptr_locals",
]

# When the float fragment is on, the top float cells are virtualized in scalar
# fields + the fixed-size fframe (never the 16384-cell fspill). When it is
# off, the plain float array reference + pointer are virtualized instead.
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
