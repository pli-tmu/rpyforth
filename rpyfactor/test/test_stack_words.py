from rpyfactor.interp import Interpreter
from rpyfactor.primitives import (
    prim_pick, prim_3dup, prim_rolldown, prim_over, prim_dup, prim_swap,
)
from rpyfactor.metastack import (
    _parse_frame_size, FRAME_SIZE_DEFAULT,
)


def _stack():
    return Interpreter().st()


def _flat(st):
    # snapshot_flat returns bottom-first; extract unboxed ints.
    return [v.val for v in st.snapshot_flat()]


def _load(st, vals):
    for v in vals:
        st.push_int(v)


def test_pick_copies_third_from_top():
    st = _stack()
    _load(st, [1, 2, 3])
    prim_pick(st)
    assert _flat(st) == [1, 2, 3, 1]


def test_pick_equiv_manual():
    # pick == over over over-style? verify against explicit dup/dip identity:
    # x y z pick == x y z [ [ dup ] dip swap ] dip swap gives x y z x
    st = _stack()
    _load(st, [7, 8, 9])
    prim_pick(st)
    manual = _stack()
    _load(manual, [7, 8, 9])
    # rebuild x y z x by hand: rot dup -> ... simplest: peek third and push.
    i, t, o = manual.peek_parts(2)
    manual.push_parts(i, t, o)
    assert _flat(st) == _flat(manual)


def test_3dup():
    st = _stack()
    _load(st, [1, 2, 3])
    prim_3dup(st)
    assert _flat(st) == [1, 2, 3, 1, 2, 3]


def test_3dup_equiv_dup_over():
    # 3dup == dup over over ? no. verify against explicit triple copy.
    st = _stack()
    _load(st, [4, 5, 6])
    prim_3dup(st)
    assert _flat(st) == [4, 5, 6, 4, 5, 6]


def test_neg_rot_factor_semantics():
    # Factor -rot: ( x y z -- z x y )
    st = _stack()
    _load(st, [1, 2, 3])
    prim_rolldown(st)
    assert _flat(st) == [3, 1, 2]


def test_pick_deep_stack():
    # Exercise the frame/spill path with a deep stack.
    st = _stack()
    _load(st, range(1, 21))
    prim_pick(st)
    assert _flat(st)[-1] == 18
    assert len(_flat(st)) == 21


def test_parse_frame_size_default():
    assert _parse_frame_size(None) == FRAME_SIZE_DEFAULT
    assert _parse_frame_size("") == FRAME_SIZE_DEFAULT


def test_parse_frame_size_valid_powers_of_two():
    assert _parse_frame_size("4") == 4
    assert _parse_frame_size("8") == 8
    assert _parse_frame_size("16") == 16
    assert _parse_frame_size("64") == 64


def test_parse_frame_size_rejects_non_power_of_two():
    assert _parse_frame_size("6") == FRAME_SIZE_DEFAULT
    assert _parse_frame_size("12") == FRAME_SIZE_DEFAULT


def test_parse_frame_size_rejects_out_of_range():
    assert _parse_frame_size("2") == FRAME_SIZE_DEFAULT
    assert _parse_frame_size("128") == FRAME_SIZE_DEFAULT


def test_parse_frame_size_rejects_garbage():
    assert _parse_frame_size("abc") == FRAME_SIZE_DEFAULT
