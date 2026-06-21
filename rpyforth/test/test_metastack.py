"""Unit tests for the stack-fragment metastacks (DSIntMetaStack et al.).

These exercise the top-cache + fragment-chain data structure directly, so they
run under plain CPython/PyPy without translating the interpreter or setting
RPYFORTH_STACK_FRAGMENT.
"""

from rpyforth.metastack import (
    DSIntMetaStack,
    DSFloatMetaStack,
    DSObjMetaStack,
    TOP_CACHE_SIZE,
    FRAGMENT_SIZE,
)
from rpyforth.inner_interp import InnerInterpreter


def test_inner_int_stack_deep_spill():
    inner = InnerInterpreter()
    n = TOP_CACHE_SIZE + 2 * FRAGMENT_SIZE + 11
    for v in range(n):
        inner.push_ds_int(v)
    assert inner.ds_int_size() == n
    assert inner.peek_ds_int(0) == n - 1
    assert inner.peek_ds_int(n - 1) == 0
    inner.poke_ds_int(n - 1, 987654)
    assert inner.peek_ds_int(n - 1) == 987654
    out = [inner.pop_ds_int() for _ in range(n)]
    assert out[0] == n - 1
    assert out[-1] == 987654
    assert inner.ds_int_size() == 0


def test_int_within_cache():
    s = DSIntMetaStack()
    s.push(1)
    s.push(2)
    s.push(3)
    s.push(4)
    assert s.size() == 4
    assert s.peek(0) == 4
    assert s.peek(3) == 1
    assert s.pop() == 4
    assert s.pop() == 3
    assert s.size() == 2


def test_int_spill_one_beyond_cache():
    # Pushing a 5th value must spill the bottom cache slot into the fragment
    # chain rather than asserting/overflowing.
    s = DSIntMetaStack()
    for v in range(1, 6):  # 1..5
        s.push(v)
    assert s.size() == 5
    assert s.peek(0) == 5
    assert s.peek(4) == 1
    # LIFO order out
    assert [s.pop() for _ in range(5)] == [5, 4, 3, 2, 1]
    assert s.size() == 0


def test_int_many_crossing_fragments():
    s = DSIntMetaStack()
    n = TOP_CACHE_SIZE + 2 * FRAGMENT_SIZE + 7  # force multiple fragments
    for v in range(n):
        s.push(v)
    assert s.size() == n
    # peek from top
    assert s.peek(0) == n - 1
    assert s.peek(n - 1) == 0
    # poke deep into the fragment chain, then read it back
    s.poke(n - 1, 12345)
    assert s.peek(n - 1) == 12345
    out = [s.pop() for _ in range(n)]
    assert out[0] == n - 1
    assert out[-1] == 12345
    assert s.size() == 0


def test_int_push_pop_interleaved():
    s = DSIntMetaStack()
    for v in range(20):
        s.push(v)
    for _ in range(15):
        s.pop()
    assert s.size() == 5
    assert s.peek(0) == 4
    for v in range(100, 110):
        s.push(v)
    assert s.size() == 15
    assert s.peek(0) == 109


def test_float_spill():
    s = DSFloatMetaStack()
    for v in range(10):
        s.push(float(v))
    assert s.size() == 10
    assert s.peek(0) == 9.0
    assert s.peek(9) == 0.0
    assert [s.pop() for _ in range(10)] == [float(v) for v in range(9, -1, -1)]


def test_obj_spill():
    s = DSObjMetaStack()
    objs = [object() for _ in range(10)]
    for o in objs:
        s.push(o)
    assert s.size() == 10
    assert s.peek(0) is objs[-1]
    assert s.peek(9) is objs[0]
    for o in reversed(objs):
        assert s.pop() is o
