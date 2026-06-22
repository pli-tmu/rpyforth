"""Unit tests for the integer stack-fragment metastack and its fragment.

The integer data stack keeps its top three elements in scalar virtualizable
fields (the stable DSIntFragment) and spills anything below them into a plain
overflow list. These tests exercise everything directly, so they run under plain
CPython/PyPy without translating the interpreter or setting RPYFORTH_STACK_FRAGMENT.

The float (DSFloatMetaStack/DSFloatFragment) and object
(DSObjMetaStack/DSObjFragment) stacks are left as a student exercise; add tests
for them here once their implementations exist.
"""

from rpyforth.metastack import (
    DSIntMetaStack,
    DSIntFragment,
    FRAGMENT_SIZE,
)
from rpyforth.inner_interp import InnerInterpreter


# ---------------------------------------------------------------------------
# DSIntMetaStack: three scalar tops + plain overflow list
# ---------------------------------------------------------------------------

def test_int_tops_and_overflow():
    s = DSIntMetaStack()
    for v in range(1, 6):           # 1..5: 3 in the scalar tops, 2 in overflow
        s.push(v)
    assert s.size() == 5
    assert s.active.top_count == 3
    assert len(s.overflow) == 2
    assert s.peek(0) == 5
    assert s.peek(4) == 1
    assert [s.pop() for _ in range(5)] == [5, 4, 3, 2, 1]
    assert s.size() == 0


def test_int_spill_to_overflow():
    s = DSIntMetaStack()
    n = FRAGMENT_SIZE + 5           # well past the 3 scalar tops
    for v in range(n):
        s.push(v)
    assert s.size() == n
    assert s.active.top_count == 3
    assert len(s.overflow) == n - 3            # everything below the 3 tops
    assert s.peek(0) == n - 1
    assert s.peek(n - 1) == 0                   # deepest, in overflow
    assert [s.pop() for _ in range(n)] == list(range(n - 1, -1, -1))
    assert s.size() == 0
    assert len(s.overflow) == 0                 # overflow drained on the way down


def test_int_many_fragments_peek_poke():
    s = DSIntMetaStack()
    n = 3 * FRAGMENT_SIZE + 7
    for v in range(n):
        s.push(v)
    assert s.size() == n
    assert s.peek(0) == n - 1
    assert s.peek(n - 1) == 0
    s.poke(n - 1, 424242)               # deepest element, in overflow
    assert s.peek(n - 1) == 424242
    s.poke(0, 99)                       # top, in the scalar tops
    assert s.peek(0) == 99
    out = [s.pop() for _ in range(n)]
    assert out[0] == 99
    assert out[-1] == 424242
    assert s.size() == 0


def test_int_clear_resets():
    s = DSIntMetaStack()
    for v in range(2 * FRAGMENT_SIZE):
        s.push(v)
    s.clear()
    assert s.size() == 0
    assert len(s.overflow) == 0
    assert s.active.top_count == 0
    s.push(42); s.push(43)
    assert s.size() == 2
    assert s.peek(0) == 43


def test_int_fragment_ops():
    f = DSIntFragment()
    assert f.top_count == 0 and not f.tops_full()
    f.push_top(10); f.push_top(20)
    assert f.top_count == 2
    assert f.peek_top(0) == 20 and f.peek_top(1) == 10
    f.poke_top(0, 99)
    assert f.peek_top(0) == 99
    assert f.pop_top() == 99 and f.pop_top() == 10
    assert f.top_count == 0
    # fill all three tops, then spill the bottom one and refill it
    f.push_top(1); f.push_top(2); f.push_top(3)
    assert f.tops_full()
    assert f.push_top_full(4) == 1      # 4 pushed on top, 1 spilled out
    assert f.peek_top(0) == 4 and f.peek_top(2) == 2
    assert f.pop_top_refill(1) == 4     # pop 4, refill third top with 1
    assert f.peek_top(2) == 1


# ---------------------------------------------------------------------------
# Cross-type parity helper (shared with the float/obj student exercise)
# ---------------------------------------------------------------------------

def _run_parity(cls, mk):
    s = cls()
    n = 600
    for v in range(n):
        s.push(mk(v))
    assert s.size() == n
    assert s.peek(0) == mk(n - 1)
    assert s.peek(n - 1) == mk(0)
    s.poke(n - 1, mk(123))
    s.poke(0, mk(456))
    assert s.peek(n - 1) == mk(123)
    assert s.peek(0) == mk(456)
    out = [s.pop() for _ in range(n)]
    assert out[0] == mk(456)
    assert out[-1] == mk(123)
    assert s.size() == 0


def test_parity_int():
    _run_parity(DSIntMetaStack, int)


# ---------------------------------------------------------------------------
# InnerInterpreter integration (the int metastack as wired into the VM)
# ---------------------------------------------------------------------------

def test_inner_int_stack_deep_spill():
    inner = InnerInterpreter()
    n = 2 * FRAGMENT_SIZE + 11
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
