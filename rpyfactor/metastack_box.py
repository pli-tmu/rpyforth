"""Host-resident stack-fragment cache for boxed W_Value cells.

Mirrors rpyforth/metastack_int.py, but cells are W_Value / None.
Frame slots are scalar fields f0..f7 (not a GC-pointer array).
No promote() here: promoting a virtualized box aborts the loop.
"""

from rpython.rlib.jit import unroll_safe
from rpython.rlib.debug import make_sure_not_resized

from rpyfactor.metastack import (
    NTOP,
    FRAME_SIZE,
    ACTIVE_MAX,
    SPILL_SIZE,
)
from rpyfactor.values import FactorError


class StackOverflow(FactorError):
    def __init__(self):
        FactorError.__init__(self, "stack overflow")


def init_fields(host):
    host.t0 = None
    host.t1 = None
    host.d = 0
    host.f0 = None
    host.f1 = None
    host.f2 = None
    host.f3 = None
    host.f4 = None
    host.f5 = None
    host.f6 = None
    host.f7 = None
    host.frag_ptr = 0
    host.spill = [None] * SPILL_SIZE
    make_sure_not_resized(host.spill)
    host.spill_ptr = 0


def _frame_get(host, si):
    if si == 0:
        return host.f0
    if si == 1:
        return host.f1
    if si == 2:
        return host.f2
    if si == 3:
        return host.f3
    if si == 4:
        return host.f4
    if si == 5:
        return host.f5
    if si == 6:
        return host.f6
    return host.f7


def _frame_set(host, si, v):
    if si == 0:
        host.f0 = v
    elif si == 1:
        host.f1 = v
    elif si == 2:
        host.f2 = v
    elif si == 3:
        host.f3 = v
    elif si == 4:
        host.f4 = v
    elif si == 5:
        host.f5 = v
    elif si == 6:
        host.f6 = v
    else:
        host.f7 = v


class JoyMetaStack(object):
    """Mixin: active-fragment + spill arena on the interpreter host."""

    def init_fields(self):
        init_fields(self)

    def push_on(self, v):
        dd = self.d
        if dd >= ACTIVE_MAX:
            self._spill_bottom()
            dd = self.d
        if dd >= NTOP:
            si = dd - NTOP
            assert si >= 0
            _frame_set(self, si, self.t1)
        self.t1 = self.t0
        self.t0 = v
        self.d = dd + 1

    def pop_on(self):
        dd = self.d
        if dd <= 0:
            return self._pop_from_arena()
        r = self.t0
        self.t0 = self.t1
        if dd > NTOP:
            si = dd - NTOP - 1
            assert si >= 0
            self.t1 = _frame_get(self, si)
        self.d = dd - 1
        return r

    def _pop_from_arena(self):
        ap = self.spill_ptr - 1
        if ap < 0:
            raise FactorError("stack underflow")
        assert ap >= 0
        r = self.spill[ap]
        self.spill_ptr = ap
        return r

    @unroll_safe
    def _spill_bottom(self):
        ap = self.spill_ptr
        if ap >= SPILL_SIZE:
            raise StackOverflow()
        assert ap >= 0
        self.spill[ap] = self.f0
        self.spill_ptr = ap + 1
        self.f0 = self.f1
        self.f1 = self.f2
        self.f2 = self.f3
        self.f3 = self.f4
        self.f4 = self.f5
        self.f5 = self.f6
        self.f6 = self.f7
        self.d = self.d - 1

    def peek_on(self, depth):
        dd = self.d
        if depth < dd:
            if depth == 0:
                return self.t0
            if depth == 1:
                return self.t1
            si = dd - 1 - depth
            assert si >= 0
            return _frame_get(self, si)
        ai = self.spill_ptr - 1 - (depth - dd)
        if ai < 0:
            raise FactorError("stack underflow")
        assert ai >= 0
        return self.spill[ai]

    def depth_on(self):
        return self.d + self.spill_ptr

    def reset_on(self):
        self.t0 = None
        self.t1 = None
        self.d = 0
        self.f0 = None
        self.f1 = None
        self.f2 = None
        self.f3 = None
        self.f4 = None
        self.f5 = None
        self.f6 = None
        self.f7 = None
        self.frag_ptr = 0
        self.spill_ptr = 0

    @unroll_safe
    def push_fragment_on(self):
        self.frag_ptr = self.frag_ptr + 1
        dd = self.d
        if dd > NTOP:
            n = dd - NTOP
            ap = self.spill_ptr
            if ap + n > SPILL_SIZE:
                raise StackOverflow()
            i = 0
            while i < n:
                self.spill[ap + i] = _frame_get(self, i)
                i += 1
            self.spill_ptr = ap + n
            self.d = NTOP

    def pop_fragment_commit_on(self):
        fp = self.frag_ptr - 1
        if fp < 0:
            raise FactorError("fragment underflow")
        self.frag_ptr = fp

    def push(self, v):
        self.push_on(v)

    def pop(self):
        return self.pop_on()

    def peek(self, depth=0):
        return self.peek_on(depth)

    def size(self):
        return self.depth_on()

    def push_fragment(self):
        self.push_fragment_on()

    def pop_fragment_commit(self):
        self.pop_fragment_commit_on()

    def snapshot_flat(self):
        n = self.size()
        out = []
        depth = n - 1
        while depth >= 0:
            out.append(self.peek(depth))
            depth -= 1
        return out

    def restore_flat(self, items):
        saved_frag = self.frag_ptr
        self.reset_on()
        self.frag_ptr = saved_frag
        i = 0
        while i < len(items):
            self.push(items[i])
            i += 1

    def replace_items(self, items):
        self.restore_flat(items)

    def reset(self):
        self.reset_on()

    @property
    def items(self):
        return self.snapshot_flat()
