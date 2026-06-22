import os

from rpython.rlib.jit import hint

STACK_SIZE = 256

FRAGMENT_SIZE = 256

TOP_CACHE_SIZE = 4

USE_STACK_FRAGMENT = bool(os.environ.get("RPYFORTH_STACK_FRAGMENT"))


class DSMetaStack(object):
    pass


class DSFragment(object):
    pass


class DSIntMetaStack(DSMetaStack):
    # The single `active` fragment is the JIT's virtualizable and its identity
    # NEVER changes (the JIT requires a stable virtualizable). The fragment holds
    # the 3 scalar tops; everything deeper lives in the plain `overflow` list
    # (overflow[0] at the bottom of the stack). So bottom->top is overflow + tops.
    def __init__(self):
        self.active = DSIntFragment()
        self.overflow = []

    # --- hot path: the stable active fragment (loop-free) ---

    def head(self):
        return self.active

    # --- cold path: the plain overflow list (no virtualizable access) ---

    def spill_top(self, v):
        self.overflow.append(v)

    def has_overflow(self):
        return len(self.overflow) > 0

    def pop_overflow(self):
        n = len(self.overflow)
        assert n > 0    # data stack underflow
        return self.overflow.pop()

    def peek_overflow(self, odepth):
        # odepth = depth - 3. Reads the plain overflow list only.
        idx = len(self.overflow) - 1 - odepth
        assert idx >= 0    # data stack underflow
        return self.overflow[idx]

    def poke_overflow(self, odepth, v):
        idx = len(self.overflow) - 1 - odepth
        assert idx >= 0    # data stack underflow
        self.overflow[idx] = v

    # --- data-stack operations (used by the interpreter and the unit tests) ---
    # Each hot path goes through an access_directly view of the stable active
    # fragment so the 3 scalar tops stay in registers under the JIT. The hint is a
    # no-op untranslated, so these double as the plain implementation for tests.

    def push(self, v):
        a = hint(self.active, access_directly=True)
        if a.top_count < 3:
            a.push_top(v)                          # hot: constant scalar slot
        else:
            self.spill_top(a.push_top_full(v))     # deeper: plain overflow list

    def pop(self):
        a = hint(self.active, access_directly=True)
        if a.top_count == 3 and self.has_overflow():
            return a.pop_top_refill(self.pop_overflow())  # deeper: refill from overflow
        return a.pop_top()                         # hot: constant scalar slot

    def peek(self, depth):
        a = hint(self.active, access_directly=True)
        if depth < 3:
            return a.peek_top(depth)               # hot: constant scalar slot
        return self.peek_overflow(depth - 3)       # deeper: read from overflow

    def poke(self, depth, v):
        a = hint(self.active, access_directly=True)
        if depth < 3:
            a.poke_top(depth, v)                   # hot: constant scalar slot
        else:
            self.poke_overflow(depth - 3, v)       # deeper: write to overflow

    def size(self):
        return len(self.overflow) + self.active.top_count

    def clear(self):
        a = self.active
        a.top0 = 0
        a.top1 = 0
        a.top2 = 0
        a.top_count = 0
        self.overflow = []


class DSFloatMetaStack(DSMetaStack):
    pass


class DSObjMetaStack(DSMetaStack):
    pass


class DSIntFragment(DSFragment):
    _virtualizable_ = ["top0", "top1", "top2", "top_count"]

    def __init__(self):
        self.top0 = 0
        self.top1 = 0
        self.top2 = 0
        self.top_count = 0

    def tops_full(self):
        return self.top_count == 3

    def push_top(self, v):
        # Precondition: top_count < 3.
        self.top2 = self.top1
        self.top1 = self.top0
        self.top0 = v
        self.top_count = self.top_count + 1

    def push_top_full(self, v):
        # Precondition: top_count == 3. Returns the spilled bottom top.
        spilled = self.top2
        self.top2 = self.top1
        self.top1 = self.top0
        self.top0 = v
        return spilled

    def pop_top(self):
        # Precondition: nothing below to refill the third top with.
        v = self.top0
        self.top0 = self.top1
        self.top1 = self.top2
        self.top2 = 0
        self.top_count = self.top_count - 1
        return v

    def pop_top_refill(self, refill):
        # Precondition: top_count == 3; `refill` comes from the overflow list.
        v = self.top0
        self.top0 = self.top1
        self.top1 = self.top2
        self.top2 = refill
        return v

    def peek_top(self, depth):
        if depth == 0:
            return self.top0
        elif depth == 1:
            return self.top1
        else:
            return self.top2

    def poke_top(self, depth, v):
        if depth == 0:
            self.top0 = v
        elif depth == 1:
            self.top1 = v
        else:
            self.top2 = v


class DSFloatFragment(DSFragment):
    pass


class DSObjFragment(DSFragment):
    pass
