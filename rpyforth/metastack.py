import os

# Fixed capacity of the data stack's overflow (depths 3 and below). The fragment
# size is stable: if a program pushes past it, spill_top raises DataStackOverflow
# instead of growing or chaining.
STACK_SIZE = 16384

FRAGMENT_SIZE = 256

TOP_CACHE_SIZE = 4

USE_STACK_FRAGMENT = bool(os.environ.get("RPYFORTH_STACK_FRAGMENT"))


class DataStackOverflow(Exception):
    """Raised when the data stack grows past the fragment's fixed capacity."""
    pass


class DSMetaStack(object):
    pass


class DSFragment(object):
    pass


class DSIntMetaStack(DSMetaStack):
    # Thin wrapper around a DSIntFragment, kept for the unit tests and as the
    # template for the float/object student exercise. The interpreter itself holds
    # a DSIntFragment directly (self.ds_int_frag) so reaching the virtualizable is
    # a single field load instead of self.ds_int_meta.active -- the data-stack
    # logic lives on the fragment now (see DSIntFragment.push/pop/peek/poke).
    _immutable_fields_ = ["active"]

    def __init__(self):
        self.active = DSIntFragment()

    def head(self):
        return self.active

    def push(self, v):
        self.active.push(v)

    def pop(self):
        return self.active.pop()

    def peek(self, depth):
        return self.active.peek(depth)

    def poke(self, depth, v):
        self.active.poke(depth, v)

    def size(self):
        return self.active.size()

    def clear(self):
        self.active.clear()


class DSFloatMetaStack(DSMetaStack):
    pass


class DSObjMetaStack(DSMetaStack):
    pass


class DSIntFragment(DSFragment):
    # The single, stable JIT virtualizable for the integer data stack. It owns:
    #   * top0/top1/top2 + top_count: the 3 shallowest slots, kept in registers so
    #     Forth's shallow ops (dup/over/rot/swap = depth 0..2) hit a constant-shape
    #     hot path -- this is what makes deep recursion (ack) fast.
    #   * overflow + overflow_ptr: a FIXED-size spill area for depths 3 and below.
    # overflow_ptr is virtual (a register), but `overflow` is an ORDINARY array
    # (NOT in _virtualizable_) so overflow[overflow_ptr] is a plain heap op; a
    # virtualizable array at a variable index forces/aborts tracing.
    _virtualizable_ = ["top0", "top1", "top2", "top_count", "overflow_ptr"]
    _immutable_fields_ = ["overflow"]

    def __init__(self):
        self.top0 = 0
        self.top1 = 0
        self.top2 = 0
        self.top_count = 0
        self.overflow = [0] * STACK_SIZE
        self.overflow_ptr = 0

    # --- data-stack operations (the interpreter calls these on an access_directly
    #     view of this fragment, so the scalar tops and overflow_ptr stay in
    #     registers; untranslated they are plain methods used by the tests) ---

    def push(self, v):
        if self.top_count < 3:
            self.push_top(v)                       # hot: constant scalar slot
        else:
            self.spill_top(self.push_top_full(v))  # deeper: fixed overflow array

    def pop(self):
        if self.top_count == 3 and self.has_overflow():
            return self.pop_top_refill(self.pop_overflow())  # deeper: from overflow
        return self.pop_top()                      # hot: constant scalar slot

    def peek(self, depth):
        if depth < 3:
            return self.peek_top(depth)            # hot: constant scalar slot
        return self.peek_overflow(depth - 3)       # deeper: read from overflow

    def poke(self, depth, v):
        if depth < 3:
            self.poke_top(depth, v)                # hot: constant scalar slot
        else:
            self.poke_overflow(depth - 3, v)       # deeper: write to overflow

    def size(self):
        return self.top_count + self.overflow_ptr

    # --- 3 scalar tops (depth 0..2) ---

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
        # Precondition: top_count == 3; `refill` comes from the overflow array.
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

    # --- overflow array (depth 3 and below; fixed size, plain array, virtual ptr) ---

    def has_overflow(self):
        return self.overflow_ptr > 0

    def spill_top(self, v):
        op = self.overflow_ptr
        assert op >= 0
        if op >= STACK_SIZE:
            raise DataStackOverflow()
        self.overflow[op] = v
        self.overflow_ptr = op + 1

    def pop_overflow(self):
        op = self.overflow_ptr - 1
        assert op >= 0                 # data stack underflow
        self.overflow_ptr = op
        return self.overflow[op]

    def peek_overflow(self, odepth):
        # odepth = depth - 3.
        idx = self.overflow_ptr - 1 - odepth
        assert idx >= 0                # data stack underflow
        return self.overflow[idx]

    def poke_overflow(self, odepth, v):
        idx = self.overflow_ptr - 1 - odepth
        assert idx >= 0                # data stack underflow
        self.overflow[idx] = v

    def clear(self):
        self.top0 = 0
        self.top1 = 0
        self.top2 = 0
        self.top_count = 0
        self.overflow_ptr = 0


class DSFloatFragment(DSFragment):
    pass


class DSObjFragment(DSFragment):
    pass
