import os

STACK_SIZE = 256

FRAGMENT_SIZE = 256

TOP_CACHE_SIZE = 4

# Stack-fragment mode keeps the hot data stack in a 4-deep scalar top cache and
# spills the overflow into a chained list of fixed-size fragments. The top cache
# of the *integer* data stack is the jitdriver's virtualizable, so those scalar
# fields must be register-resident; everything else stays in the heap.
USE_STACK_FRAGMENT = bool(os.environ.get("RPYFORTH_STACK_FRAGMENT"))


class DSMetaStack(object):
    pass


class DSFragment(object):
    pass


class DSIntMetaStack(DSMetaStack):
    _virtualizable_ = ["top0", "top1", "top2", "top3", "top_count"]

    def __init__(self):
        self.current = DSIntFragment(None)
        self.top0 = 0
        self.top1 = 0
        self.top2 = 0
        self.top3 = 0
        self.top_count = 0

    def push_cache(self, intval):
        self.top3 = self.top2
        self.top2 = self.top1
        self.top1 = self.top0
        self.top0 = intval
        self.top_count = self.top_count + 1

    def push_cache_full(self, intval):
        spilled = self.top3
        self.top3 = self.top2
        self.top2 = self.top1
        self.top1 = self.top0
        self.top0 = intval
        return spilled

    def pop_cache(self):
        result = self.top0
        self.top0 = self.top1
        self.top1 = self.top2
        self.top2 = self.top3
        self.top3 = 0
        self.top_count = self.top_count - 1
        return result

    def pop_cache_refill(self, refill):
        result = self.top0
        self.top0 = self.top1
        self.top1 = self.top2
        self.top2 = self.top3
        self.top3 = refill
        return result

    def peek_top(self, depth):
        assert depth < self.top_count
        if depth == 0:
            return self.top0
        elif depth == 1:
            return self.top1
        elif depth == 2:
            return self.top2
        else:
            return self.top3

    def poke_top(self, depth, intval):
        assert depth < self.top_count
        if depth == 0:
            self.top0 = intval
        elif depth == 1:
            self.top1 = intval
        elif depth == 2:
            self.top2 = intval
        else:
            self.top3 = intval

    def cache_count(self):
        return self.top_count

    def clear_cache(self):
        self.top0 = 0
        self.top1 = 0
        self.top2 = 0
        self.top3 = 0
        self.top_count = 0

    def has_frag(self):
        return self.current.sp > 0

    def spill(self, intval):
        cur = self.current
        if cur.sp + 1 >= FRAGMENT_SIZE:
            cur = DSIntFragment(cur)
            self.current = cur
        cur.push(intval)

    def refill(self):
        cur = self.current
        v = cur.pop()
        if cur.sp == 0 and cur.parent is not None:
            self.current = cur.parent
        return v

    def peek_deep(self, depth):
        frag_pos = depth - TOP_CACHE_SIZE
        cur = self.current
        while cur is not None:
            if frag_pos < cur.sp:
                return cur.cells[cur.sp - frag_pos]
            frag_pos -= cur.sp
            cur = cur.parent
        assert False  # data stack underflow
        return 0  # unreachable; satisfies RPython return-type inference

    def poke_deep(self, depth, intval):
        frag_pos = depth - TOP_CACHE_SIZE
        cur = self.current
        while cur is not None:
            if frag_pos < cur.sp:
                cur.cells[cur.sp - frag_pos] = intval
                return
            frag_pos -= cur.sp
            cur = cur.parent
        assert False  # data stack underflow

    def push(self, intval):
        if self.top_count < TOP_CACHE_SIZE:
            self.push_cache(intval)
        else:
            self.spill(self.push_cache_full(intval))

    def pop(self):
        if self.has_frag():
            return self.pop_cache_refill(self.refill())
        return self.pop_cache()

    def peek(self, depth):
        if depth < TOP_CACHE_SIZE:
            return self.peek_top(depth)
        return self.peek_deep(depth)

    def poke(self, depth, intval):
        if depth < TOP_CACHE_SIZE:
            self.poke_top(depth, intval)
        else:
            self.poke_deep(depth, intval)

    def size(self):
        n = self.top_count
        cur = self.current
        while cur is not None:
            n += cur.sp
            cur = cur.parent
        return n

    def clear(self):
        self.clear_cache()
        cur = self.current
        while cur.parent is not None:
            cur = cur.parent
        cur.sp = 0
        self.current = cur


class DSFloatMetaStack(DSMetaStack):
    def __init__(self):
        self.current = DSFloatFragment(None)
        self.top0 = 0.0
        self.top1 = 0.0
        self.top2 = 0.0
        self.top3 = 0.0
        self.top_count = 0

    def push(self, val):
        tc = self.top_count
        if tc == TOP_CACHE_SIZE:
            cur = self.current
            if cur.sp + 1 >= FRAGMENT_SIZE:
                cur = DSFloatFragment(cur)
                self.current = cur
            cur.push(self.top3)
        else:
            self.top_count = tc + 1
        self.top3 = self.top2
        self.top2 = self.top1
        self.top1 = self.top0
        self.top0 = val

    def pop(self):
        result = self.top0
        cur = self.current
        if cur.sp > 0:
            self.top0 = self.top1
            self.top1 = self.top2
            self.top2 = self.top3
            self.top3 = cur.pop()
            if cur.sp == 0 and cur.parent is not None:
                self.current = cur.parent
        else:
            self.top0 = self.top1
            self.top1 = self.top2
            self.top2 = self.top3
            self.top3 = 0.0
            self.top_count = self.top_count - 1
        return result

    def peek(self, depth):
        if depth < TOP_CACHE_SIZE:
            assert depth < self.top_count
            if depth == 0:
                return self.top0
            elif depth == 1:
                return self.top1
            elif depth == 2:
                return self.top2
            else:
                return self.top3
        frag_pos = depth - TOP_CACHE_SIZE
        cur = self.current
        while cur is not None:
            if frag_pos < cur.sp:
                return cur.cells[cur.sp - frag_pos]
            frag_pos -= cur.sp
            cur = cur.parent
        assert False
        return 0.0  # unreachable; satisfies RPython return-type inference

    def size(self):
        n = self.top_count
        cur = self.current
        while cur is not None:
            n += cur.sp
            cur = cur.parent
        return n

    def clear(self):
        self.top0 = 0.0
        self.top1 = 0.0
        self.top2 = 0.0
        self.top3 = 0.0
        self.top_count = 0
        cur = self.current
        while cur.parent is not None:
            cur = cur.parent
        cur.sp = 0
        self.current = cur


class DSObjMetaStack(DSMetaStack):
    # Plain heap object holding boxed W_Object references.
    def __init__(self):
        self.current = DSObjFragment(None)
        self.top0 = None
        self.top1 = None
        self.top2 = None
        self.top3 = None
        self.top_count = 0

    def push(self, val):
        tc = self.top_count
        if tc == TOP_CACHE_SIZE:
            # Cache full: spill the bottom slot into the fragment chain.
            cur = self.current
            if cur.sp + 1 >= FRAGMENT_SIZE:
                cur = DSObjFragment(cur)
                self.current = cur
            cur.push(self.top3)
        else:
            self.top_count = tc + 1
        self.top3 = self.top2
        self.top2 = self.top1
        self.top1 = self.top0
        self.top0 = val

    def pop(self):
        result = self.top0
        cur = self.current
        if cur.sp > 0:
            self.top0 = self.top1
            self.top1 = self.top2
            self.top2 = self.top3
            self.top3 = cur.pop()
            if cur.sp == 0 and cur.parent is not None:
                self.current = cur.parent
        else:
            self.top0 = self.top1
            self.top1 = self.top2
            self.top2 = self.top3
            self.top3 = None
            self.top_count = self.top_count - 1
        return result

    def peek(self, depth):
        if depth < TOP_CACHE_SIZE:
            assert depth < self.top_count
            if depth == 0:
                return self.top0
            elif depth == 1:
                return self.top1
            elif depth == 2:
                return self.top2
            else:
                return self.top3
        frag_pos = depth - TOP_CACHE_SIZE
        cur = self.current
        while cur is not None:
            if frag_pos < cur.sp:
                return cur.cells[cur.sp - frag_pos]
            frag_pos -= cur.sp
            cur = cur.parent
        assert False
        return None  # unreachable; satisfies RPython return-type inference

    def size(self):
        n = self.top_count
        cur = self.current
        while cur is not None:
            n += cur.sp
            cur = cur.parent
        return n

    def clear(self):
        self.top0 = None
        self.top1 = None
        self.top2 = None
        self.top3 = None
        self.top_count = 0
        cur = self.current
        while cur.parent is not None:
            cur = cur.parent
        cur.sp = 0
        self.current = cur


class DSIntFragment(DSFragment):
    _immutable_fields_ = ["parent", "cells"]
    _virtualizable_ = ["cells[*]", "sp"]

    def __init__(self, parent):
        self.parent = parent
        self.cells = [0] * FRAGMENT_SIZE
        self.sp = 0

    def push(self, v):
        sp = self.sp + 1
        assert 0 <= sp < FRAGMENT_SIZE
        self.cells[sp] = v
        self.sp = sp

    def pop(self):
        sp = self.sp
        assert 0 <= sp < FRAGMENT_SIZE
        v = self.cells[sp]
        self.sp = sp - 1
        return v


class DSFloatFragment(DSFragment):
    _immutable_fields_ = ["parent", "cells"]

    def __init__(self, parent):
        self.parent = parent
        self.cells = [0.0] * FRAGMENT_SIZE
        self.sp = 0

    def push(self, v):
        sp = self.sp + 1
        assert 0 <= sp < FRAGMENT_SIZE
        self.cells[sp] = v
        self.sp = sp

    def pop(self):
        sp = self.sp
        assert 0 <= sp < FRAGMENT_SIZE
        v = self.cells[sp]
        self.sp = sp - 1
        return v


class DSObjFragment(DSFragment):
    _immutable_fields_ = ["parent", "cells"]

    def __init__(self, parent):
        self.parent = parent
        self.cells = [None] * FRAGMENT_SIZE
        self.sp = 0

    def push(self, v):
        sp = self.sp + 1
        assert 0 <= sp < FRAGMENT_SIZE
        self.cells[sp] = v
        self.sp = sp

    def pop(self):
        sp = self.sp
        assert 0 <= sp < FRAGMENT_SIZE
        v = self.cells[sp]
        self.sp = sp - 1
        return v
