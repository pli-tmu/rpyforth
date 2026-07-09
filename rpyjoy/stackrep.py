"""Stack representations for rpyjoy (naive list + optional fragment cache)."""

import os

from rpyjoy.values import JoyError


NTOP = 2
SPILL_SIZE = 16384


def _parse_frame_size():
    raw = os.environ.get("RPYJOY_FRAME_SIZE")
    if raw is None or raw == "":
        return 8
    n = 0
    ok = True
    i = 0
    while i < len(raw):
        ch = raw[i]
        if "0" <= ch <= "9":
            n = n * 10 + (ord(ch) - ord("0"))
        else:
            ok = False
            break
        i += 1
    if not ok or n < 1:
        return 8
    if n > 64:
        return 64
    return n


USE_STACK_FRAGMENT = bool(os.environ.get("RPYJOY_STACK_FRAGMENT"))


def use_fragment_stack():
    return USE_STACK_FRAGMENT


class JoyStack(object):
    pass


class StackOverflow(JoyError):
    def __init__(self):
        JoyError.__init__(self, "stack overflow")


class NaiveStack(JoyStack):
    def __init__(self):
        self.items = []

    def push(self, val):
        self.items.append(val)

    def pop(self):
        if not self.items:
            raise JoyError("stack underflow")
        return self.items.pop()

    def peek(self, n=0):
        if n >= len(self.items):
            raise JoyError("stack underflow")
        return self.items[-1 - n]

    def size(self):
        return len(self.items)

    def push_fragment(self):
        pass

    def pop_fragment_commit(self):
        pass

    def snapshot_flat(self):
        return list(self.items)

    def restore_flat(self, items):
        self.items = list(items)

    def replace_items(self, items):
        self.items = list(items)

    def reset(self):
        self.items = []


class FragmentStack(JoyStack):
    """Three-tier stack cache for boxed W_Value cells (P3)."""

    def __init__(self):
        self.frame_size = _parse_frame_size()
        self.active_max = NTOP + self.frame_size
        self.t0 = None
        self.t1 = None
        self.d = 0
        self.frame = [None] * self.frame_size
        self.frag_ptr = 0
        self.spill = [None] * SPILL_SIZE
        self.spill_ptr = 0

    def push(self, val):
        dd = self.d
        if dd >= self.active_max:
            self._spill_bottom()
            dd = self.d
        if dd >= NTOP:
            si = dd - NTOP
            self.frame[si] = self.t1
        self.t1 = self.t0
        self.t0 = val
        self.d = dd + 1

    def pop(self):
        dd = self.d
        if dd <= 0:
            return self._pop_from_arena()
        r = self.t0
        self.t0 = self.t1
        if dd > NTOP:
            si = dd - NTOP - 1
            self.t1 = self.frame[si]
        self.d = dd - 1
        return r

    def _pop_from_arena(self):
        ap = self.spill_ptr - 1
        if ap < 0:
            raise JoyError("stack underflow")
        r = self.spill[ap]
        self.spill_ptr = ap
        return r

    def _spill_bottom(self):
        ap = self.spill_ptr
        if ap >= SPILL_SIZE:
            raise StackOverflow()
        self.spill[ap] = self.frame[0]
        self.spill_ptr = ap + 1
        i = 0
        while i < self.frame_size - 1:
            self.frame[i] = self.frame[i + 1]
            i += 1
        self.d = self.d - 1

    def peek(self, depth):
        dd = self.d
        if depth < dd:
            if depth == 0:
                return self.t0
            if depth == 1:
                return self.t1
            si = dd - 1 - depth
            return self.frame[si]
        ai = self.spill_ptr - 1 - (depth - dd)
        if ai < 0:
            raise JoyError("stack underflow")
        return self.spill[ai]

    def size(self):
        return self.d + self.spill_ptr

    def push_fragment(self):
        self.frag_ptr = self.frag_ptr + 1
        dd = self.d
        if dd > NTOP:
            n = dd - NTOP
            ap = self.spill_ptr
            if ap + n > SPILL_SIZE:
                raise StackOverflow()
            i = 0
            while i < n:
                self.spill[ap + i] = self.frame[i]
                i += 1
            self.spill_ptr = ap + n
            self.d = NTOP

    def pop_fragment_commit(self):
        fp = self.frag_ptr - 1
        if fp < 0:
            raise JoyError("fragment underflow")
        self.frag_ptr = fp

    def snapshot_flat(self):
        n = self.size()
        out = []
        depth = n - 1
        while depth >= 0:
            out.append(self.peek(depth))
            depth -= 1
        return out

    @property
    def items(self):
        return self.snapshot_flat()

    def restore_flat(self, items):
        self.t0 = None
        self.t1 = None
        self.d = 0
        self.spill_ptr = 0
        i = 0
        while i < self.frame_size:
            self.frame[i] = None
            i += 1
        i = 0
        while i < len(items):
            self.push(items[i])
            i += 1

    def replace_items(self, items):
        self.restore_flat(items)

    def reset(self):
        self.t0 = None
        self.t1 = None
        self.d = 0
        self.frag_ptr = 0
        self.spill_ptr = 0
        i = 0
        while i < self.frame_size:
            self.frame[i] = None
            i += 1


def make_stack():
    if USE_STACK_FRAGMENT:
        return FragmentStack()
    return NaiveStack()


DataStack = NaiveStack
