"""Stack representations for rpyfactor.

FragmentBase is inherited by the Interpreter (rpyforth pattern): the cache
fields live on the interpreter itself so the virtualizable is the portal's
own red variable. Dual representation per cell:
  * unboxed ints in scalar/array fields (t0i, frame_i[*], ...)
  * objects in parallel fields (t0o, frame_o[*], ...)
so hot integer tops register-allocate the way Forth cells do, while
quotations and lists still flow through the same cache.
"""

from rpython.rlib.jit import unroll_safe
from rpython.rlib.debug import make_sure_not_resized

from rpyfactor.values import W_Int, FactorError
from rpyfactor.metastack import (
    USE_STACK_FRAGMENT,
    NTOP,
    FRAME_SIZE,
    ACTIVE_MAX,
    SPILL_SIZE,
    TAG_INT,
    TAG_OBJ,
)


def use_fragment_stack():
    return USE_STACK_FRAGMENT


class JoyStack(object):
    pass


class NaiveStack(JoyStack):
    def __init__(self):
        self.items = []

    def push(self, val):
        self.items.append(val)

    def push_int(self, n):
        self.items.append(W_Int(n))

    def pop(self):
        if not self.items:
            raise FactorError("stack underflow")
        return self.items.pop()

    def pop_int(self):
        v = self.pop()
        if not isinstance(v, W_Int):
            raise FactorError("expected integer")
        return v.val

    def peek(self, n=0):
        if n >= len(self.items):
            raise FactorError("stack underflow")
        return self.items[-1 - n]

    def peek_int(self, n=0):
        v = self.peek(n)
        if not isinstance(v, W_Int):
            raise FactorError("expected integer")
        return v.val

    def top_is_int(self):
        if not self.items:
            return False
        return isinstance(self.items[-1], W_Int)

    def pop_truthy(self):
        if self.top_is_int():
            return self.pop_int() != 0
        from rpyfactor.values import truthy
        return truthy(self.pop())

    def peek_parts(self, depth=0):
        v = self.peek(depth)
        if isinstance(v, W_Int):
            return v.val, TAG_INT, None
        return 0, TAG_OBJ, v

    def push_parts(self, i, t, o):
        if t == TAG_INT:
            self.push_int(i)
        else:
            self.push(o)

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


class StackOverflow(FactorError):
    def __init__(self):
        FactorError.__init__(self, "stack overflow")


class FragmentBase(JoyStack):
    """Fragment-cache fields and operations, mixed into the Interpreter.

    The interpreter class declares which of these fields are virtualizable;
    everything here only ever touches them through ``self`` so all access
    flows through the portal's red variable.
    """

    def init_fragment_fields(self):
        self.t0i = 0
        self.t1i = 0
        self.t0t = TAG_OBJ
        self.t1t = TAG_OBJ
        self.t0o = None
        self.t1o = None
        self.d = 0
        self.frame_i = [0] * FRAME_SIZE
        make_sure_not_resized(self.frame_i)
        self.frame_t = [TAG_OBJ] * FRAME_SIZE
        make_sure_not_resized(self.frame_t)
        self.frame_o = [None] * FRAME_SIZE
        make_sure_not_resized(self.frame_o)
        self.frag_ptr = 0
        self.spill_i = [0] * SPILL_SIZE
        make_sure_not_resized(self.spill_i)
        self.spill_t = [TAG_OBJ] * SPILL_SIZE
        make_sure_not_resized(self.spill_t)
        self.spill_o = [None] * SPILL_SIZE
        make_sure_not_resized(self.spill_o)
        self.spill_ptr = 0

    def _cell_to_value(self, i, t, o):
        if t == TAG_INT:
            return W_Int(i)
        return o

    def _value_parts(self, v):
        if isinstance(v, W_Int):
            return v.val, TAG_INT, None
        return 0, TAG_OBJ, v

    def push(self, val):
        i, t, o = self._value_parts(val)
        self._push_parts(i, t, o)

    def push_int(self, n):
        self._push_parts(n, TAG_INT, None)

    def _push_parts(self, i, t, o):
        dd = self.d
        if dd >= ACTIVE_MAX:
            self._spill_bottom()
            dd = self.d
        if dd >= NTOP:
            si = dd - NTOP
            assert si >= 0
            self.frame_i[si] = self.t1i
            self.frame_t[si] = self.t1t
            self.frame_o[si] = self.t1o
        self.t1i = self.t0i
        self.t1t = self.t0t
        self.t1o = self.t0o
        self.t0i = i
        self.t0t = t
        self.t0o = o
        self.d = dd + 1

    def pop(self):
        i, t, o = self._pop_parts()
        return self._cell_to_value(i, t, o)

    def pop_int(self):
        i, t, o = self._pop_parts()
        if t == TAG_INT:
            return i
        if isinstance(o, W_Int):
            return o.val
        raise FactorError("expected integer")

    def _pop_parts(self):
        dd = self.d
        if dd <= 0:
            return self._pop_from_arena()
        ri = self.t0i
        rt = self.t0t
        ro = self.t0o
        self.t0i = self.t1i
        self.t0t = self.t1t
        self.t0o = self.t1o
        if dd > NTOP:
            si = dd - NTOP - 1
            assert si >= 0
            self.t1i = self.frame_i[si]
            self.t1t = self.frame_t[si]
            self.t1o = self.frame_o[si]
        self.d = dd - 1
        return ri, rt, ro

    def _pop_from_arena(self):
        ap = self.spill_ptr - 1
        if ap < 0:
            raise FactorError("stack underflow")
        assert ap >= 0
        ri = self.spill_i[ap]
        rt = self.spill_t[ap]
        ro = self.spill_o[ap]
        self.spill_ptr = ap
        return ri, rt, ro

    @unroll_safe
    def _spill_bottom(self):
        ap = self.spill_ptr
        if ap >= SPILL_SIZE:
            raise StackOverflow()
        assert ap >= 0
        self.spill_i[ap] = self.frame_i[0]
        self.spill_t[ap] = self.frame_t[0]
        self.spill_o[ap] = self.frame_o[0]
        self.spill_ptr = ap + 1
        i = 0
        while i < FRAME_SIZE - 1:
            self.frame_i[i] = self.frame_i[i + 1]
            self.frame_t[i] = self.frame_t[i + 1]
            self.frame_o[i] = self.frame_o[i + 1]
            i += 1
        self.d = self.d - 1

    def peek(self, depth=0):
        i, t, o = self._peek_parts(depth)
        return self._cell_to_value(i, t, o)

    def peek_int(self, depth=0):
        i, t, o = self._peek_parts(depth)
        if t == TAG_INT:
            return i
        if isinstance(o, W_Int):
            return o.val
        raise FactorError("expected integer")

    def top_is_int(self):
        dd = self.d
        if dd <= 0:
            if self.spill_ptr <= 0:
                return False
            return self.spill_t[self.spill_ptr - 1] == TAG_INT
        return self.t0t == TAG_INT

    def pop_truthy(self):
        if self.top_is_int():
            return self.pop_int() != 0
        from rpyfactor.values import truthy
        return truthy(self.pop())

    def _peek_parts(self, depth):
        dd = self.d
        if depth < dd:
            if depth == 0:
                return self.t0i, self.t0t, self.t0o
            if depth == 1:
                return self.t1i, self.t1t, self.t1o
            si = dd - 1 - depth
            assert si >= 0
            return self.frame_i[si], self.frame_t[si], self.frame_o[si]
        ai = self.spill_ptr - 1 - (depth - dd)
        if ai < 0:
            raise FactorError("stack underflow")
        assert ai >= 0
        return self.spill_i[ai], self.spill_t[ai], self.spill_o[ai]

    def peek_parts(self, depth=0):
        return self._peek_parts(depth)

    def push_parts(self, i, t, o):
        self._push_parts(i, t, o)

    def size(self):
        return self.d + self.spill_ptr

    @unroll_safe
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
                self.spill_i[ap + i] = self.frame_i[i]
                self.spill_t[ap + i] = self.frame_t[i]
                self.spill_o[ap + i] = self.frame_o[i]
                i += 1
            self.spill_ptr = ap + n
            self.d = NTOP

    def pop_fragment_commit(self):
        fp = self.frag_ptr - 1
        if fp < 0:
            raise FactorError("fragment underflow")
        self.frag_ptr = fp

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
        self.reset()
        self.frag_ptr = saved_frag
        i = 0
        while i < len(items):
            self.push(items[i])
            i += 1

    def replace_items(self, items):
        self.restore_flat(items)

    def reset(self):
        self.t0i = 0
        self.t1i = 0
        self.t0t = TAG_OBJ
        self.t1t = TAG_OBJ
        self.t0o = None
        self.t1o = None
        self.d = 0
        self.frag_ptr = 0
        self.spill_ptr = 0
        i = 0
        while i < FRAME_SIZE:
            self.frame_i[i] = 0
            self.frame_t[i] = TAG_OBJ
            self.frame_o[i] = None
            i += 1


DataStack = NaiveStack
