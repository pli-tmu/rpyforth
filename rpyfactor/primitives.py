"""Built-in Factor-subset words for rpyfactor (P1 naive interpreter)."""

import time

from rpython.rlib.rfile import create_stdio

from rpyfactor.values import (
    W_Int, W_String, W_Symbol, W_List, W_Cons, W_Nil, W_Array, W_Quotation,
    FactorError, truthy,
    nil_list, w_list_from_items, list_is_empty, list_length,
)
from rpyfactor.program import item_to_value, LitQuot, CallWord


def _pop_int(st):
    return st.pop_int()


def _push_int(st, n):
    st.push_int(n)


def _pop_list(st):
    v = st.pop()
    if not isinstance(v, W_List):
        raise FactorError("expected list")
    return v


def _pop_quot(st):
    v = st.pop()
    if not isinstance(v, W_Quotation):
        raise FactorError("expected quotation")
    return v


def _pop_string(st):
    v = st.pop()
    if not isinstance(v, W_String):
        raise FactorError("expected string")
    return v.s


def _pop_symbol(st):
    v = st.pop()
    if isinstance(v, W_Symbol):
        return v
    if isinstance(v, W_String):
        return W_Symbol.intern(v.s)
    raise FactorError("expected symbol")


def _push_bool(st, b):
    # Store flags as unboxed 0/1 so int-heavy traces stay in the
    # virtualizable int path (Forth-style) instead of boxing W_Bool.
    st.push_int(1 if b else 0)


def _values_equal(a, b):
    if isinstance(a, W_Int) and isinstance(b, W_Int):
        return a.val == b.val
    if isinstance(a, W_String) and isinstance(b, W_String):
        return a.s == b.s
    if isinstance(a, W_Symbol) and isinstance(b, W_Symbol):
        return a.name == b.name
    if isinstance(a, W_List) and isinstance(b, W_List):
        while isinstance(a, W_Cons) and isinstance(b, W_Cons):
            if not _values_equal(a.head, b.head):
                return False
            a = a.tail
            b = b.tail
        return list_is_empty(a) and list_is_empty(b)
    if isinstance(a, W_Quotation) and isinstance(b, W_Quotation):
        return a.program == b.program
    return False


def prim_add(st):
    b = _pop_int(st)
    a = _pop_int(st)
    _push_int(st, a + b)


def prim_sub(st):
    b = _pop_int(st)
    a = _pop_int(st)
    _push_int(st, a - b)


def prim_mul(st):
    b = _pop_int(st)
    a = _pop_int(st)
    _push_int(st, a * b)


def prim_div(st):
    b = _pop_int(st)
    a = _pop_int(st)
    if b == 0:
        raise FactorError("division by zero")
    _push_int(st, a // b)


def prim_rem(st):
    b = _pop_int(st)
    a = _pop_int(st)
    if b == 0:
        raise FactorError("division by zero")
    _push_int(st, a % b)


def prim_lt(st):
    b = _pop_int(st)
    a = _pop_int(st)
    _push_bool(st, a < b)


def prim_le(st):
    b = _pop_int(st)
    a = _pop_int(st)
    _push_bool(st, a <= b)


def prim_gt(st):
    b = _pop_int(st)
    a = _pop_int(st)
    _push_bool(st, a > b)


def prim_ge(st):
    b = _pop_int(st)
    a = _pop_int(st)
    _push_bool(st, a >= b)


def prim_eq(st):
    if st.top_is_int():
        b = st.pop_int()
        if st.top_is_int():
            a = st.pop_int()
            _push_bool(st, a == b)
            return
        st.push_int(b)
    b = st.pop()
    a = st.pop()
    _push_bool(st, _values_equal(a, b))


def prim_ne(st):
    if st.top_is_int():
        b = st.pop_int()
        if st.top_is_int():
            a = st.pop_int()
            _push_bool(st, a != b)
            return
        st.push_int(b)
    b = st.pop()
    a = st.pop()
    _push_bool(st, not _values_equal(a, b))


def prim_and(st):
    b = st.pop_truthy()
    a = st.pop_truthy()
    _push_bool(st, a and b)


def prim_or(st):
    b = st.pop_truthy()
    a = st.pop_truthy()
    _push_bool(st, a or b)


def prim_not(st):
    _push_bool(st, not st.pop_truthy())


def prim_dup(st):
    i, t, o = st.peek_parts(0)
    st.push_parts(i, t, o)


def prim_2drop(st):
    st.pop()
    st.pop()


def prim_2dup(st):
    i1, t1, o1 = st.peek_parts(1)
    i0, t0, o0 = st.peek_parts(0)
    st.push_parts(i1, t1, o1)
    st.push_parts(i0, t0, o0)


def prim_2over(st):
    i2, t2, o2 = st.peek_parts(2)
    i1, t1, o1 = st.peek_parts(1)
    st.push_parts(i2, t2, o2)
    st.push_parts(i1, t1, o1)


def prim_swap(st):
    i0, t0, o0 = st.peek_parts(0)
    st.pop()
    i1, t1, o1 = st.peek_parts(0)
    st.pop()
    st.push_parts(i0, t0, o0)
    st.push_parts(i1, t1, o1)


def prim_nip(st):
    i0, t0, o0 = st.peek_parts(0)
    st.pop()
    st.pop()
    st.push_parts(i0, t0, o0)


def prim_pop(st):
    st.pop()


def prim_over(st):
    i, t, o = st.peek_parts(1)
    st.push_parts(i, t, o)


def prim_pick(st):
    # pick ( x y z -- x y z x ): copy the third element from the top.
    i, t, o = st.peek_parts(2)
    st.push_parts(i, t, o)


def prim_3dup(st):
    # 3dup ( x y z -- x y z x y z )
    i2, t2, o2 = st.peek_parts(2)
    i1, t1, o1 = st.peek_parts(1)
    i0, t0, o0 = st.peek_parts(0)
    st.push_parts(i2, t2, o2)
    st.push_parts(i1, t1, o1)
    st.push_parts(i0, t0, o0)


def prim_rot(st):
    i0, t0, o0 = st.peek_parts(0)
    st.pop()
    i1, t1, o1 = st.peek_parts(0)
    st.pop()
    i2, t2, o2 = st.peek_parts(0)
    st.pop()
    st.push_parts(i1, t1, o1)
    st.push_parts(i0, t0, o0)
    st.push_parts(i2, t2, o2)


def prim_rolldown(st):
    i0, t0, o0 = st.peek_parts(0)
    st.pop()
    i1, t1, o1 = st.peek_parts(0)
    st.pop()
    i2, t2, o2 = st.peek_parts(0)
    st.pop()
    st.push_parts(i0, t0, o0)
    st.push_parts(i2, t2, o2)
    st.push_parts(i1, t1, o1)


def prim_rollup(st):
    i0, t0, o0 = st.peek_parts(0)
    st.pop()
    i1, t1, o1 = st.peek_parts(0)
    st.pop()
    i2, t2, o2 = st.peek_parts(0)
    st.pop()
    st.push_parts(i1, t1, o1)
    st.push_parts(i2, t2, o2)
    st.push_parts(i0, t0, o0)


def prim_dupd(st):
    # dupd == [dup] dip : X Y -> X X Y
    i0, t0, o0 = st.peek_parts(0)
    st.pop()
    i1, t1, o1 = st.peek_parts(0)
    st.push_parts(i1, t1, o1)
    st.push_parts(i0, t0, o0)


def prim_swapd(st):
    # swapd == [swap] dip : X Y Z -> Y X Z
    i0, t0, o0 = st.peek_parts(0)
    st.pop()
    i1, t1, o1 = st.peek_parts(0)
    st.pop()
    i2, t2, o2 = st.peek_parts(0)
    st.pop()
    st.push_parts(i1, t1, o1)
    st.push_parts(i2, t2, o2)
    st.push_parts(i0, t0, o0)


def prim_stack(st):
    st.push(w_list_from_items(st.snapshot_flat()))


def prim_unstack(st):
    lst = _pop_list(st)
    items = []
    node = lst
    while isinstance(node, W_Cons):
        items.append(node.head)
        node = node.tail
    st.replace_items(items)


def prim_cons(st):
    lst = _pop_list(st)
    val = st.pop()
    st.push(W_Cons(val, lst))


def prim_uncons(st):
    lst = _pop_list(st)
    if not isinstance(lst, W_Cons):
        raise FactorError("uncons of empty list")
    st.push(lst.tail)
    st.push(lst.head)


def prim_swons(st):
    prim_swap(st)
    prim_cons(st)


def prim_first(st):
    lst = _pop_list(st)
    if not isinstance(lst, W_Cons):
        raise FactorError("first of empty list")
    st.push(lst.head)


def prim_rest(st):
    lst = _pop_list(st)
    if not isinstance(lst, W_Cons):
        raise FactorError("rest of empty list")
    st.push(lst.tail)


def prim_concat(st):
    b = _pop_list(st)
    a = _pop_list(st)
    heads = []
    node = a
    while isinstance(node, W_Cons):
        heads.append(node.head)
        node = node.tail
    node = b
    i = len(heads) - 1
    while i >= 0:
        node = W_Cons(heads[i], node)
        i -= 1
    st.push(node)


def prim_size(st):
    v = st.pop()
    if isinstance(v, W_List):
        _push_int(st, list_length(v))
    elif isinstance(v, W_Array):
        _push_int(st, len(v.items))
    else:
        raise FactorError("expected list or array")


def prim_null(st):
    if st.top_is_int():
        _push_bool(st, st.pop_int() == 0)
        return
    v = st.pop()
    if isinstance(v, W_Int):
        _push_bool(st, v.val == 0)
    elif isinstance(v, W_List):
        _push_bool(st, list_is_empty(v))
    elif isinstance(v, W_String):
        _push_bool(st, len(v.s) == 0)
    else:
        _push_bool(st, False)


def prim_small(st):
    lst = _pop_list(st)
    if not isinstance(lst, W_Cons):
        _push_bool(st, True)
        return
    _push_bool(st, list_is_empty(lst.tail))


def prim_reverse(st):
    lst = _pop_list(st)
    out = nil_list()
    node = lst
    while isinstance(node, W_Cons):
        out = W_Cons(node.head, out)
        node = node.tail
    st.push(out)


def prim_succ(st):
    _push_int(st, _pop_int(st) + 1)


def prim_pred(st):
    n = _pop_int(st)
    if n > 0:
        n -= 1
    _push_int(st, n)


def prim_dot(st):
    _, stdout, _ = create_stdio()
    if st.top_is_int():
        stdout.write("%d " % st.pop_int())
        return
    v = st.pop()
    if isinstance(v, W_Int):
        stdout.write("%d " % v.val)
    elif isinstance(v, W_String):
        stdout.write("%s " % v.s)
    elif isinstance(v, W_List):
        stdout.write("[")
        node = v
        first = True
        while isinstance(node, W_Cons):
            if not first:
                stdout.write(" ")
            first = False
            it = node.head
            if isinstance(it, W_Int):
                stdout.write("%d" % it.val)
            elif isinstance(it, W_String):
                stdout.write(it.s)
            else:
                stdout.write("?")
            node = node.tail
        stdout.write("] ")
    else:
        stdout.write("? ")


def prim_put(st):
    _, stdout, _ = create_stdio()
    if st.top_is_int():
        stdout.write("%d" % st.pop_int())
        return
    v = st.pop()
    if isinstance(v, W_Int):
        stdout.write("%d" % v.val)
    elif isinstance(v, W_String):
        stdout.write(v.s)
    else:
        stdout.write("?")


def prim_putchars(st):
    _, stdout, _ = create_stdio()
    stdout.write(_pop_string(st))


def prim_nil(st):
    st.push(nil_list())


def prim_new_array(st):
    # <array> ( n -- arr ): mutable array of n zero cells (Phase B).
    n = _pop_int(st)
    if n < 0:
        raise FactorError("<array> expects non-negative integer")
    items = []
    i = 0
    while i < n:
        items.append(W_Int(0))
        i += 1
    st.push(W_Array(items))


def prim_nth(st):
    # nth ( n seq -- elt ): works on both W_List and W_Array.
    seq = st.pop()
    n = _pop_int(st)
    if isinstance(seq, W_Array):
        items = seq.items
        if n < 0 or n >= len(items):
            raise FactorError("nth index out of range")
        st.push(items[n])
        return
    if isinstance(seq, W_List):
        if n < 0:
            raise FactorError("nth index out of range")
        node = seq
        while n > 0 and isinstance(node, W_Cons):
            node = node.tail
            n -= 1
        if not isinstance(node, W_Cons):
            raise FactorError("nth index out of range")
        st.push(node.head)
        return
    raise FactorError("nth expects array or list")


def prim_set_nth(st):
    # set-nth ( elt n seq -- ): W_Array only (mutates in place).
    seq = st.pop()
    if not isinstance(seq, W_Array):
        raise FactorError("set-nth expects array")
    n = _pop_int(st)
    val = st.pop()
    if n < 0 or n >= len(seq.items):
        raise FactorError("set-nth index out of range")
    seq.items[n] = val


def prim_intern(st):
    st.push(W_Symbol.intern(_pop_string(st)))


def prim_name(st):
    sym = _pop_symbol(st)
    st.push(W_String(sym.name))


def prim_clock(st):
    _push_int(st, int(time.time() * 1000000.0))


def is_combinator(name):
    return (name == "i" or name == "call" or name == "x" or name == "dip" or
            name == "ifte" or name == "if" or name == "branch" or name == "times" or
            name == "step" or name == "each" or name == "map" or name == "fold" or
            name == "reduce" or name == "filter" or name == "while" or
            name == "linrec" or name == "binrec" or name == "primrec")


def is_primitive(name):
    return (name == "+" or name == "-" or name == "*" or name == "/" or
            name == "rem" or name == "mod" or name == "<" or name == "<=" or
            name == ">" or name == ">=" or name == "=" or name == "!=" or
            name == "and" or name == "or" or name == "not" or name == "dup" or
            name == "2dup" or name == "2drop" or name == "2over" or name == "swap" or
            name == "nip" or name == "pop" or name == "drop" or name == "over" or
            name == "pick" or name == "3dup" or
            name == "rot" or name == "rolldown" or name == "-rot" or name == "rollup" or
            name == "dupd" or name == "swapd" or name == "stack" or name == "unstack" or
            name == "cons" or name == "uncons" or name == "swons" or name == "first" or
            name == "rest" or name == "concat" or name == "size" or name == "length" or
            name == "null" or name == "empty?" or name == "small" or name == "reverse" or
            name == "nil" or name == "succ" or name == "1+" or name == "pred" or
            name == "1-" or name == "." or name == "put" or name == "putchars" or
            name == "intern" or name == "name" or name == "clock" or
            name == "<array>" or name == "nth" or name == "set-nth")
