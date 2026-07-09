"""Built-in Joy words for rpyjoy (P1 naive interpreter)."""

from rpython.rlib.rfile import create_stdio

from rpyjoy.values import (
    W_Int, W_Bool, W_String, W_Symbol, W_List, W_Quotation,
    JoyError, w_true, w_false, truthy,
)
from rpyjoy.program import item_to_value, LitQuot, CallWord


def _pop_int(st):
    v = st.pop()
    if not isinstance(v, W_Int):
        raise JoyError("expected integer")
    return v.val


def _pop_bool(st):
    v = st.pop()
    return w_true() if truthy(v) else w_false()


def _pop_list(st):
    v = st.pop()
    if not isinstance(v, W_List):
        raise JoyError("expected list")
    return v


def _pop_quot(st):
    v = st.pop()
    if not isinstance(v, W_Quotation):
        raise JoyError("expected quotation")
    return v


def _pop_string(st):
    v = st.pop()
    if not isinstance(v, W_String):
        raise JoyError("expected string")
    return v.s


def _pop_symbol(st):
    v = st.pop()
    if isinstance(v, W_Symbol):
        return v
    if isinstance(v, W_String):
        return W_Symbol.intern(v.s)
    raise JoyError("expected symbol")


def _push_bool(st, b):
    st.push(w_true() if b else w_false())


def _values_equal(a, b):
    if isinstance(a, W_Int) and isinstance(b, W_Int):
        return a.val == b.val
    if isinstance(a, W_Bool) and isinstance(b, W_Bool):
        return a.val == b.val
    if isinstance(a, W_String) and isinstance(b, W_String):
        return a.s == b.s
    if isinstance(a, W_Symbol) and isinstance(b, W_Symbol):
        return a.name == b.name
    if isinstance(a, W_List) and isinstance(b, W_List):
        if len(a.items) != len(b.items):
            return False
        i = 0
        while i < len(a.items):
            if not _values_equal(a.items[i], b.items[i]):
                return False
            i += 1
        return True
    if isinstance(a, W_Quotation) and isinstance(b, W_Quotation):
        return a.program == b.program
    return False


def _bin_int_op(st, fn):
    b = _pop_int(st)
    a = _pop_int(st)
    st.push(W_Int(fn(a, b)))


def _add(a, b):
    return a + b


def _sub(a, b):
    return a - b


def _mul(a, b):
    return a * b


def prim_add(st):
    _bin_int_op(st, _add)


def prim_sub(st):
    _bin_int_op(st, _sub)


def prim_mul(st):
    _bin_int_op(st, _mul)


def prim_div(st):
    b = _pop_int(st)
    a = _pop_int(st)
    if b == 0:
        raise JoyError("division by zero")
    st.push(W_Int(a // b))


def prim_rem(st):
    b = _pop_int(st)
    a = _pop_int(st)
    if b == 0:
        raise JoyError("division by zero")
    st.push(W_Int(a % b))


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
    b = st.pop()
    a = st.pop()
    _push_bool(st, _values_equal(a, b))


def prim_ne(st):
    b = st.pop()
    a = st.pop()
    _push_bool(st, not _values_equal(a, b))


def prim_and(st):
    b = truthy(st.pop())
    a = truthy(st.pop())
    _push_bool(st, a and b)


def prim_or(st):
    b = truthy(st.pop())
    a = truthy(st.pop())
    _push_bool(st, a or b)


def prim_not(st):
    _push_bool(st, not truthy(st.pop()))


def prim_dup(st):
    st.push(st.peek(0))


def prim_2drop(st):
    st.pop()
    st.pop()


def prim_2dup(st):
    st.push(st.peek(1))
    st.push(st.peek(1))


def prim_2over(st):
    st.push(st.peek(2))
    st.push(st.peek(2))


def prim_swap(st):
    a = st.pop()
    b = st.pop()
    st.push(a)
    st.push(b)


def prim_nip(st):
    a = st.pop()
    st.pop()
    st.push(a)


def prim_pop(st):
    st.pop()


def prim_over(st):
    st.push(st.peek(1))


def prim_rot(st):
    c = st.pop()
    b = st.pop()
    a = st.pop()
    st.push(b)
    st.push(c)
    st.push(a)


def prim_rolldown(st):
    c = st.pop()
    b = st.pop()
    a = st.pop()
    st.push(c)
    st.push(a)
    st.push(b)


def prim_rollup(st):
    c = st.pop()
    b = st.pop()
    a = st.pop()
    st.push(b)
    st.push(a)
    st.push(c)


def prim_dupd(st):
    st.push(st.peek(1))


def prim_swapd(st):
    c = st.pop()
    b = st.pop()
    a = st.pop()
    st.push(a)
    st.push(c)
    st.push(b)


def prim_stack(st):
    st.push(W_List(list(st.items)))


def prim_unstack(st):
    lst = _pop_list(st)
    st.replace_items(list(lst.items))


def prim_cons(st):
    lst = _pop_list(st)
    val = st.pop()
    st.push(W_List([val] + lst.items))


def prim_uncons(st):
    lst = _pop_list(st)
    if not lst.items:
        raise JoyError("uncons of empty list")
    st.push(W_List(lst.items[1:]))
    st.push(lst.items[0])


def prim_swons(st):
    prim_swap(st)
    prim_cons(st)


def prim_first(st):
    lst = _pop_list(st)
    if not lst.items:
        raise JoyError("first of empty list")
    st.push(lst.items[0])


def prim_rest(st):
    lst = _pop_list(st)
    if not lst.items:
        raise JoyError("rest of empty list")
    st.push(W_List(lst.items[1:]))


def prim_concat(st):
    b = _pop_list(st)
    a = _pop_list(st)
    st.push(W_List(a.items + b.items))


def prim_size(st):
    lst = _pop_list(st)
    st.push(W_Int(len(lst.items)))


def prim_null(st):
    v = st.pop()
    if isinstance(v, W_Int):
        _push_bool(st, v.val == 0)
    elif isinstance(v, W_List):
        _push_bool(st, len(v.items) == 0)
    elif isinstance(v, W_String):
        _push_bool(st, len(v.s) == 0)
    else:
        _push_bool(st, False)


def prim_small(st):
    lst = _pop_list(st)
    _push_bool(st, len(lst.items) <= 1)


def prim_reverse(st):
    lst = _pop_list(st)
    items = lst.items
    out = []
    i = len(items) - 1
    while i >= 0:
        out.append(items[i])
        i -= 1
    st.push(W_List(out))


def prim_succ(st):
    st.push(W_Int(_pop_int(st) + 1))


def prim_pred(st):
    n = _pop_int(st)
    if n > 0:
        n -= 1
    st.push(W_Int(n))


def prim_dot(st):
    v = st.pop()
    _, stdout, _ = create_stdio()
    if isinstance(v, W_Int):
        stdout.write("%d " % v.val)
    elif isinstance(v, W_Bool):
        stdout.write("%s " % ("true" if v.val else "false"))
    elif isinstance(v, W_String):
        stdout.write("%s " % v.s)
    elif isinstance(v, W_List):
        stdout.write("[")
        for i, it in enumerate(v.items):
            if i:
                stdout.write(" ")
            if isinstance(it, W_Int):
                stdout.write("%d" % it.val)
            elif isinstance(it, W_Bool):
                stdout.write("true" if it.val else "false")
            elif isinstance(it, W_String):
                stdout.write(it.s)
            else:
                stdout.write("?")
        stdout.write("] ")
    else:
        stdout.write("? ")


def prim_put(st):
    v = st.pop()
    _, stdout, _ = create_stdio()
    if isinstance(v, W_Int):
        stdout.write("%d" % v.val)
    elif isinstance(v, W_String):
        stdout.write(v.s)
    elif isinstance(v, W_Bool):
        stdout.write("true" if v.val else "false")
    else:
        stdout.write("?")


def prim_putchars(st):
    _, stdout, _ = create_stdio()
    stdout.write(_pop_string(st))


def prim_nil(st):
    st.push(W_List([]))


def prim_intern(st):
    st.push(W_Symbol.intern(_pop_string(st)))


def prim_name(st):
    sym = _pop_symbol(st)
    st.push(W_String(sym.name))


def is_combinator(name):
    return (name == "i" or name == "x" or name == "dip" or name == "ifte" or
            name == "branch" or name == "times" or name == "step" or name == "map" or
            name == "fold" or name == "filter" or name == "while" or name == "linrec" or
            name == "binrec" or name == "primrec")


def make_prim_table(interp):
    """Return name -> stack handler."""
    table = {
        "+": prim_add,
        "-": prim_sub,
        "*": prim_mul,
        "/": prim_div,
        "rem": prim_rem,
        "<": prim_lt,
        "<=": prim_le,
        ">": prim_gt,
        ">=": prim_ge,
        "=": prim_eq,
        "!=": prim_ne,
        "and": prim_and,
        "or": prim_or,
        "not": prim_not,
        "dup": prim_dup,
        "2dup": prim_2dup,
        "2drop": prim_2drop,
        "2over": prim_2over,
        "swap": prim_swap,
        "nip": prim_nip,
        "pop": prim_pop,
        "over": prim_over,
        "rot": prim_rot,
        "rolldown": prim_rolldown,
        "rollup": prim_rollup,
        "dupd": prim_dupd,
        "swapd": prim_swapd,
        "stack": prim_stack,
        "unstack": prim_unstack,
        "cons": prim_cons,
        "uncons": prim_uncons,
        "swons": prim_swons,
        "first": prim_first,
        "rest": prim_rest,
        "concat": prim_concat,
        "size": prim_size,
        "null": prim_null,
        "small": prim_small,
        "reverse": prim_reverse,
        "nil": prim_nil,
        "succ": prim_succ,
        "pred": prim_pred,
        ".": prim_dot,
        "put": prim_put,
        "putchars": prim_putchars,
        "intern": prim_intern,
        "name": prim_name,
    }
    return table
