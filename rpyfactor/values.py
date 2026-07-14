"""Boxed runtime values for rpyfactor.

W_List holds a Python list of W_Value that is never mutated in place;
cons/uncons/rest build fresh lists, so backing storage can be shared on
copy (P1 decision per SECOND_SESL_PLAN.md §2).
"""


class FactorError(Exception):
    def __init__(self, msg):
        self.msg = str(msg)

    def __str__(self):
        return self.msg


class W_Value(object):
    pass


class W_Int(W_Value):
    _immutable_ = True

    def __init__(self, val):
        self.val = int(val)

    def __repr__(self):
        return "W_Int(%s)" % (self.val,)

    def __eq__(self, other):
        return isinstance(other, W_Int) and self.val == other.val

    def __ne__(self, other):
        return not self.__eq__(other)


class W_Bool(W_Value):
    _immutable_ = True

    def __init__(self, val):
        self.val = bool(val)

    def __repr__(self):
        return "W_Bool(%s)" % (self.val,)

    def __eq__(self, other):
        return isinstance(other, W_Bool) and self.val == other.val

    def __ne__(self, other):
        return not self.__eq__(other)


class W_String(W_Value):
    _immutable_ = True

    def __init__(self, s):
        self.s = str(s)

    def __repr__(self):
        return "W_String(%r)" % (self.s,)

    def __eq__(self, other):
        return isinstance(other, W_String) and self.s == other.s

    def __ne__(self, other):
        return not self.__eq__(other)


class W_Symbol(W_Value):
    _immutable_ = True

    def __init__(self, name):
        self.name = str(name)

    @staticmethod
    def intern(name):
        return W_Symbol(str(name))

    def __repr__(self):
        return "W_Symbol(%r)" % (self.name,)

    def __eq__(self, other):
        return isinstance(other, W_Symbol) and self.name == other.name

    def __ne__(self, other):
        return not self.__eq__(other)


class W_List(W_Value):
    """Immutable singly-linked list (cons cells).

    Two concrete forms: W_Nil (the empty singleton) and W_Cons (head + tail).
    Both head and tail are immutable, so a cons cell is a compile-time
    constant once built from a literal and stays foldable in traces.
    cons/first/rest/swons/null? are O(1); size/concat/reverse traverse once.

    ``items`` is a read-only Python-list view over the chain, kept only for
    tests and diagnostics -- never call it on a hot path."""

    _immutable_ = True

    @property
    def items(self):
        out = []
        node = self
        while isinstance(node, W_Cons):
            out.append(node.head)
            node = node.tail
        return out


class W_Nil(W_List):
    _immutable_ = True

    def __repr__(self):
        return "W_Nil()"

    def __eq__(self, other):
        return isinstance(other, W_Nil)

    def __ne__(self, other):
        return not self.__eq__(other)


_NIL = W_Nil()


def nil_list():
    return _NIL


class W_Cons(W_List):
    _immutable_ = True
    _immutable_fields_ = ["head", "tail"]

    def __init__(self, head, tail):
        self.head = head
        self.tail = tail

    def __repr__(self):
        return "W_Cons(%r, %r)" % (self.head, self.tail)

    def __eq__(self, other):
        if not isinstance(other, W_List):
            return False
        return self.items == other.items

    def __ne__(self, other):
        return not self.__eq__(other)


def w_list_from_items(items):
    node = nil_list()
    i = len(items) - 1
    while i >= 0:
        node = W_Cons(items[i], node)
        i -= 1
    return node


def list_is_empty(lst):
    return not isinstance(lst, W_Cons)


def list_length(lst):
    n = 0
    node = lst
    while isinstance(node, W_Cons):
        n += 1
        node = node.tail
    return n


class W_Array(W_Value):
    """Mutable fixed-size array (Phase B). Backing python list is mutated
    in place by set-nth -- unlike W_List, the object identity is shared
    across dup/pushes so mutation is visible through every reference."""

    def __init__(self, items):
        self.items = list(items)

    def __repr__(self):
        return "W_Array(%r)" % (self.items,)

    def __eq__(self, other):
        return isinstance(other, W_Array) and self.items == other.items

    def __ne__(self, other):
        return not self.__eq__(other)


class W_Quotation(W_Value):
    """Executable program (list of ProgramItem from program.py).

    The body list is shared, never copied: quotation identity (and the JIT
    green key of any trace entering it) must be stable across pushes."""

    _immutable_ = True

    def __init__(self, program):
        self.program = program

    def __repr__(self):
        return "W_Quotation(len=%d)" % (len(self.program),)

    def __eq__(self, other):
        if not isinstance(other, W_Quotation):
            return False
        return self.program == other.program

    def __ne__(self, other):
        return not self.__eq__(other)


def w_true():
    return W_Bool(True)


def w_false():
    return W_Bool(False)


def truthy(val):
    if isinstance(val, W_Int):
        return val.val != 0
    if isinstance(val, W_List):
        return isinstance(val, W_Cons)
    if isinstance(val, W_Array):
        return len(val.items) != 0
    if isinstance(val, W_String):
        return len(val.s) != 0
    return True
