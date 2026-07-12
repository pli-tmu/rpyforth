"""Boxed runtime values for rpyfactor.

W_List uses immutable Python tuples of W_Value for structural sharing on
cons/uncons/rest (P1 decision per SECOND_SESL_PLAN.md §2).
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
    """Immutable list of values (tuple-backed)."""

    _immutable_ = True

    def __init__(self, items):
        if isinstance(items, W_List):
            self.items = items.items
        elif isinstance(items, tuple):
            self.items = list(items)
        else:
            self.items = list(items)

    def __repr__(self):
        return "W_List(%r)" % (self.items,)

    def __eq__(self, other):
        if not isinstance(other, W_List):
            return False
        return self.items == other.items

    def __ne__(self, other):
        return not self.__eq__(other)


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
        return len(val.items) != 0
    if isinstance(val, W_Array):
        return len(val.items) != 0
    if isinstance(val, W_String):
        return len(val.s) != 0
    return True
