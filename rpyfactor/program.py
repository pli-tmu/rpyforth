"""Program representation: compile-time items executed by the interpreter.

Items are parse-time constants: every field is immutable and literal items
pre-build their runtime value once. Pushing a quotation must reuse the same
W_Quotation (and the same body list) on every encounter -- a fresh copy per
push gives each trace a different green key and makes the promoted program
a trace-local virtual, which aborts the loop (promote-of-virtual).
"""

from rpyfactor.values import (
    W_Int, W_String, W_Symbol, W_List, W_Quotation, w_list_from_items,
)


class ProgramItem(object):
    pass


class LitInt(ProgramItem):
    _immutable_fields_ = ["n"]

    def __init__(self, n):
        self.n = int(n)

    def __repr__(self):
        return "LitInt(%d)" % (self.n,)


class LitBool(ProgramItem):
    _immutable_fields_ = ["b"]

    def __init__(self, b):
        self.b = bool(b)

    def __repr__(self):
        return "LitBool(%s)" % (self.b,)


class LitString(ProgramItem):
    _immutable_fields_ = ["s", "w_val"]

    def __init__(self, s):
        self.s = str(s)
        self.w_val = W_String(self.s)

    def __repr__(self):
        return "LitString(%r)" % (self.s,)


class LitSymbol(ProgramItem):
    _immutable_fields_ = ["name", "w_val"]

    def __init__(self, name):
        self.name = str(name)
        self.w_val = W_Symbol.intern(self.name)

    def __repr__(self):
        return "LitSymbol(%r)" % (self.name,)


class LitQuot(ProgramItem):
    _immutable_fields_ = ["body", "w_val"]

    def __init__(self, body):
        self.body = body
        self.w_val = W_Quotation(body)

    def __repr__(self):
        return "LitQuot(len=%d)" % (len(self.body),)


class LitArray(ProgramItem):
    """Factor `{ ... }` array literal; elements are program items."""
    _immutable_fields_ = ["elems", "w_val?"]

    def __init__(self, elems):
        self.elems = elems
        self.w_val = None

    def materialize(self):
        if self.w_val is None:
            values = []
            i = 0
            while i < len(self.elems):
                values.append(item_to_value(self.elems[i]))
                i += 1
            self.w_val = w_list_from_items(values)
        return self.w_val

    def __repr__(self):
        return "LitArray(len=%d)" % (len(self.elems),)


class Word(object):
    _immutable_fields_ = ["body?"]

    def __init__(self, body):
        self.body = body

    def redefine(self, body):
        self.body = body


class CallWord(ProgramItem):
    _immutable_fields_ = ["name", "cell?"]

    def __init__(self, name):
        self.name = str(name)
        self.cell = None

    def __repr__(self):
        return "CallWord(%r)" % (self.name,)


def item_to_value(item):
    if isinstance(item, LitInt):
        return W_Int(item.n)
    if isinstance(item, LitBool):
        return W_Int(1 if item.b else 0)
    if isinstance(item, LitString):
        return item.w_val
    if isinstance(item, LitSymbol):
        return item.w_val
    if isinstance(item, LitQuot):
        return item.w_val
    if isinstance(item, LitArray):
        return item.materialize()
    raise TypeError("not a literal item")


def is_literal_item(item):
    return (isinstance(item, LitInt) or isinstance(item, LitBool) or
            isinstance(item, LitString) or isinstance(item, LitSymbol) or
            isinstance(item, LitQuot) or isinstance(item, LitArray))
