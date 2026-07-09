"""Program representation: compile-time items executed by the interpreter."""

from rpyjoy.values import W_Int, W_Bool, W_String, W_Symbol, W_List, W_Quotation


class ProgramItem(object):
    pass


class LitInt(ProgramItem):
    def __init__(self, n):
        self.n = int(n)

    def __repr__(self):
        return "LitInt(%d)" % (self.n,)


class LitBool(ProgramItem):
    def __init__(self, b):
        self.b = bool(b)

    def __repr__(self):
        return "LitBool(%s)" % (self.b,)


class LitString(ProgramItem):
    def __init__(self, s):
        self.s = str(s)

    def __repr__(self):
        return "LitString(%r)" % (self.s,)


class LitSymbol(ProgramItem):
    def __init__(self, name):
        self.name = str(name)

    def __repr__(self):
        return "LitSymbol(%r)" % (self.name,)


class LitQuot(ProgramItem):
    def __init__(self, body):
        self.body = body

    def __repr__(self):
        return "LitQuot(len=%d)" % (len(self.body),)


class CallWord(ProgramItem):
    def __init__(self, name):
        self.name = str(name)

    def __repr__(self):
        return "CallWord(%r)" % (self.name,)


def item_to_value(item):
    if isinstance(item, LitInt):
        return W_Int(item.n)
    if isinstance(item, LitBool):
        return W_Bool(item.b)
    if isinstance(item, LitString):
        return W_String(item.s)
    if isinstance(item, LitSymbol):
        return W_Symbol.intern(item.name)
    if isinstance(item, LitQuot):
        return W_Quotation(item.body)
    raise TypeError("not a literal item")


def is_literal_item(item):
    return (isinstance(item, LitInt) or isinstance(item, LitBool) or
            isinstance(item, LitString) or isinstance(item, LitSymbol) or
            isinstance(item, LitQuot))
