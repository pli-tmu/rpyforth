STACK_SIZE = 256

FRAGMENT_SIZE = 256
CALL_WINDOW = 8

class DSMetaStack(object):
    pass

class DSFragment(object):
    pass

class DSIntMetaStack(DSMetaStack):
    def __init__(self):
        self.current = DSIntFragment(None)
        self.top0 = 0; self.top1 = 0
        self.top2 = 0; self.top3 = 0
        self.top_count = 0

class DSFloatMetaStack(DSMetaStack):
    def __init__(self):
        self.current = DSFloatFragment(None)
        self.top0 = 0.0; self.top1 = 0.0
        self.top2 = 0.0; self.top3 = 0.0
        self.top_count = 0

class DSObjMetaStack(DSMetaStack):
    def __init__(self):
        self.current = DSIntFragment(None)
        self.top0 = None; self.top1 = None
        self.top2 = None; self.top3 = None
        self.top_count = 0

class DSIntFragment(DSFragment):
    _immutable_fields_ = ["parent", "cells"]

    def __init__(self, parent):
        self.parent = parent
        self.cells = [0] * FRAGMENT_SIZE
        self.sp = 0
        self.saved_parent_sp = 0
        self.import_counts = 0

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
        self.cells = [0] * FRAGMENT_SIZE
        self.sp = 0
        self.saved_parent_sp = 0
        self.import_counts = 0

    def push(self, v):
        sp = self.sp + 1
        assert -1 < sp < FRAGMENT_SIZE
        self.cells[sp] = v
        self.sp = sp

    def pop(self):
        sp = self.sp
        assert -1 < sp < FRAGMENT_SIZE
        v = self.cells[sp]
        self.sp = sp - 1
        return v

class DSObjFragment(DSFragment):
    _immutable_fields_ = ["parent", "cells"]

    def __init__(self, parent):
        self.parent = parent
        self.cells = [0] * FRAGMENT_SIZE
        self.sp = 0
        self.saved_parent_sp = 0
        self.import_counts = 0

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
