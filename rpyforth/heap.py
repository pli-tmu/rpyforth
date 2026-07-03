from rpyforth.objects import ZERO, W_FloatObject, make_int
from rpython.rlib.debug import make_sure_not_resized

HEAP_CELL_COUNT = 65536
HEAP_SIZE_BYTES = HEAP_CELL_COUNT

TAG_CELL = 1


class Heap(object):
    # bytes/cells/tags/float_mem are reference-immutable: the array pointer
    # never changes after __init__, so the JIT can hoist the load once.
    _immutable_fields_ = ["size", "bytes", "cells", "tags", "float_mem"]

    def __init__(self, size):
        self.size = size
        self.bytes = make_sure_not_resized([0] * size)
        self.cells = make_sure_not_resized([0] * size)
        self.tags = make_sure_not_resized([0] * size)
        self.float_mem = make_sure_not_resized([0.0] * size)

    def _ensure_addr(self, addr, span):
        assert 0 <= addr < self.size
        assert addr + span <= self.size

    def cell_tagged(self, addr):
        return self.tags[addr] & TAG_CELL

    def cell_store(self, addr, intval):
        self._ensure_addr(addr, 1)
        self.cells[addr] = intval
        self.tags[addr] = TAG_CELL

    def cell_fetch_int(self, addr):
        self._ensure_addr(addr, 1)
        return self.cells[addr]

    def cell_fetch(self, addr):
        self._ensure_addr(addr, 1)
        if not self.cell_tagged(addr):
            return ZERO
        return make_int(self.cells[addr])

    def char_store(self, addr, intval):
        self._ensure_addr(addr, 1)
        self.bytes[addr] = intval & 0xFF

    def char_fetch(self, addr):
        self._ensure_addr(addr, 1)
        return self.bytes[addr]

    def float_store(self, addr, value):
        self._ensure_addr(addr, 1)
        self.float_mem[addr] = value

    def float_fetch_float(self, addr):
        self._ensure_addr(addr, 1)
        return self.float_mem[addr]

    def float_fetch(self, addr):
        return W_FloatObject(self.float_fetch_float(addr))
