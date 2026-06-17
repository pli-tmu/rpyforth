from rpyforth.objects import ZERO, W_FloatObject, make_int

HEAP_CELL_COUNT = 65536
HEAP_SIZE_BYTES = HEAP_CELL_COUNT

TAG_CELL = 1


class Heap(object):
    """Lazy unboxed data space: bytes (C!), cells+tags (@/!), float_mem (f@/f!)."""

    _immutable_fields_ = ["size"]

    def __init__(self, size):
        self.size = size
        self.bytes = [0] * size
        self.cells = None
        self.tags = None
        self.float_mem = None

    def _ensure_cells(self):
        if self.cells is None:
            self.cells = [0] * self.size
            self.tags = [0] * self.size

    def _ensure_floats(self):
        if self.float_mem is None:
            self.float_mem = [0.0] * self.size

    def _ensure_addr(self, addr, span):
        assert 0 <= addr < self.size
        assert addr + span <= self.size

    def cell_tagged(self, addr):
        if self.tags is None:
            return False
        return self.tags[addr] & TAG_CELL

    def cell_store(self, addr, intval):
        self._ensure_addr(addr, 1)
        self._ensure_cells()
        self.cells[addr] = intval
        self.tags[addr] = self.tags[addr] | TAG_CELL

    def cell_fetch_int(self, addr):
        self._ensure_addr(addr, 1)
        if self.cells is None:
            return 0
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
        self._ensure_floats()
        self.float_mem[addr] = value

    def float_fetch_float(self, addr):
        self._ensure_addr(addr, 1)
        if self.float_mem is None:
            return 0.0
        return self.float_mem[addr]

    def float_fetch(self, addr):
        return W_FloatObject(self.float_fetch_float(addr))
