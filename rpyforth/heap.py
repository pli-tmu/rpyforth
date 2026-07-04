from rpython.rtyper.lltypesystem import lltype, rffi
from rpython.rlib.rarithmetic import intmask

from rpyforth.objects import W_FloatObject, make_int

# Byte-addressed data space backed by one raw (GC-untracked) byte buffer, so
# C! / C@ and ! / @ reinterpret the same storage exactly like real Forth
# memory: a cell is the 8 little-endian bytes at its address, and both views
# compile to single raw loads/stores under the JIT.
HEAP_SIZE_BYTES = 1 << 23
HEAP_CELL_COUNT = HEAP_SIZE_BYTES >> 3

CELL_BYTES = 8


class Heap(object):
    # The raw buffer pointer never changes after __init__, so the JIT hoists
    # the load once per trace.
    _immutable_fields_ = ["size", "raw"]

    def __init__(self, size):
        assert size & 7 == 0
        self.size = size
        self.raw = lltype.malloc(rffi.CCHARP.TO, size, flavor='raw', zero=True)

    def __del__(self):
        lltype.free(self.raw, flavor='raw')

    def _ensure_addr(self, addr, span):
        assert 0 <= addr
        assert addr + span <= self.size

    def cell_store(self, addr, intval):
        self._ensure_addr(addr, CELL_BYTES)
        p = rffi.cast(rffi.SIGNEDP, rffi.ptradd(self.raw, addr))
        p[0] = rffi.cast(rffi.SIGNED, intval)

    def cell_fetch_int(self, addr):
        self._ensure_addr(addr, CELL_BYTES)
        p = rffi.cast(rffi.SIGNEDP, rffi.ptradd(self.raw, addr))
        return intmask(p[0])

    def cell_fetch(self, addr):
        return make_int(self.cell_fetch_int(addr))

    def char_store(self, addr, intval):
        self._ensure_addr(addr, 1)
        self.raw[addr] = chr(intval & 0xFF)

    def char_fetch(self, addr):
        self._ensure_addr(addr, 1)
        return ord(self.raw[addr])

    def fill_bytes(self, addr, count, b):
        if count <= 0:
            return
        self._ensure_addr(addr, count)
        c = chr(b & 0xFF)
        i = 0
        while i < count:
            self.raw[addr + i] = c
            i += 1

    def move_bytes(self, src, dst, count):
        """memmove semantics: overlap-safe byte copy."""
        if count <= 0 or src == dst:
            return
        self._ensure_addr(src, count)
        self._ensure_addr(dst, count)
        if dst < src:
            i = 0
            while i < count:
                self.raw[dst + i] = self.raw[src + i]
                i += 1
        else:
            i = count - 1
            while i >= 0:
                self.raw[dst + i] = self.raw[src + i]
                i -= 1

    def float_store(self, addr, value):
        self._ensure_addr(addr, CELL_BYTES)
        p = rffi.cast(rffi.DOUBLEP, rffi.ptradd(self.raw, addr))
        p[0] = value

    def float_fetch_float(self, addr):
        self._ensure_addr(addr, CELL_BYTES)
        p = rffi.cast(rffi.DOUBLEP, rffi.ptradd(self.raw, addr))
        return p[0]

    def float_fetch(self, addr):
        return W_FloatObject(self.float_fetch_float(addr))
