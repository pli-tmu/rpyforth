from rpyforth.objects import (
    DECIMAL,
    Word,
    CodeThread,
    ZERO,
    W_Object,
    W_IntObject,
    W_StringObject,
    W_PtrObject,
    W_FloatObject,
    CELL_SIZE_BYTES,
    CELL_SIZE,
    make_int,
    THREAD_REGISTRY,
)


import os

from rpython.rlib.jit import JitDriver, promote, elidable, unroll_safe, promote_string, hint
from rpython.rlib.debug import make_sure_not_resized
from rpython.rlib.rfile import create_stdio

from rpyforth.heap import HEAP_CELL_COUNT, HEAP_SIZE_BYTES, Heap

USE_VIRTUALIZATION = bool(os.environ.get("RPYFORTH_VIRTUALIZE"))

USE_STACK_FRAGMENT = bool(os.environ.get("RPYFORTH_STACK_FRAGMENT"))

from rpyforth.metastack import (
    STACK_FRAGMENT_VIRTUALIZABLES,
    push_ds_fragments,
    pop_ds_fragments_commit,
    reset_ds_fragments,
)

if USE_STACK_FRAGMENT:
    from rpyforth.metastack_int import DSIntMetaStack
    InterpBase = DSIntMetaStack
else:
    class InterpBase(object):
        pass

STACK_SIZE = 16384  # Increased for deeper nesting (ack(3,10))
BUF_SIZE = 1024

# Sentinel value for EXIT - indicates return from current definition
EXIT_SENTINEL = -1

# Sentinel value for TAILCALL - indicates a tail call to another word
TAILCALL_SENTINEL = -2

class Exit(Exception):
    pass

class Bye(Exception):
    """Raised by BYE to exit the Forth system cleanly."""
    pass

def get_printable_location(ip, thread):
    return "ip=%d %s %s" % (ip, thread.code[ip].to_string(), thread.lits[ip].to_string())

if USE_STACK_FRAGMENT:
    jitdriver = JitDriver(
        greens=['ip', 'thread'],
        reds=['self'],
        virtualizables=['self'],
        get_printable_location=get_printable_location,
    )
elif USE_VIRTUALIZATION:
    jitdriver = JitDriver(
        greens=['ip', 'thread'],
        reds=['self'],
        virtualizables=['self'],
        get_printable_location=get_printable_location,
    )
else:
    jitdriver = JitDriver(
        greens=['ip', 'thread'],
        reds=['self'],
        get_printable_location=get_printable_location,
    )


class InnerInterpreter(InterpBase, object):
    # The metastack arena is allocated once and never reassigned, so its
    # reference is immutable even though its elements are mutated in place; this
    # lets the JIT hoist the array-pointer load to the loop header instead of
    # reloading it on every spill/refill.
    _immutable_fields_ = ["cell_size", "cell_size_bytes", "base", "spill",
                          "lc_is", "lc_ls", "rs", "ds_locals"]

    if USE_VIRTUALIZATION:
        _virtualizable_ = ["ds_ints", "ds_floats",
                           "ds_ptr_ints", "ds_ptr_floats", "ds_ptr_locals",
                           "rs_ptr", "cs_tids", "cs_ips", "cs_ptr", "li",
                           "cell_size", "cell_size_bytes", "base"]
    elif USE_STACK_FRAGMENT:
        _virtualizable_ = STACK_FRAGMENT_VIRTUALIZABLES
    else:
        _virtualizable_ = []


    def __init__(self):
        # Reference to outer interpreter (set later)
        self.outer = None

        # Pre-allocate larger stacks to reduce growth overhead
        self.ds_ints = [0] * STACK_SIZE  # unboxed integer data stack (fixed path)
        self.ds_ptr_ints = 0

        self.ds_floats = [0.0] * STACK_SIZE
        self.ds_ptr_floats = 0

        self.ds_locals = [None] * STACK_SIZE
        self.ds_ptr_locals = 0

        self.rs = [0] * STACK_SIZE  # return stack
        self.rs_ptr = 0

        # dedicated loop-control stack
        self.li = 0
        self.ll = 0
        self.lc_depth = 0
        self.lc_is = [0] * STACK_SIZE
        self.lc_ls = [0] * STACK_SIZE

        # Virtualized call stack for JIT optimization. The return thread is stored
        # by id (int, no GC barrier) and recovered from THREAD_REGISTRY; the
        # foldable lookup removes the per-return thread guard.
        self.cs_tids = [0] * STACK_SIZE
        self.cs_ips = [0] * STACK_SIZE
        self.cs_ptr = 0

        self.heap = None
        self.cell_size = CELL_SIZE
        self.cell_size_bytes = CELL_SIZE_BYTES

        # for string
        self.buf = [None] * HEAP_SIZE_BYTES
        self.here = 0

        self.base = 10
        self._pno_active = False
        self._pno_buf = []
        self.argv = []

        if USE_STACK_FRAGMENT:
            self.init_fields()
        else:
            # Placeholders so the attribute set is consistent; unused on the
            # fixed path (the translator folds the dead USE_STACK_FRAGMENT
            # branch, so the active frame and fragment pool are only built when
            # fragmented).
            self.t0 = 0
            self.t1 = 0
            self.d = 0
            self.frame = [0]
            self.frag_ptr = 0
            self.spill = [0]
            self.spill_ptr = 0

    def push_call(self, thread, ip):
        """Push return address (thread id + ip) onto the virtualized call stack."""
        ptr = self.cs_ptr
        assert ptr < len(self.cs_tids)
        self.cs_tids[ptr] = thread.tid
        self.cs_ips[ptr] = ip
        self.cs_ptr = ptr + 1

    def pop_call(self):
        """Pop return address; recover the thread from its id."""
        ptr = self.cs_ptr - 1
        assert ptr >= 0
        if USE_STACK_FRAGMENT:
            pop_ds_fragments_commit(self)
        self.cs_ptr = ptr
        tid = self.cs_tids[ptr]
        ip = self.cs_ips[ptr]
        # Clear the slot (mirrors the old null write): helps the JIT treat the
        # tail above cs_ptr as dead and elide the reads on recursive traces.
        self.cs_tids[ptr] = 0
        thread = THREAD_REGISTRY.threads[tid]
        return thread, ip

    def is_call_stack_empty(self):
        """Check if call stack is empty."""
        return self.cs_ptr == 0

    def push_loop(self, limit, counter):
        """Push loop parameters onto the dedicated loop-control stack."""
        d = self.lc_depth
        assert d < len(self.lc_is)
        if d > 0:
            self.lc_is[d - 1] = self.li
            self.lc_ls[d - 1] = self.ll
        self.li = counter
        self.ll = limit
        self.lc_depth = d + 1

    def pop_loop(self):
        """Pop loop parameters from the loop-control stack."""
        d = self.lc_depth - 1
        assert d >= 0
        limit = self.ll
        counter = self.li
        if d > 0:
            self.li = self.lc_is[d - 1]
            self.ll = self.lc_ls[d - 1]
        self.lc_depth = d
        return limit, counter

    @unroll_safe
    def peek_loop_counter(self, depth=0):
        """Get current loop counter without popping (raw int)."""
        depth = promote(depth)
        if depth == 0:
            return self.li
        return self.lc_is[self.lc_depth - 1 - depth]

    @unroll_safe
    def peek_loop_limit(self, depth=0):
        """Get current loop limit without popping (raw int)."""
        depth = promote(depth)
        if depth == 0:
            return self.ll
        return self.lc_ls[self.lc_depth - 1 - depth]

    @unroll_safe
    def set_loop_counter(self, depth, value):
        """Set loop counter in place (raw int)."""
        depth = promote(depth)
        if depth == 0:
            self.li = value
        else:
            self.lc_is[self.lc_depth - 1 - depth] = value

    def push_ds(self, w_x):
        assert isinstance(w_x, W_Object)
        ds_ptr = self.ds_ptr_locals
        self.ds_locals[ds_ptr] = w_x
        self.ds_ptr_locals = ds_ptr + 1

    def pop_ds(self):
        ds_ptr = self.ds_ptr_locals - 1
        assert ds_ptr >= 0
        w_x = self.ds_locals[ds_ptr]
        assert isinstance(w_x, W_Object)
        self.ds_locals[ds_ptr] = None
        self.ds_ptr_locals = ds_ptr
        return w_x

    def push_ds_int(self, intval):
        if USE_STACK_FRAGMENT:
            self.push_on(intval)
        else:
            self.push_ds_int_fixed(intval)

    def push_ds_int_fixed(self, intval):
        ds_ptr = self.ds_ptr_ints
        self.ds_ints[ds_ptr] = intval
        self.ds_ptr_ints = ds_ptr + 1

    def push_ds_float(self, floatval):
        ds_ptr = self.ds_ptr_floats
        self.ds_floats[ds_ptr] = floatval
        self.ds_ptr_floats = ds_ptr + 1

    def pop_ds_int(self):
        if USE_STACK_FRAGMENT:
            return self.pop_on()
        return self.pop_ds_int_fixed()

    def pop_ds_int_fixed(self):
        ds_ptr = self.ds_ptr_ints - 1
        assert ds_ptr >= 0
        intval = self.ds_ints[ds_ptr]
        self.ds_ptr_ints = ds_ptr
        return intval

    def pop_ds_float(self):
        ds_ptr = self.ds_ptr_floats - 1
        assert ds_ptr >= 0
        floatval = self.ds_floats[ds_ptr]
        self.ds_ptr_floats = ds_ptr
        return floatval

    def peek_ds_int(self, depth=0):
        if USE_STACK_FRAGMENT:
            return self.peek_on(depth)
        return self.peek_ds_int_fixed(depth)

    def peek_ds_int_fixed(self, depth=0):
        ptr = self.ds_ptr_ints - 1 - depth
        assert ptr >= 0
        return self.ds_ints[ptr]

    def peek_ds_float(self, depth=0):
        ptr = self.ds_ptr_floats - 1 - depth
        assert ptr >= 0
        return self.ds_floats[ptr]

    def peek_ds(self, depth=0):
        ptr = self.ds_ptr_locals - 1 - depth
        assert ptr >= 0
        return self.ds_locals[ptr]

    def poke_ds_int(self, depth, intval):
        if USE_STACK_FRAGMENT:
            self.poke_on(depth, intval)
        else:
            self.poke_ds_int_fixed(depth, intval)

    def poke_ds_int_fixed(self, depth, intval):
        ptr = self.ds_ptr_ints - 1 - depth
        assert ptr >= 0
        self.ds_ints[ptr] = intval

    def depth_ds_int(self):
        if USE_STACK_FRAGMENT:
            return self.depth_on()
        return self.ds_ptr_ints

    def reset_ds_int(self):
        if USE_STACK_FRAGMENT:
            reset_ds_fragments(self)
        else:
            self.ds_ptr_ints = 0

    def ds_int_size(self):
        return self.depth_ds_int()

    def clear_ds_int(self):
        self.reset_ds_int()

    def top2_ds(self):
        w_y = self.pop_ds()
        w_x = self.pop_ds()
        return w_x, w_y

    def top2_ds_int(self):
        y = self.pop_ds_int()
        x = self.pop_ds_int()
        return x, y

    def peek_rs(self, depth=0):
        ptr = self.rs_ptr - 1 - depth
        assert ptr >= 0
        return self.rs[ptr]

    def poke_rs(self, depth, w_x):
        ptr = self.rs_ptr - 1 - depth
        assert ptr >= 0
        self.rs[ptr] = w_x

    def push_rs(self, w_x):
        rs_ptr = self.rs_ptr
        self.rs[rs_ptr] = w_x
        self.rs_ptr = rs_ptr + 1

    def pop_rs(self):
        rs_ptr = self.rs_ptr - 1
        assert rs_ptr >= 0
        w_x = self.rs[rs_ptr]
        self.rs_ptr = rs_ptr
        return w_x

    def print_int(self, x):
        assert isinstance(x, W_IntObject)
        _, stdout, _ = create_stdio()
        stdout.write(x.to_string())
        stdout.flush()

    def print_str(self, s):
        assert isinstance(s, W_StringObject)
        _, stdout, _ = create_stdio()
        stdout.write(s.to_string())
        stdout.flush()

    def alloc_buf(self, content, size):
        addr = self.here
        self.buf[addr] = W_StringObject(content[:size])
        self.here += 1
        return addr

    def _get_heap(self):
        heap = self.heap
        if heap is None:
            heap = Heap(HEAP_SIZE_BYTES)
            self.heap = heap
        return heap

    def _ensure_addr(self, addr, span):
        assert 0 <= addr < HEAP_SIZE_BYTES
        assert addr + span <= HEAP_SIZE_BYTES

    def cell_store(self, addr, intval):
        assert isinstance(addr, int)
        self._get_heap().cell_store(addr, intval)

    def cell_fetch_int(self, addr):
        heap = self.heap
        if heap is None:
            return 0
        return heap.cell_fetch_int(addr)

    def cell_fetch(self, addr):
        heap = self.heap
        if heap is None:
            return ZERO
        return heap.cell_fetch(addr)

    def cell_2store(self, addr, x1_int, x2_int):
        assert 0 <= addr < HEAP_CELL_COUNT - self.cell_size_bytes
        self.cell_store(addr, x1_int)
        self.cell_store(addr + self.cell_size_bytes, x2_int)

    def cell_2fetch(self, addr):
        assert 0 <= addr < HEAP_CELL_COUNT - self.cell_size_bytes
        x1 = make_int(self.cell_fetch_int(addr))
        x2 = make_int(self.cell_fetch_int(addr + self.cell_size_bytes))
        return x1, x2

    def char_store(self, addr, intval):
        self._get_heap().char_store(addr, intval)

    def char_fetch(self, addr):
        heap = self.heap
        if heap is None:
            return 0
        return heap.char_fetch(addr)

    def float_store(self, addr, value):
        self._get_heap().float_store(addr, value)

    def cell_float_fetch(self, addr):
        heap = self.heap
        if heap is None:
            return 0.0
        return heap.float_fetch_float(addr)

    def float_fetch(self, addr):
        heap = self.heap
        if heap is None:
            return W_FloatObject(0.0)
        return heap.float_fetch(addr)

    def execute_thread(self, thread, ip=0):
        while True:
            jitdriver.jit_merge_point(ip=ip, thread=thread, self=self)
            if ip >= len(thread.code):
                if not self.is_call_stack_empty():
                    thread, ip = self.pop_call()
                    continue
                else:
                    break

            w = promote(thread.code[ip])
            if w is None:
                if not self.is_call_stack_empty():
                    thread, ip = self.pop_call()
                    continue
                else:
                    break
            ip += 1

            prim = promote(w.prim)
            if prim is not None:
                ip = prim(self, thread, ip)
                if ip == EXIT_SENTINEL:
                    if not self.is_call_stack_empty():
                        thread, ip = self.pop_call()
                    else:
                        break
                    continue
                if ip == TAILCALL_SENTINEL:
                    from rpyforth.objects import W_WordObject
                    tailcall_idx = len(thread.code) - 1
                    target = promote(thread.lits[tailcall_idx])
                    if isinstance(target, W_WordObject):
                        target_word = promote(target.word)
                        nested_thread = promote(target_word.thread)
                        if nested_thread is not None:
                            thread = nested_thread
                            ip = 0
                            jitdriver.can_enter_jit(ip=ip, thread=thread, self=self)
                            continue
                    if not self.is_call_stack_empty():
                        thread, ip = self.pop_call()
                    else:
                        break
                    continue
            else:
                nested_thread = promote(w.thread)
                self.push_call(thread, ip)
                if USE_STACK_FRAGMENT:
                    push_ds_fragments(self)
                thread = nested_thread
                ip = 0
                jitdriver.can_enter_jit(ip=ip, thread=thread, self=self)

    def execute_word_now(self, w):
        code = [w]
        lits = [ZERO]
        self.execute_thread(CodeThread(code, lits), 0)
