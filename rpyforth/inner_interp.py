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
)


import os

from rpython.rlib.jit import JitDriver, promote, elidable, unroll_safe, promote_string, hint
from rpython.rlib.rfile import create_stdio

from rpyforth.heap import HEAP_CELL_COUNT, HEAP_SIZE_BYTES, Heap

DONT_USE_VIRTUALIZATION = bool(os.environ.get("RPYFORTH_NO_VIRTUALIZE"))

USE_STACK_FRAGMENT = bool(os.environ.get("RPYFORTH_STACK_FRAGMENT"))

from rpyforth.metastack import DSIntFragment

TOP_CACHE_SIZE = 4
CALL_WINDOW = 8

STACK_SIZE = 16384 # Increased for deeper nesting (ack(3,10))
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

if DONT_USE_VIRTUALIZATION:
    jitdriver = JitDriver(
        greens=['ip', 'thread'],
        reds=['self'],
        get_printable_location=get_printable_location
    )
else:
    jitdriver = JitDriver(
        greens=['ip', 'thread'],
        reds=['self'],
        virtualizables=['self'],
        get_printable_location=get_printable_location
    )

class InnerInterpreter(object):
    _immutable_fields_ = ["cell_size", "cell_size_bytes", "base"]

    if DONT_USE_VIRTUALIZATION:
        _virtualizable_ = []
    elif USE_STACK_FRAGMENT:
        _virtualizable_ = ['top0', 'top1', 'top2', 'top3', 'top_count']
    else:
        _virtualizable_ = ["ds_ints", "ds_floats", "ds_locals",
                           "ds_ptr_ints", "ds_ptr_floats", "ds_ptr_locals",
                           "rs", "rs_ptr", "cs_threads",  "cs_ips", "cs_ptr"]


    def __init__(self):
        # Reference to outer interpreter (set later)
        self.outer = None

        # Pre-allocate larger stacks to reduce growth overhead
        self.ds_ints = [0] * STACK_SIZE # unboxed integer data stack
        self.ds_ptr_ints = 0

        self.ds_floats = [0.0] * STACK_SIZE
        self.ds_ptr_floats = 0

        self.ds_locals = [None] * STACK_SIZE
        self.ds_ptr_locals = 0

        self.rs = [0] * STACK_SIZE  # return stack
        self.rs_ptr = 0

        # Virtualized call stack for JIT optimization
        self.cs_threads = [None] * STACK_SIZE  # return threads
        self.cs_ips = [0] * STACK_SIZE       # return IPs
        self.cs_ptr = 0

        self.heap = None
        self.cell_size = CELL_SIZE
        self.cell_size_bytes = CELL_SIZE_BYTES

        # for string
        self.buf = [None] * HEAP_SIZE_BYTES
        self.here = 0

        self.base = 10                # DECIMAL
        self._pno_active = False      # inside <# ... #> or not
        self._pno_buf = []            # buffer for pno (pictured numeric output)
        self.argv = []                # command-line arguments (set by target)

        self.top0 = 0
        self.top1 = 0
        self.top2 = 0
        self.top3 = 0
        self.top_count = 0

        self.ds_int_current = DSIntFragment(None)


    def push_call(self, thread, ip):
        """Push return address onto virtualized call stack."""
        ptr = self.cs_ptr
        assert ptr < len(self.cs_threads)
        self.cs_threads[ptr] = thread
        self.cs_ips[ptr] = ip
        self.cs_ptr = ptr + 1

    def pop_call(self):
        """Pop return address from virtualized call stack."""
        ptr = self.cs_ptr - 1
        assert ptr >= 0
        self.cs_ptr = ptr
        thread = self.cs_threads[ptr]
        ip = self.cs_ips[ptr]
        self.cs_threads[ptr] = None
        return thread, ip

    def is_call_stack_empty(self):
        """Check if call stack is empty."""
        return self.cs_ptr == 0

    def push_loop(self, limit, counter):
        """Push loop parameters onto return stack (limit first, then counter on top)."""
        self.push_rs(limit)
        self.push_rs(counter)

    def pop_loop(self):
        """Pop loop parameters from return stack."""
        counter = self.pop_rs()
        limit = self.pop_rs()
        return limit, counter

    @unroll_safe
    def peek_loop_counter(self, depth=0):
        """Get current loop counter without popping (raw int)."""
        # Counter is at top of each loop frame (2 cells per loop)
        return self.peek_rs(promote(depth) * 2)

    @unroll_safe
    def peek_loop_limit(self, depth=0):
        """Get current loop limit without popping (raw int)."""
        # Limit is below counter in each loop frame (2 cells per loop)
        return self.peek_rs(promote(depth) * 2 + 1)

    @unroll_safe
    def set_loop_counter(self, depth, value):
        """Set loop counter in place (raw int)."""
        # Counter is at top of each loop frame (2 cells per loop)
        self.poke_rs(promote(depth) * 2, value)

    def push_ds(self, w_x):
        assert isinstance(w_x, W_Object)
        ds_ptr = self.ds_ptr_locals
        self.ds_locals[ds_ptr] = w_x
        self.ds_ptr_locals = ds_ptr + 1

    def pop_ds(self):
        ds_ptr = self.ds_ptr_locals - 1
        assert ds_ptr >= 0
        # Return boxed object
        w_x = self.ds_locals[ds_ptr]
        assert isinstance(w_x, W_Object)
        self.ds_locals[ds_ptr] = None
        self.ds_ptr_locals = ds_ptr
        return w_x

    def push_ds_int(self, intval):
        if USE_STACK_FRAGMENT:
            self.push_ds_int_frag(intval)
        else:
            self.push_ds_int_fixed(intval)

    def push_ds_int_frag(self, intval):
        tc = self.top_count
        if tc < TOP_CACHE_SIZE:
            self.top3 = self.top2
            self.top2 = self.top1
            self.top0 = intval
            self.top_count = tc + 1
        else:
            # spill
            self.ds_int_current.push(self.top3)
            self.top3 = self.top2
            self.top2 = self.top1
            self.top1 = self.top0
            self.top0 = intval

    def push_ds_int_fixed(self, intval):
        ds_ptr = self.ds_ptr_ints
        self.ds_ints[ds_ptr] = intval
        #self.ds_tags[ds_ptr] = 0
        self.ds_ptr_ints = ds_ptr + 1

    def push_ds_float(self, floatval):
        ds_ptr = self.ds_ptr_floats
        self.ds_floats[ds_ptr] = floatval
        #self.ds_tags[ds_ptr] = 1
        self.ds_ptr_floats = ds_ptr + 1

    def pop_ds_int(self):
        if USE_STACK_FRAGMENT:
            return self.pop_ds_int_frag()
        else:
            return self.pop_ds_int_fixed()

    def pop_ds_int_frag(self):
        tc = self.top_count
        if tc < TOP_CACHE_SIZE:
            top2 = self.top2; top1 = self.top1; top0 = self.top0
            self.top3 = 0 # initialize
            self.top2 = self.top3
            self.top1 = top2
            self.top0 = top1
            self.top_count = tc - 1
            return top0
        else:
            # spill
            old_top3 = self.top3; old_top2 = self.top2
            old_top1 = self.top1; old_top0 = self.top0

            old_ds_top = self.ds_int_current.pop()
            self.top3 = old_ds_top
            self.top2 = old_top3
            self.top1 = old_top2
            self.top0 = old_top1
            return old_top0

    def pop_ds_int_fixed(self):
        ds_ptr = self.ds_ptr_ints - 1
        assert ds_ptr >= 0
        #assert self.ds_tags[ds_ptr] == 0, "Expected int on stack"
        intval = self.ds_ints[ds_ptr]
        self.ds_ptr_ints = ds_ptr
        return intval

    def pop_ds_float(self):
        ds_ptr = self.ds_ptr_floats - 1
        assert ds_ptr >= 0
        #assert self.ds_tags[ds_ptr] == 1, "Expected float on stack"
        floatval = self.ds_floats[ds_ptr]
        self.ds_ptr_floats = ds_ptr
        return floatval

    def peek_ds_int(self, depth=0):
        ptr = self.ds_ptr_ints - 1 - depth
        assert ptr >= 0
        #assert self.ds_tags[ptr] == 0, "Expected int on stack"
        return self.ds_ints[ptr]

    def peek_ds_float(self, depth=0):
        ptr = self.ds_ptr_floats - 1 - depth
        assert ptr >= 0
        #assert self.ds_tags[ptr] == 1, "Expected float on stack"
        return self.ds_floats[ptr]

    def peek_ds(self, depth=0):
        """Peek at stack value without popping (returns boxed W_Object)."""
        ptr = self.ds_ptr_locals - 1 - depth
        assert ptr >= 0
        return self.ds_locals[ptr]

    def poke_ds_int(self, depth, intval):
        """Set raw integer at depth (unboxed)."""
        ptr = self.ds_ptr_ints - 1 - depth
        assert ptr >= 0
        self.ds_ints[ptr] = intval

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
        """! ( x addr -- ) - Store value at cell address (unboxed)."""
        assert isinstance(addr, int)
        self._get_heap().cell_store(addr, intval)

    def cell_fetch_int(self, addr):
        """Fetch raw integer from cell address (0 if never stored)."""
        heap = self.heap
        if heap is None:
            return 0
        return heap.cell_fetch_int(addr)

    def cell_fetch(self, addr):
        """@ ( addr -- x ) - Fetch value as W_IntObject for outer interpreter."""
        heap = self.heap
        if heap is None:
            return ZERO
        return heap.cell_fetch(addr)

    def cell_2store(self, addr, x1_int, x2_int):
        """2! ( x1 x2 addr -- ) - Store cell pair."""
        assert 0 <= addr < HEAP_CELL_COUNT - self.cell_size_bytes
        self.cell_store(addr, x1_int)
        self.cell_store(addr + self.cell_size_bytes, x2_int)

    def cell_2fetch(self, addr):
        """2@ ( addr -- x1 x2 ) - Fetch cell pair."""
        assert 0 <= addr < HEAP_CELL_COUNT - self.cell_size_bytes
        x1 = make_int(self.cell_fetch_int(addr))
        x2 = make_int(self.cell_fetch_int(addr + self.cell_size_bytes))
        return x1, x2

    def char_store(self, addr, intval):
        """C! ( char c-addr -- ) - Store character (unboxed, no allocation)."""
        self._get_heap().char_store(addr, intval)

    def char_fetch(self, addr):
        """C@ ( c-addr -- char ) - Fetch character (unboxed, no allocation)."""
        heap = self.heap
        if heap is None:
            return 0
        return heap.char_fetch(addr)

    def float_store(self, addr, value):
        """F! ( addr -- ) ( F: f -- ) - Store float (unboxed)."""
        self._get_heap().float_store(addr, value)

    def cell_float_fetch(self, addr):
        """Fetch raw float from address (0.0 if never stored)."""
        heap = self.heap
        if heap is None:
            return 0.0
        return heap.float_fetch_float(addr)

    def float_fetch(self, addr):
        """F@ ( addr -- ) ( F: -- f ) - Fetch float as W_FloatObject."""
        heap = self.heap
        if heap is None:
            return W_FloatObject(0.0)
        return heap.float_fetch(addr)

    def execute_thread(self, thread, ip=0):
        while True:
            jitdriver.jit_merge_point(
                ip=ip,
                thread=thread,
                self=self
            )
            if ip >= len(thread.code):
                if not self.is_call_stack_empty():
                    thread, ip = self.pop_call()
                    continue
                else:
                    break

            # Promote the word to allow JIT to specialize on it
            w = promote(thread.code[ip])
            if w is None:
                if not self.is_call_stack_empty():
                    thread, ip = self.pop_call()
                    continue
                else:
                    break
            ip += 1

            # Promote the primitive function pointer for better inlining
            prim = promote(w.prim)
            if prim is not None:
                ip = prim(self, thread, ip)
                # Check for EXIT sentinel (faster than exception)
                if ip == EXIT_SENTINEL:
                    if not self.is_call_stack_empty():
                        thread, ip = self.pop_call()
                    else:
                        break
                    continue
                # Check for TAILCALL sentinel - perform tail call optimization
                if ip == TAILCALL_SENTINEL:
                    # TAILCALL was the last instruction (at code[len(code)-1])
                    # The target word is stored in lits[len(code)-1]
                    from rpyforth.objects import W_WordObject
                    tailcall_idx = len(thread.code) - 1
                    target = promote(thread.lits[tailcall_idx])
                    if isinstance(target, W_WordObject):
                        target_word = promote(target.word)
                        nested_thread = promote(target_word.thread)
                        if nested_thread is not None:
                            # Just replace thread
                            thread = nested_thread
                            ip = 0
                            jitdriver.can_enter_jit(
                                ip=ip,
                                thread=thread,
                                self=self
                            )
                            continue
                    if not self.is_call_stack_empty():
                        # Fall back: just pop call stack
                        thread, ip = self.pop_call()
                    else:
                        break
                    continue
            else:
                # Colon definition: push current state and enter nested thread
                nested_thread = promote(w.thread)
                self.push_call(thread, ip)
                thread = nested_thread
                ip = 0
                jitdriver.can_enter_jit(
                    ip=ip,
                    thread=thread,
                    self=self
                )

    def execute_word_now(self, w):
        code = [w]
        lits = [ZERO]
        self.execute_thread(CodeThread(code, lits), 0)
