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
)


from rpython.rlib.rstruct.ieee import float_pack, float_unpack
from rpython.rlib.rarithmetic import r_ulonglong
from rpython.rlib.jit import JitDriver, promote, elidable, unroll_safe, promote_string, hint
from rpython.rlib.rfile import create_stdio

STACK_SIZE = 4096 # Increased for deeper nesting
BUF_SIZE = 1024
HEAP_CELL_COUNT = 65536
HEAP_SIZE_BYTES = HEAP_CELL_COUNT

# Sentinel value for EXIT - indicates return from current definition
EXIT_SENTINEL = -1

class Exit(Exception):
    pass

class Bye(Exception):
    """Raised by BYE to exit the Forth system cleanly."""
    pass

def get_printable_location(ip, thread):
    return "ip=%d %s %s" % (ip, thread.code[ip].to_string(), thread.lits[ip].to_string())

jitdriver = JitDriver(
    greens=['ip', 'thread'],
    reds=['self'],
    virtualizables=['self'],
    get_printable_location=get_printable_location
)

class InnerInterpreter(object):
    _immutable_fields_ = ["cell_size", "cell_size_bytes", "base"]
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

        self.mem = [None] * HEAP_SIZE_BYTES
        self.cell_size = CELL_SIZE
        self.cell_size_bytes = CELL_SIZE_BYTES

        # for string
        self.buf = [None] * HEAP_SIZE_BYTES
        self.here = 0

        self.base = 10                # DECIMAL
        self._pno_active = False      # inside <# ... #> or not
        self._pno_buf = []            # buffer for pno (pictured numeric output)

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

    def _ensure_addr(self, addr, span):
        assert 0 <= addr < len(self.mem)
        assert addr + span <= len(self.mem)

    def cell_store(self, addr, intval):
        """! ( x addr -- ) - Store value at cell address."""
        assert isinstance(addr, int)
        assert 0 <= addr < HEAP_CELL_COUNT
        value_obj = W_IntObject(intval)
        self.mem[addr] = value_obj

    def cell_fetch(self, addr):
        """@ ( addr -- x ) - Fetch value from cell address."""
        assert 0 <= addr < HEAP_CELL_COUNT
        result = self.mem[addr]
        if result is None:
            return ZERO
        return result

    def cell_2store(self, addr, x1_int, x2_int):
        """2! ( x1 x2 addr -- ) - Store cell pair."""
        assert 0 <= addr < HEAP_CELL_COUNT - self.cell_size_bytes
        self.mem[addr] = W_IntObject(x1_int)
        self.mem[addr + self.cell_size_bytes] = W_IntObject(x2_int)

    def cell_2fetch(self, addr):
        """2@ ( addr -- x1 x2 ) - Fetch cell pair."""
        assert 0 <= addr < HEAP_CELL_COUNT - self.cell_size_bytes
        x1 = self.mem[addr]
        x2 = self.mem[addr + self.cell_size_bytes]
        if x1 is None:
            x1 = ZERO
        if x2 is None:
            x2 = ZERO
        return x1, x2

    def float_store(self, addr, value):
        """F! ( addr -- ) ( F: f -- ) - Store float."""
        assert 0 <= addr < HEAP_CELL_COUNT
        self.mem[addr] = W_FloatObject(value)

    def float_fetch(self, addr):
        """F@ ( addr -- ) ( F: -- f ) - Fetch float."""
        assert 0 <= addr < HEAP_CELL_COUNT
        result = self.mem[addr]
        if result is None:
            return W_FloatObject(0.0)
        return result

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
            else:
                # Colon definition - push current state and enter nested thread
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
