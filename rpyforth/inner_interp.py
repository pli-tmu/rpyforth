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
from rpython.rlib.rarithmetic import r_ulonglong, intmask
from rpython.rlib.jit import JitDriver, promote, elidable, unroll_safe, promote_string
from rpython.rlib.rfile import create_stdio

STACK_SIZE = 64 # Increased for deeper nesting
BUF_SIZE = 1024
HEAP_CELL_COUNT = 65536
HEAP_SIZE_BYTES = HEAP_CELL_COUNT


class Exit(Exception):
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
    _virtualizable_ = ["ds_ptr", "ds_ints[*]", "ds_tags[*]", "ds[*]", "rs_ptr", "rs[*]",
                       "cs_ptr", "cs_threads[*]", "cs_ips[*]",
                       "loop_ptr", "loop_counters[*]", "loop_limits[*]"]


    def __init__(self):
        # Pre-allocate larger stacks to reduce growth overhead
        self.ds_ints = [0] * STACK_SIZE # unboxed integer data stack
        self.ds_tags = [0] * STACK_SIZE # 0=int, 1=object
        self.ds = [None] * STACK_SIZE # boxed object data stack (for non-ints)
        self.ds_ptr = 0

        self.rs = [None] * STACK_SIZE  # return stack
        self.rs_ptr = 0

        # Virtualized call stack for JIT optimization
        self.cs_threads = [None] * STACK_SIZE  # return threads
        self.cs_ips = [0] * STACK_SIZE        # return IPs
        self.cs_ptr = 0

        # Dedicated integer loop stack for DO...LOOP (avoids W_IntObject allocation)
        self.loop_counters = [0] * 32  # raw int counters
        self.loop_limits = [0] * 32    # raw int limits
        self.loop_ptr = 0

        self.mem = [None] * HEAP_SIZE_BYTES
        self.cell_size = CELL_SIZE
        self.cell_size_bytes = CELL_SIZE_BYTES

        # for string
        self.buf = [None] * HEAP_SIZE_BYTES
        self.here = 0

        self.base = DECIMAL
        self._pno_active = False      # inside <# ... #> or not
        self._pno_buf = []            # buffer for pno (pictured numeric output)

    def push_call(self, thread, ip):
        """Push return address onto virtualized call stack."""
        ptr = self.cs_ptr
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
        """Push loop parameters (raw integers) onto dedicated loop stack."""
        ptr = self.loop_ptr
        self.loop_limits[ptr] = limit
        self.loop_counters[ptr] = counter
        self.loop_ptr = ptr + 1

    def pop_loop(self):
        """Pop loop parameters from dedicated loop stack."""
        ptr = self.loop_ptr - 1
        assert ptr >= 0
        self.loop_ptr = ptr
        limit = self.loop_limits[ptr]
        counter = self.loop_counters[ptr]
        return limit, counter

    def peek_loop_counter(self, depth=0):
        """Get current loop counter without popping (raw int)."""
        ptr = self.loop_ptr - 1 - depth
        assert ptr >= 0
        return self.loop_counters[ptr]

    def peek_loop_limit(self, depth=0):
        """Get current loop limit without popping (raw int)."""
        ptr = self.loop_ptr - 1 - depth
        assert ptr >= 0
        return self.loop_limits[ptr]

    def set_loop_counter(self, depth, value):
        """Set loop counter in place (raw int)."""
        ptr = self.loop_ptr - 1 - depth
        assert ptr >= 0
        self.loop_counters[ptr] = value

    def push_ds(self, w_x):
        ds_ptr = self.ds_ptr
        if isinstance(w_x, W_IntObject):
            # Store unboxed integer
            self.ds_ints[ds_ptr] = w_x.intval
            self.ds_tags[ds_ptr] = 0
        else:
            # Store boxed object
            self.ds[ds_ptr] = w_x
            self.ds_tags[ds_ptr] = 1
        self.ds_ptr = ds_ptr + 1

    def pop_ds(self):
        ds_ptr = self.ds_ptr - 1
        assert ds_ptr >= 0
        tag = self.ds_tags[ds_ptr]
        if tag == 0:
            # Reconstruct W_IntObject from unboxed int
            intval = self.ds_ints[ds_ptr]
            self.ds_ptr = ds_ptr
            return W_IntObject(intval)
        else:
            # Return boxed object
            w_x = self.ds[ds_ptr]
            self.ds[ds_ptr] = None
            self.ds_ptr = ds_ptr
            return w_x

    def push_ds_int(self, intval):
        """Push raw integer directly (unboxed)."""
        ds_ptr = self.ds_ptr
        self.ds_ints[ds_ptr] = intval
        self.ds_tags[ds_ptr] = 0
        self.ds_ptr = ds_ptr + 1

    def pop_ds_int(self):
        """Pop and return raw integer (unboxed)."""
        ds_ptr = self.ds_ptr - 1
        assert ds_ptr >= 0
        assert self.ds_tags[ds_ptr] == 0, "Expected int on stack"
        intval = self.ds_ints[ds_ptr]
        self.ds_ptr = ds_ptr
        return intval

    def peek_ds_int(self, depth=0):
        """Peek at raw integer without popping (unboxed)."""
        ptr = self.ds_ptr - 1 - depth
        assert ptr >= 0
        assert self.ds_tags[ptr] == 0, "Expected int on stack"
        return self.ds_ints[ptr]

    def peek_ds(self, depth=0):
        """Peek at stack value without popping (returns boxed W_Object)."""
        ptr = self.ds_ptr - 1 - depth
        assert ptr >= 0
        tag = self.ds_tags[ptr]
        if tag == 0:
            return W_IntObject(self.ds_ints[ptr])
        else:
            return self.ds[ptr]

    def poke_ds_int(self, depth, intval):
        """Set raw integer at depth (unboxed)."""
        ptr = self.ds_ptr - 1 - depth
        assert ptr >= 0
        self.ds_ints[ptr] = intval
        self.ds_tags[ptr] = 0

    def top2_ds(self):
        w_y = self.pop_ds()
        w_x = self.pop_ds()
        return w_x, w_y

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
        self.rs[rs_ptr] = None
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
        return W_PtrObject(addr)

    def _ensure_addr(self, addr, span):
        assert 0 <= addr < len(self.mem)
        assert addr + span <= len(self.mem)

    def cell_store(self, addr_obj, value_obj):
        """! ( x addr -- ) - Store value at cell address."""
        assert isinstance(addr_obj, W_IntObject)
        addr = addr_obj.intval
        assert 0 <= addr < HEAP_CELL_COUNT
        self.mem[addr] = value_obj

    def cell_fetch(self, addr_obj):
        """@ ( addr -- x ) - Fetch value from cell address."""
        assert isinstance(addr_obj, W_IntObject)
        addr = addr_obj.intval
        assert 0 <= addr < HEAP_CELL_COUNT
        result = self.mem[addr]
        if result is None:
            return ZERO
        return result

    def cell_2store(self, addr_obj, x1_obj, x2_obj):
        """2! ( x1 x2 addr -- ) - Store cell pair."""
        assert isinstance(addr_obj, W_IntObject)
        addr = addr_obj.intval
        assert 0 <= addr < HEAP_CELL_COUNT - self.cell_size_bytes
        self.mem[addr] = x1_obj
        self.mem[addr + self.cell_size_bytes] = x2_obj

    def cell_2fetch(self, addr_obj):
        """2@ ( addr -- x1 x2 ) - Fetch cell pair."""
        assert isinstance(addr_obj, W_IntObject)
        addr = addr_obj.intval
        assert 0 <= addr < HEAP_CELL_COUNT - self.cell_size_bytes
        x1 = self.mem[addr]
        x2 = self.mem[addr + self.cell_size_bytes]
        if x1 is None:
            x1 = ZERO
        if x2 is None:
            x2 = ZERO
        return x1, x2

    def float_store(self, addr_obj, value_obj):
        """F! ( addr -- ) ( F: f -- ) - Store float."""
        assert isinstance(addr_obj, W_IntObject)
        assert isinstance(value_obj, W_FloatObject)
        addr = addr_obj.intval
        assert 0 <= addr < HEAP_CELL_COUNT
        self.mem[addr] = value_obj

    def float_fetch(self, addr_obj):
        """F@ ( addr -- ) ( F: -- f ) - Fetch float."""
        assert isinstance(addr_obj, W_IntObject)
        addr = addr_obj.intval
        assert 0 <= addr < HEAP_CELL_COUNT
        result = self.mem[addr]
        if result is None:
            return W_FloatObject(0.0)
        assert isinstance(result, W_FloatObject)
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
                try:
                    ip = prim(self, thread, ip)
                except Exit:
                    # EXIT - return to caller
                    if not self.is_call_stack_empty():
                        thread, ip = self.pop_call()
                    else:
                        break
            else:
                # Colon definition - push current state and enter nested thread
                nested_thread = promote(w.thread)
                self.push_call(thread, ip)
                thread = nested_thread
                ip = 0
                # Signal JIT that this is a potential loop entry point (for recursion)
                jitdriver.can_enter_jit(
                    ip=ip,
                    thread=thread,
                    self=self
                )

    def execute_word_now(self, w):
        code = [w]
        lits = [ZERO]
        self.execute_thread(CodeThread(code, lits), 0)
