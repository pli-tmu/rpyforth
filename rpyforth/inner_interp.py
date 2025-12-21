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

# Window size for virtualized stacks - small enough for full JIT virtualization
WINDOW_SIZE = 8
# Backing store size
BACKING_SIZE = 4096
# Total stack size for non-virtualized stacks
STACK_SIZE = 4096

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
    # Note: virtualizables are not used because RPython's strict requirements
    # for virtualizable arrays (constant indices, no complex iteration) conflict
    # with the spill/fill operations. The window architecture still provides:
    # 1. Better cache locality (8-element window vs 4096-element array)
    # 2. Hot path stays in small window most of the time
    # 3. @unroll_safe on spill/fill allows JIT optimization
    get_printable_location=get_printable_location
)

class InnerInterpreter(object):
    _immutable_fields_ = ["cell_size", "cell_size_bytes", "base"]
    # Window stacks provide cache benefits without JIT virtualization

    def __init__(self):
        # Reference to outer interpreter (set later)
        self.outer = None

        # Integer data stack - virtualized window + backing store
        self.int_window = [0] * WINDOW_SIZE
        self.int_window_ptr = 0
        self.int_backing = [0] * BACKING_SIZE
        self.int_backing_ptr = 0

        # Float data stack - not virtualized (less common in hot paths)
        self.float_stack = [0.0] * STACK_SIZE
        self.float_ptr = 0

        # Return stack - virtualized window + backing store
        self.rs_window = [0] * WINDOW_SIZE
        self.rs_window_ptr = 0
        self.rs_backing = [0] * BACKING_SIZE
        self.rs_backing_ptr = 0

        # Call stack - virtualized window + backing store
        self.cs_window_threads = [None] * WINDOW_SIZE
        self.cs_window_ips = [0] * WINDOW_SIZE
        self.cs_window_ptr = 0
        self.cs_backing_threads = [None] * BACKING_SIZE
        self.cs_backing_ips = [0] * BACKING_SIZE
        self.cs_backing_ptr = 0

        # Boxed object stack (for W_WordObject, etc.) - not virtualized
        self.ds_locals = [None] * STACK_SIZE
        self.ds_ptr_locals = 0

        self.mem = [None] * HEAP_SIZE_BYTES
        self.cell_size = CELL_SIZE
        self.cell_size_bytes = CELL_SIZE_BYTES

        # for string
        self.buf = [None] * HEAP_SIZE_BYTES
        self.here = 0

        self.base = 10                # DECIMAL
        self._pno_active = False      # inside <# ... #> or not
        self._pno_buf = []            # buffer for pno (pictured numeric output)

    # ==================== Call Stack Operations ====================

    @unroll_safe
    def _cs_spill_to_backing(self):
        """Move bottom half of call stack window to backing store."""
        spill_count = WINDOW_SIZE // 2
        for i in range(spill_count):
            self.cs_backing_threads[self.cs_backing_ptr + i] = self.cs_window_threads[i]
            self.cs_backing_ips[self.cs_backing_ptr + i] = self.cs_window_ips[i]
        self.cs_backing_ptr += spill_count
        for i in range(spill_count, WINDOW_SIZE):
            self.cs_window_threads[i - spill_count] = self.cs_window_threads[i]
            self.cs_window_ips[i - spill_count] = self.cs_window_ips[i]
        self.cs_window_ptr -= spill_count

    @unroll_safe
    def _cs_fill_from_backing(self):
        """Fill call stack window from backing store."""
        if self.cs_backing_ptr == 0:
            return
        fill_count = min(self.cs_backing_ptr, WINDOW_SIZE // 2)
        for i in range(self.cs_window_ptr - 1, -1, -1):
            self.cs_window_threads[i + fill_count] = self.cs_window_threads[i]
            self.cs_window_ips[i + fill_count] = self.cs_window_ips[i]
        for i in range(fill_count):
            self.cs_window_threads[i] = self.cs_backing_threads[self.cs_backing_ptr - fill_count + i]
            self.cs_window_ips[i] = self.cs_backing_ips[self.cs_backing_ptr - fill_count + i]
        self.cs_backing_ptr -= fill_count
        self.cs_window_ptr += fill_count

    def push_call(self, thread, ip):
        """Push return address onto virtualized call stack."""
        ptr = self.cs_window_ptr
        if ptr >= WINDOW_SIZE:
            self._cs_spill_to_backing()
            ptr = self.cs_window_ptr
        self.cs_window_threads[ptr] = thread
        self.cs_window_ips[ptr] = ip
        self.cs_window_ptr = ptr + 1

    def pop_call(self):
        """Pop return address from virtualized call stack."""
        ptr = self.cs_window_ptr - 1
        if ptr < 0:
            self._cs_fill_from_backing()
            ptr = self.cs_window_ptr - 1
        if ptr < 0:
            raise IndexError("Call stack underflow")
        thread = self.cs_window_threads[ptr]
        ip = self.cs_window_ips[ptr]
        self.cs_window_threads[ptr] = None
        self.cs_window_ptr = ptr
        return thread, ip

    def is_call_stack_empty(self):
        """Check if call stack is empty."""
        return self.cs_window_ptr == 0 and self.cs_backing_ptr == 0

    # ==================== Integer Stack Operations ====================

    @unroll_safe
    def _int_spill_to_backing(self):
        """Move bottom half of int window to backing store."""
        spill_count = WINDOW_SIZE // 2
        for i in range(spill_count):
            self.int_backing[self.int_backing_ptr + i] = self.int_window[i]
        self.int_backing_ptr += spill_count
        for i in range(spill_count, WINDOW_SIZE):
            self.int_window[i - spill_count] = self.int_window[i]
        self.int_window_ptr -= spill_count

    @unroll_safe
    def _int_fill_from_backing(self):
        """Fill int window from backing store."""
        if self.int_backing_ptr == 0:
            return
        fill_count = min(self.int_backing_ptr, WINDOW_SIZE // 2)
        for i in range(self.int_window_ptr - 1, -1, -1):
            self.int_window[i + fill_count] = self.int_window[i]
        for i in range(fill_count):
            self.int_window[i] = self.int_backing[self.int_backing_ptr - fill_count + i]
        self.int_backing_ptr -= fill_count
        self.int_window_ptr += fill_count

    def push_ds_int(self, intval):
        """Push integer onto data stack."""
        ptr = self.int_window_ptr
        if ptr >= WINDOW_SIZE:
            self._int_spill_to_backing()
            ptr = self.int_window_ptr
        self.int_window[ptr] = intval
        self.int_window_ptr = ptr + 1

    def pop_ds_int(self):
        """Pop integer from data stack."""
        ptr = self.int_window_ptr - 1
        if ptr < 0:
            self._int_fill_from_backing()
            ptr = self.int_window_ptr - 1
        if ptr < 0:
            raise IndexError("Integer stack underflow")
        value = self.int_window[ptr]
        self.int_window_ptr = ptr
        return value

    def peek_ds_int(self, depth=0):
        """Peek at integer on data stack."""
        depth = promote(depth)
        ptr = self.int_window_ptr - 1 - depth
        if ptr >= 0:
            return self.int_window[ptr]
        else:
            backing_idx = self.int_backing_ptr + ptr
            if backing_idx < 0:
                raise IndexError("Integer stack underflow in peek")
            return self.int_backing[backing_idx]

    def poke_ds_int(self, depth, intval):
        """Set raw integer at depth (unboxed)."""
        depth = promote(depth)
        ptr = self.int_window_ptr - 1 - depth
        if ptr >= 0:
            self.int_window[ptr] = intval
        else:
            backing_idx = self.int_backing_ptr + ptr
            if backing_idx < 0:
                raise IndexError("Integer stack underflow in poke")
            self.int_backing[backing_idx] = intval

    @unroll_safe
    def top2_ds_int(self):
        """Pop two integers from data stack, returns (second, top)."""
        ptr = self.int_window_ptr
        if ptr >= 2:
            y = self.int_window[ptr - 1]
            x = self.int_window[ptr - 2]
            self.int_window_ptr = ptr - 2
            return x, y
        else:
            y = self.pop_ds_int()
            x = self.pop_ds_int()
            return x, y

    # ==================== Float Stack Operations ====================

    def push_ds_float(self, floatval):
        """Push float onto float stack."""
        ptr = self.float_ptr
        self.float_stack[ptr] = floatval
        self.float_ptr = ptr + 1

    def pop_ds_float(self):
        """Pop float from float stack."""
        ptr = self.float_ptr - 1
        assert ptr >= 0
        floatval = self.float_stack[ptr]
        self.float_ptr = ptr
        return floatval

    def peek_ds_float(self, depth=0):
        """Peek at float on float stack."""
        ptr = self.float_ptr - 1 - depth
        assert ptr >= 0
        return self.float_stack[ptr]

    # ==================== Return Stack Operations ====================

    @unroll_safe
    def _rs_spill_to_backing(self):
        """Move bottom half of RS window to backing store."""
        spill_count = WINDOW_SIZE // 2
        for i in range(spill_count):
            self.rs_backing[self.rs_backing_ptr + i] = self.rs_window[i]
        self.rs_backing_ptr += spill_count
        for i in range(spill_count, WINDOW_SIZE):
            self.rs_window[i - spill_count] = self.rs_window[i]
        self.rs_window_ptr -= spill_count

    @unroll_safe
    def _rs_fill_from_backing(self):
        """Fill RS window from backing store."""
        if self.rs_backing_ptr == 0:
            return
        fill_count = min(self.rs_backing_ptr, WINDOW_SIZE // 2)
        for i in range(self.rs_window_ptr - 1, -1, -1):
            self.rs_window[i + fill_count] = self.rs_window[i]
        for i in range(fill_count):
            self.rs_window[i] = self.rs_backing[self.rs_backing_ptr - fill_count + i]
        self.rs_backing_ptr -= fill_count
        self.rs_window_ptr += fill_count

    def push_rs(self, value):
        """Push value onto return stack."""
        ptr = self.rs_window_ptr
        if ptr >= WINDOW_SIZE:
            self._rs_spill_to_backing()
            ptr = self.rs_window_ptr
        self.rs_window[ptr] = value
        self.rs_window_ptr = ptr + 1

    def pop_rs(self):
        """Pop value from return stack."""
        ptr = self.rs_window_ptr - 1
        if ptr < 0:
            self._rs_fill_from_backing()
            ptr = self.rs_window_ptr - 1
        if ptr < 0:
            raise IndexError("Return stack underflow")
        value = self.rs_window[ptr]
        self.rs_window_ptr = ptr
        return value

    def peek_rs(self, depth=0):
        """Peek at return stack element without removing."""
        depth = promote(depth)
        ptr = self.rs_window_ptr - 1 - depth
        if ptr >= 0:
            return self.rs_window[ptr]
        else:
            backing_idx = self.rs_backing_ptr + ptr
            if backing_idx < 0:
                raise IndexError("Return stack underflow in peek")
            return self.rs_backing[backing_idx]

    def poke_rs(self, depth, value):
        """Set return stack element at depth."""
        depth = promote(depth)
        ptr = self.rs_window_ptr - 1 - depth
        if ptr >= 0:
            self.rs_window[ptr] = value
        else:
            backing_idx = self.rs_backing_ptr + ptr
            if backing_idx < 0:
                raise IndexError("Return stack underflow in poke")
            self.rs_backing[backing_idx] = value

    # ==================== Loop Operations ====================

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
        depth = promote(depth)
        return self.peek_rs(depth * 2)

    @unroll_safe
    def peek_loop_limit(self, depth=0):
        """Get current loop limit without popping (raw int)."""
        depth = promote(depth)
        return self.peek_rs(depth * 2 + 1)

    @unroll_safe
    def set_loop_counter(self, depth, value):
        """Set loop counter in place (raw int)."""
        depth = promote(depth)
        self.poke_rs(depth * 2, value)

    # ==================== Boxed Object Stack ====================

    def push_ds(self, w_x):
        """Push boxed object onto object stack."""
        assert isinstance(w_x, W_Object)
        ds_ptr = self.ds_ptr_locals
        self.ds_locals[ds_ptr] = w_x
        self.ds_ptr_locals = ds_ptr + 1

    def pop_ds(self):
        """Pop boxed object from object stack."""
        ds_ptr = self.ds_ptr_locals - 1
        assert ds_ptr >= 0
        w_x = self.ds_locals[ds_ptr]
        assert isinstance(w_x, W_Object)
        self.ds_locals[ds_ptr] = None
        self.ds_ptr_locals = ds_ptr
        return w_x

    def peek_ds(self, depth=0):
        """Peek at boxed object on object stack."""
        ptr = self.ds_ptr_locals - 1 - depth
        assert ptr >= 0
        return self.ds_locals[ptr]

    def top2_ds(self):
        """Pop two boxed objects from object stack."""
        w_y = self.pop_ds()
        w_x = self.pop_ds()
        return w_x, w_y

    # ==================== Stack Depth and Backward Compatibility ====================

    def ds_depth(self):
        """Return integer stack depth (for DEPTH primitive)."""
        return self.int_backing_ptr + self.int_window_ptr

    def float_depth(self):
        """Return float stack depth (for FDEPTH primitive)."""
        return self.float_ptr

    def rs_depth(self):
        """Return return stack depth."""
        return self.rs_backing_ptr + self.rs_window_ptr

    @property
    def ds_ptr_ints(self):
        """Backward compatible: get integer stack depth."""
        return self.int_backing_ptr + self.int_window_ptr

    @ds_ptr_ints.setter
    def ds_ptr_ints(self, value):
        """Backward compatible: setting to 0 clears the stack."""
        if value == 0:
            self.int_window_ptr = 0
            self.int_backing_ptr = 0
        else:
            raise ValueError("Can only set ds_ptr_ints to 0 to clear stack")

    @property
    def ds_ptr_floats(self):
        """Backward compatible: get float stack depth."""
        return self.float_ptr

    @ds_ptr_floats.setter
    def ds_ptr_floats(self, value):
        """Backward compatible: setting to 0 clears the stack."""
        if value == 0:
            self.float_ptr = 0
        else:
            raise ValueError("Can only set ds_ptr_floats to 0 to clear stack")

    @property
    def rs_ptr(self):
        """Backward compatible: get return stack depth."""
        return self.rs_backing_ptr + self.rs_window_ptr

    @rs_ptr.setter
    def rs_ptr(self, value):
        """Backward compatible: setting to 0 clears the stack."""
        if value == 0:
            self.rs_window_ptr = 0
            self.rs_backing_ptr = 0
        else:
            raise ValueError("Can only set rs_ptr to 0 to clear stack")

    @property
    def cs_ptr(self):
        """Backward compatible: get call stack depth."""
        return self.cs_backing_ptr + self.cs_window_ptr

    @cs_ptr.setter
    def cs_ptr(self, value):
        """Backward compatible: setting to 0 clears the stack."""
        if value == 0:
            self.cs_window_ptr = 0
            self.cs_backing_ptr = 0
            for i in range(WINDOW_SIZE):
                self.cs_window_threads[i] = None
        else:
            raise ValueError("Can only set cs_ptr to 0 to clear stack")

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
