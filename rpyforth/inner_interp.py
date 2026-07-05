from rpyforth.objects import (
    DECIMAL,
    ForthException,
    Word,
    CodeThread,
    ZERO,
    W_Object,
    W_IntObject,
    W_StringObject,
    W_PtrObject,
    CELL_SIZE_BYTES,
    CELL_SIZE,
    make_int,
    THREAD_REGISTRY,
)


import os

from rpython.rlib.jit import JitDriver, promote, elidable, unroll_safe, promote_string, hint
from rpython.rlib.debug import make_sure_not_resized
from rpython.rlib.rfile import create_stdio

from rpyforth.heap import (HEAP_CELL_COUNT, HEAP_SIZE_BYTES, DICT_SIZE_BYTES,
                           ALLOC_BASE, Heap, _alloc_region_bytes)


def alloc_region_bytes():
    return _alloc_region_bytes()

USE_VIRTUALIZATION = bool(os.environ.get("RPYFORTH_VIRTUALIZE"))

USE_STACK_FRAGMENT = bool(os.environ.get("RPYFORTH_STACK_FRAGMENT"))

from rpyforth.metastack import (
    FRAME_SIZE,
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

# Sentinel value for a primitive-initiated call (EXECUTE, CATCH): the primitive
# has already pushed the return frame(s); the dispatch loop transfers control to
# self.pending_word's thread, keeping the call inside the traced loop.
CALL_SENTINEL = -3

# Maximum number of simultaneously active CATCH frames.
CATCH_DEPTH = 16384

# Call-stack packing: a return address is (tid << CS_IP_BITS) | ip. 24 bits
# bound the ip within one code thread; tids are bounded by MAX_THREADS.
CS_IP_BITS = 24
CS_IP_MASK = (1 << CS_IP_BITS) - 1

class Exit(Exception):
    pass

class Bye(Exception):
    """Raised by BYE to exit the Forth system cleanly."""
    pass

class Abort(Exception):
    """Raised by ABORT" at runtime: unwinds every active portal (the stacks are
    already cleared by the raiser); the outer interpreter resumes."""
    pass


# Portal-boundary sentinel: execute_thread pushes it on entry, and popping it
# ends that invocation's dispatch loop. Inside a trace the return thread id is
# promoted, so the halt test constant-folds away on real returns.
HALT_THREAD = CodeThread([], [])

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
                          "lc_is", "lc_ls", "rs", "ds_locals", "heap",
                          "ca_tids", "ca_ips", "ca_dsi", "ca_dsf", "ca_dsl",
                          "ca_rs", "ca_li", "ca_lc", "ca_cs",
                          "ca_t0", "ca_t1", "ca_d", "ca_frag", "ca_spill",
                          "ca_frames"]

    if USE_VIRTUALIZATION:
        _virtualizable_ = ["ds_ints", "ds_floats",
                           "ds_ptr_ints", "ds_ptr_floats", "ds_ptr_locals",
                           "rs_ptr", "cs_pcs", "cs_ptr", "cs_base",
                           "li",
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

        # Virtualized call stack for JIT optimization. A return address is one
        # packed int (thread id << CS_IP_BITS | ip): no GC write barrier, one
        # array access per push/pop. Only the thread id is promoted on return so
        # the THREAD_REGISTRY lookup folds and the target thread inlines; the
        # return ip is left as a runtime value, so a caller that returns to many
        # different positions (EXECUTE/opcode-table dispatch) does not spawn a
        # guard_value per return site and shatter the trace into bridges.
        self.cs_pcs = [0] * STACK_SIZE
        self.cs_ptr = 0
        # Lower bound for execute_thread: it stops popping return frames when the
        # call stack drains back to this depth. execute_word_now raises it so a
        # nested run (e.g. CATCH) returns instead of escaping into outer frames.
        self.cs_base = 0

        # Total heap = dictionary region + a runtime-sized ALLOCATE region.
        self.heap_size = DICT_SIZE_BYTES + alloc_region_bytes()
        self.heap = Heap(self.heap_size)
        self.cell_size = CELL_SIZE
        self.cell_size_bytes = CELL_SIZE_BYTES

        # for string. Sized to the dictionary region only: boxed strings are only
        # ever parked at HERE-allocated (dictionary) addresses, never in the
        # separate high ALLOCATE region.
        self.buf = [None] * (DICT_SIZE_BYTES >> 3)
        self.here = 0
        # Bump pointer for ALLOCATE, in the high region above dictionary space,
        # and the highest address it may reach.
        self.alloc_ptr = ALLOC_BASE
        self.alloc_limit = self.heap_size
        # Free-list reuse for ALLOCATE/FREE: freed blocks are bucketed by usable
        # size so a later same-size ALLOCATE hands the block back instead of
        # bumping. gc.fs FREEs and re-ALLOCATEs same-size grain-info bitvectors on
        # every collection, so without this the bump pointer would grow without
        # bound across a long run. Keyed by usable size -> list of user addresses.
        self.alloc_free = {}

        self.base = 10
        self._pno_active = False
        self._pno_buf = []
        self.argv = []

        # Words bound to DEFER slots; a deferred word executes deferred_words[id].
        self.deferred_words = []

        # Catch-frame stack: flat parallel arrays (no allocation per CATCH).
        # A frame records the resume address plus every piece of machine state
        # THROW must restore; catch_ptr indexes the top. The frame's cached
        # stack-cache cells live at ca_frames[i*FRAME_SIZE : (i+1)*FRAME_SIZE].
        self.ca_tids = [0] * CATCH_DEPTH
        self.ca_ips = [0] * CATCH_DEPTH
        self.ca_dsi = [0] * CATCH_DEPTH
        self.ca_dsf = [0] * CATCH_DEPTH
        self.ca_dsl = [0] * CATCH_DEPTH
        self.ca_rs = [0] * CATCH_DEPTH
        self.ca_li = [0] * CATCH_DEPTH
        self.ca_lc = [0] * CATCH_DEPTH
        self.ca_cs = [0] * CATCH_DEPTH
        self.ca_t0 = [0] * CATCH_DEPTH
        self.ca_t1 = [0] * CATCH_DEPTH
        self.ca_d = [0] * CATCH_DEPTH
        self.ca_frag = [0] * CATCH_DEPTH
        self.ca_spill = [0] * CATCH_DEPTH
        self.ca_frames = [0] * (CATCH_DEPTH * FRAME_SIZE)
        self.catch_ptr = 0

        # Target of a CALL_SENTINEL transfer, set by the initiating primitive.
        self.pending_word = None

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
        """Push packed return address (tid << CS_IP_BITS | ip)."""
        ptr = self.cs_ptr
        assert ptr < len(self.cs_pcs)
        assert 0 <= ip < (1 << CS_IP_BITS)
        self.cs_pcs[ptr] = (thread.tid << CS_IP_BITS) | ip
        self.cs_ptr = ptr + 1

    def pop_call(self):
        """Pop packed return address; recover the thread from its id."""
        ptr = self.cs_ptr - 1
        assert ptr >= 0
        if USE_STACK_FRAGMENT:
            pop_ds_fragments_commit(self)
        self.cs_ptr = ptr
        pc = self.cs_pcs[ptr]
        # Clear the slot (mirrors the old null write): helps the JIT treat the
        # tail above cs_ptr as dead and elide the reads on recursive traces.
        self.cs_pcs[ptr] = 0
        # Promote only the thread id: the registry lookup folds and the return
        # thread inlines, while the return ip stays a runtime value so a
        # polymorphic caller (EXECUTE/opcode dispatch) does not guard_value each
        # return site into its own bridge.
        tid = promote(pc >> CS_IP_BITS)
        ip = pc & CS_IP_MASK
        thread = THREAD_REGISTRY.threads[tid]
        return thread, ip

    def is_call_stack_empty(self):
        """Check if call stack is empty."""
        return self.cs_ptr == 0

    def reset_after_abort(self):
        """Clear every machine stack. Runs at the Abort catch site, outside
        any portal, so compiled frames never see half-cleared state."""
        self.reset_ds_int()
        self.ds_ptr_floats = 0
        self.ds_ptr_locals = 0
        self.rs_ptr = 0
        self.lc_depth = 0
        self.cs_ptr = 0
        self.catch_ptr = 0

    @unroll_safe
    def catch_push_frame(self, thread, ip):
        """Record a CATCH resume point: return address plus all the state a
        THROW must restore. Flat array writes, no allocation."""
        cp = self.catch_ptr
        assert 0 <= cp < CATCH_DEPTH
        self.ca_tids[cp] = thread.tid
        self.ca_ips[cp] = ip
        self.ca_dsf[cp] = self.ds_ptr_floats
        self.ca_dsl[cp] = self.ds_ptr_locals
        self.ca_rs[cp] = self.rs_ptr
        self.ca_li[cp] = self.li
        self.ca_lc[cp] = self.lc_depth
        self.ca_cs[cp] = self.cs_ptr
        if USE_STACK_FRAGMENT:
            self.ca_t0[cp] = self.t0
            self.ca_t1[cp] = self.t1
            self.ca_d[cp] = self.d
            self.ca_frag[cp] = self.frag_ptr
            self.ca_spill[cp] = self.spill_ptr
            fbase = cp * FRAME_SIZE
            i = 0
            while i < FRAME_SIZE:
                self.ca_frames[fbase + i] = self.frame[i]
                i += 1
        else:
            self.ca_dsi[cp] = self.ds_ptr_ints
        self.catch_ptr = cp + 1

    def catch_drop_frame(self):
        """Discard the top catch frame (protected word returned normally)."""
        cp = self.catch_ptr - 1
        assert cp >= 0
        self.catch_ptr = cp

    @unroll_safe
    def throw_unwind(self, code):
        """Unwind to the nearest CATCH frame of this portal: restore its saved
        state, push the throw code, and return the resume (thread, ip). If the
        top frame belongs to an outer portal (below cs_base), re-raise so the
        exception propagates out of this execute_thread invocation."""
        cp = self.catch_ptr - 1
        if cp < 0 or self.ca_cs[cp] < self.cs_base:
            raise ForthException(code)
        assert cp >= 0
        self.catch_ptr = cp
        self.ds_ptr_floats = self.ca_dsf[cp]
        self.ds_ptr_locals = self.ca_dsl[cp]
        self.rs_ptr = self.ca_rs[cp]
        self.li = self.ca_li[cp]
        self.lc_depth = self.ca_lc[cp]
        self.cs_ptr = self.ca_cs[cp]
        if USE_STACK_FRAGMENT:
            self.t0 = self.ca_t0[cp]
            self.t1 = self.ca_t1[cp]
            self.d = self.ca_d[cp]
            self.frag_ptr = self.ca_frag[cp]
            self.spill_ptr = self.ca_spill[cp]
            fbase = cp * FRAME_SIZE
            i = 0
            while i < FRAME_SIZE:
                self.frame[i] = self.ca_frames[fbase + i]
                i += 1
        else:
            self.ds_ptr_ints = self.ca_dsi[cp]
        self.push_ds_int(code)
        thread = THREAD_REGISTRY.threads[self.ca_tids[cp]]
        ip = self.ca_ips[cp]
        return thread, ip

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

    def buf_get(self, addr):
        """Boxed string parked at a cell-aligned data-space address, or None."""
        i = addr >> 3
        if 0 <= i < len(self.buf):
            return self.buf[i]
        return None

    def buf_set(self, addr, w_str):
        i = addr >> 3
        assert 0 <= i < len(self.buf)
        self.buf[i] = w_str

    def alloc_buf(self, content, size):
        # The string lives both as a boxed object (legacy consumers use
        # buf_get) and as real bytes in data space, so char-level words
        # (MOVE, COUNT, filename assembly) see the same characters. here
        # is aligned first so each string owns a distinct buf slot.
        addr = (self.here + CELL_SIZE_BYTES - 1) & ~(CELL_SIZE_BYTES - 1)
        assert size >= 0
        self.buf_set(addr, W_StringObject(content[:size]))
        i = 0
        while i < size:
            self.char_store(addr + i, ord(content[i]))
            i += 1
        self.here = (addr + size + CELL_SIZE_BYTES) & ~(CELL_SIZE_BYTES - 1)
        return addr

    def _ensure_addr(self, addr, span):
        assert 0 <= addr < self.heap_size
        assert addr + span <= self.heap_size

    def cell_store(self, addr, intval):
        assert isinstance(addr, int)
        self.heap.cell_store(addr, intval)

    def cell_fetch_int(self, addr):
        return self.heap.cell_fetch_int(addr)

    def cell_fetch(self, addr):
        return self.heap.cell_fetch(addr)

    def cell_2store(self, addr, x1_int, x2_int):
        """Standard 2!: x2 (the top cell) lands at addr, x1 at the next."""
        assert 0 <= addr < self.heap_size - self.cell_size_bytes
        self.cell_store(addr, x2_int)
        self.cell_store(addr + self.cell_size_bytes, x1_int)

    def cell_2fetch(self, addr):
        """Standard 2@: returns (x1, x2) with x2 (the top) taken from addr."""
        assert 0 <= addr < self.heap_size - self.cell_size_bytes
        x1 = make_int(self.cell_fetch_int(addr + self.cell_size_bytes))
        x2 = make_int(self.cell_fetch_int(addr))
        return x1, x2

    def char_store(self, addr, intval):
        self.heap.char_store(addr, intval)

    def char_fetch(self, addr):
        return self.heap.char_fetch(addr)

    def set_base(self, n):
        """Set the conversion radix and mirror it into the BASE cell so `BASE @`
        reads the current value. Keeps HEX/DECIMAL/etc. in step with the cell that
        Forth code manipulates through @ / !."""
        self.base = n
        if self.outer is not None:
            self.cell_store(self.outer.base_addr, n)

    def float_store(self, addr, value):
        self.heap.float_store(addr, value)

    def cell_float_fetch(self, addr):
        return self.heap.float_fetch_float(addr)

    def float_fetch(self, addr):
        return self.heap.float_fetch(addr)

    def execute_thread(self, thread, ip=0):
        # A halt frame marks the portal boundary. Every return pops
        # unconditionally; the pop's promoted return address folds the halt test
        # away inside traces, so returning costs one guard and no depth compare.
        self.push_call(HALT_THREAD, 0)
        if USE_STACK_FRAGMENT:
            push_ds_fragments(self)
        while True:
            jitdriver.jit_merge_point(ip=ip, thread=thread, self=self)
            if ip >= len(thread.code):
                thread, ip = self.pop_call()
                if thread is HALT_THREAD:
                    break
                continue

            w = promote(thread.code[ip])
            if w is None:
                thread, ip = self.pop_call()
                if thread is HALT_THREAD:
                    break
                continue
            ip += 1

            prim = promote(w.prim)
            if prim is not None:
                try:
                    ip = prim(self, thread, ip)
                except ForthException as e:
                    thread, ip = self.throw_unwind(e.code)
                    continue
                if ip == EXIT_SENTINEL:
                    thread, ip = self.pop_call()
                    if thread is HALT_THREAD:
                        break
                    continue
                if ip == CALL_SENTINEL:
                    target = promote(self.pending_word)
                    self.pending_word = None
                    thread = target.thread
                    ip = 0
                    jitdriver.can_enter_jit(ip=ip, thread=thread, self=self)
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
                    thread, ip = self.pop_call()
                    if thread is HALT_THREAD:
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
        # Run w as a self-contained call: bound the interpreter to the current
        # call-stack depth so it returns when w finishes instead of draining the
        # shared call stack into the caller's frames. CATCH relies on this to
        # avoid native-stack growth when nested inside a loop.
        nt = w.now_thread
        if nt is None:
            nt = CodeThread([w], [ZERO])
            w.now_thread = nt
        saved_base = self.cs_base
        self.cs_base = self.cs_ptr
        try:
            self.execute_thread(nt, 0)
        finally:
            self.cs_base = saved_base
