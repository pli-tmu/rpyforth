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
    ACTIVE_MAX,
    EFFECTIVE_NTOP,
    USE_FLOAT_FRAGMENT,
    USE_FRAME_ONLY,
    USE_NTOP_VARIANT,
    STACK_FRAGMENT_VIRTUALIZABLES,
    push_ds_fragments,
    pop_ds_fragments_commit,
    reset_ds_fragments,
)

# CATCH saves a full copy of the int cache. In the flagship the frame holds
# FRAME_SIZE cells (the two scalar tops are saved separately in ca_t0/ca_t1);
# in the frame-only ablation the whole cache is the frame, so its rows are
# ACTIVE_MAX wide. In the parametric-NTOP ablation the EFFECTIVE_NTOP scalar tops
# have no dedicated ca_t* arrays -- they are folded into the ca_frames row,
# stored at offsets 0..EFFECTIVE_NTOP-1 with the frame cells at EFFECTIVE_NTOP..,
# so the row is EFFECTIVE_NTOP + FRAME_SIZE wide. A flag-constant width keeps
# every path a single sized array.
if USE_FRAME_ONLY:
    CA_FRAME_WIDTH = ACTIVE_MAX
elif USE_NTOP_VARIANT:
    CA_FRAME_WIDTH = EFFECTIVE_NTOP + FRAME_SIZE
else:
    CA_FRAME_WIDTH = FRAME_SIZE

if USE_STACK_FRAGMENT:
    if USE_FRAME_ONLY:
        from rpyforth.metastack_int_frameonly import DSIntMetaStackFrameOnly
        InterpBase = DSIntMetaStackFrameOnly
    elif USE_NTOP_VARIANT:
        from rpyforth.metastack_int_ntop import DSIntMetaStackN
        InterpBase = DSIntMetaStackN
    elif USE_FLOAT_FRAGMENT:
        from rpyforth.metastack_float import DSFloatMetaStack
        InterpBase = DSFloatMetaStack
    else:
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
# self.pending_box's thread, keeping the call inside the traced loop.
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
    # Must be total for every (ip, thread) the JIT ever logs: ip == len(code)
    # is the thread-end/return state and code/lits slots may hold None, both
    # legitimate at merge points. Translated builds drop bounds checks, so an
    # out-of-range index or None dereference here segfaults the VM whenever
    # PYPYLOG jit logging is enabled.
    if ip < 0 or ip >= len(thread.code):
        return "ip=%d <thread-end>" % ip
    w = thread.code[ip]
    lit = thread.lits[ip]
    wname = w.to_string() if w is not None else "<none>"
    lname = lit.to_string() if lit is not None else "<none>"
    return "ip=%d %s %s" % (ip, wname, lname)

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
    # The metastack spill is allocated once and never reassigned, so its
    # reference is immutable even though its elements are mutated in place; this
    # lets the JIT hoist the array-pointer load to the loop header instead of
    # reloading it on every spill/refill.
    _immutable_fields_ = ["cell_size", "cell_size_bytes", "base", "spill",
                          "lc_is", "lc_ls", "rs", "ds_locals", "heap",
                          "ca_tids", "ca_ips", "ca_dsi", "ca_dsf", "ca_dsl",
                          "ca_rs", "ca_li", "ca_lc", "ca_cs",
                          "ca_t0", "ca_t1", "ca_cache_depth", "ca_frag", "ca_spill",
                          "ca_frames",
                          "ca_fft0", "ca_fft1", "ca_ffd", "ca_ffrag",
                          "ca_fspill", "ca_fframes", "pending_box"]

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
        self.ca_cache_depth = [0] * CATCH_DEPTH
        self.ca_frag = [0] * CATCH_DEPTH
        self.ca_spill = [0] * CATCH_DEPTH
        self.ca_frames = [0] * (CATCH_DEPTH * CA_FRAME_WIDTH)
        # Parallel float-cache save slots, mirroring the int ones above.
        self.ca_fft0 = [0.0] * CATCH_DEPTH
        self.ca_fft1 = [0.0] * CATCH_DEPTH
        self.ca_ffd = [0] * CATCH_DEPTH
        self.ca_ffrag = [0] * CATCH_DEPTH
        self.ca_fspill = [0] * CATCH_DEPTH
        self.ca_fframes = [0.0] * (CATCH_DEPTH * FRAME_SIZE)
        self.catch_ptr = 0

        # Target of a CALL_SENTINEL transfer, set by the initiating primitive.
        # Held in a one-element list whose reference never changes: writing the
        # slot is an array store on a separate object, not a store to a field
        # of the virtualizable self. A field store on the vable would force it
        # to materialize at the handoff point; the stable box keeps that store
        # off the vable so EXECUTE/CATCH dispatch cannot force an escape here.
        self.pending_box = [None]

        if USE_STACK_FRAGMENT:
            self.init_fields()
            if USE_FLOAT_FRAGMENT:
                self.init_float_fields()
        else:
            # Placeholders so the attribute set is consistent; unused on the
            # fixed path (the translator folds the dead USE_STACK_FRAGMENT
            # branch, so the active frame and fragment pool are only built when
            # fragmented).
            self.t0 = 0
            self.t1 = 0
            self.cache_depth = 0
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
        # return site into its own bridge. An adaptive variant that skipped the
        # promote for statically-megamorphic callees (>=8 compiled call sites)
        # was measured 2x SLOWER on cd16sim: ending the trace at every such
        # return and re-entering through the green-keyed trace map costs more
        # than the guard_value bridge chain it replaces.
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
        self.reset_ds_float()
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
        self.ca_dsf[cp] = self.depth_ds_float()
        self.ca_dsl[cp] = self.ds_ptr_locals
        self.ca_rs[cp] = self.rs_ptr
        self.ca_li[cp] = self.li
        self.ca_lc[cp] = self.lc_depth
        self.ca_cs[cp] = self.cs_ptr
        if USE_STACK_FRAGMENT:
            if not USE_FRAME_ONLY and not USE_NTOP_VARIANT:
                self.ca_t0[cp] = self.t0
                self.ca_t1[cp] = self.t1
            self.ca_cache_depth[cp] = self.cache_depth
            self.ca_frag[cp] = self.frag_ptr
            self.ca_spill[cp] = self.spill_ptr
            fbase = cp * CA_FRAME_WIDTH
            if USE_NTOP_VARIANT:
                k = 0
                while k < EFFECTIVE_NTOP:
                    self.ca_frames[fbase + k] = self._get_scalar(k)
                    k += 1
                i = 0
                while i < FRAME_SIZE:
                    self.ca_frames[fbase + EFFECTIVE_NTOP + i] = self.frame[i]
                    i += 1
            else:
                i = 0
                while i < CA_FRAME_WIDTH:
                    self.ca_frames[fbase + i] = self.frame[i]
                    i += 1
        else:
            self.ca_dsi[cp] = self.ds_ptr_ints
        if USE_FLOAT_FRAGMENT:
            self.ca_fft0[cp] = self.ft0
            self.ca_fft1[cp] = self.ft1
            self.ca_ffd[cp] = self.fdep
            self.ca_ffrag[cp] = self.ffrag_ptr
            self.ca_fspill[cp] = self.fspill_ptr
            ffbase = cp * FRAME_SIZE
            i = 0
            while i < FRAME_SIZE:
                self.ca_fframes[ffbase + i] = self.fframe[i]
                i += 1
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
        if not USE_FLOAT_FRAGMENT:
            self.ds_ptr_floats = self.ca_dsf[cp]
        self.ds_ptr_locals = self.ca_dsl[cp]
        self.rs_ptr = self.ca_rs[cp]
        self.li = self.ca_li[cp]
        self.lc_depth = self.ca_lc[cp]
        self.cs_ptr = self.ca_cs[cp]
        if USE_STACK_FRAGMENT:
            if not USE_FRAME_ONLY and not USE_NTOP_VARIANT:
                self.t0 = self.ca_t0[cp]
                self.t1 = self.ca_t1[cp]
            self.cache_depth = self.ca_cache_depth[cp]
            self.frag_ptr = self.ca_frag[cp]
            self.spill_ptr = self.ca_spill[cp]
            fbase = cp * CA_FRAME_WIDTH
            if USE_NTOP_VARIANT:
                k = 0
                while k < EFFECTIVE_NTOP:
                    self._set_scalar(k, self.ca_frames[fbase + k])
                    k += 1
                i = 0
                while i < FRAME_SIZE:
                    self.frame[i] = self.ca_frames[fbase + EFFECTIVE_NTOP + i]
                    i += 1
            else:
                i = 0
                while i < CA_FRAME_WIDTH:
                    self.frame[i] = self.ca_frames[fbase + i]
                    i += 1
        else:
            self.ds_ptr_ints = self.ca_dsi[cp]
        if USE_FLOAT_FRAGMENT:
            self.ft0 = self.ca_fft0[cp]
            self.ft1 = self.ca_fft1[cp]
            self.fdep = self.ca_ffd[cp]
            self.ffrag_ptr = self.ca_ffrag[cp]
            self.fspill_ptr = self.ca_fspill[cp]
            ffbase = cp * FRAME_SIZE
            i = 0
            while i < FRAME_SIZE:
                self.fframe[i] = self.ca_fframes[ffbase + i]
                i += 1
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
        if USE_FLOAT_FRAGMENT:
            self.fpush_on(floatval)
        else:
            self.push_ds_float_fixed(floatval)

    def push_ds_float_fixed(self, floatval):
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
        if USE_FLOAT_FRAGMENT:
            return self.fpop_on()
        return self.pop_ds_float_fixed()

    def pop_ds_float_fixed(self):
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
        if USE_FLOAT_FRAGMENT:
            return self.fpeek_on(depth)
        return self.peek_ds_float_fixed(depth)

    def peek_ds_float_fixed(self, depth=0):
        ptr = self.ds_ptr_floats - 1 - depth
        assert ptr >= 0
        return self.ds_floats[ptr]

    def poke_ds_float(self, depth, floatval):
        if USE_FLOAT_FRAGMENT:
            self.fpoke_on(depth, floatval)
        else:
            ptr = self.ds_ptr_floats - 1 - depth
            assert ptr >= 0
            self.ds_floats[ptr] = floatval

    def depth_ds_float(self):
        if USE_FLOAT_FRAGMENT:
            return self.fdepth_on()
        return self.ds_ptr_floats

    def reset_ds_float(self):
        if USE_FLOAT_FRAGMENT:
            self.freset_on()
        else:
            self.ds_ptr_floats = 0

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
                    target = promote(self.pending_box[0])
                    self.pending_box[0] = None
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
