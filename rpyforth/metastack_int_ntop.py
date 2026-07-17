"""Parametric-NTOP variant of the int metastack (metastack_int.py).

Generalizes the flagship's NTOP=2 scalar tops to EFFECTIVE_NTOP, a
translation-time constant from RPYFORTH_NTOP (2/4/8/16; 2 validates against the
flagship). Layout: t0..t{EFFECTIVE_NTOP-1} scalars + frame[FRAME_SIZE] (both
virtualizable) + one shared spill array.

All sixteen scalar fields t0..t15 always exist; hot methods are unrolled chains
gated by constant compares against EFFECTIVE_NTOP, which the translator folds so
exactly EFFECTIVE_NTOP scalars stay live.

The call boundary keeps CALL_WINDOW = NTOP = 2 for calling-convention parity
across every NTOP value, so push_fragment_on parks deepest-first (the scalar tops
beyond the window as well as the frame cells) and pop_fragment_commit is O(1).
See docs/NOTE_STACK_LAYOUT.md for the full scheme.
"""

from rpython.rlib.jit import promote, unroll_safe
from rpython.rlib.debug import make_sure_not_resized

from rpyforth.metastack import (
    EFFECTIVE_NTOP,
    FRAME_SIZE,
    CALL_WINDOW,
    SPILL_SIZE,
    DataStackOverflow,
    DSMetaStack,
)

# Total cached cells: EFFECTIVE_NTOP scalar tops + FRAME_SIZE frame cells.
NTOP_ACTIVE_MAX = EFFECTIVE_NTOP + FRAME_SIZE


def init_fields(host):
    """Install the host-resident active-cache + metastack spill state."""
    # All sixteen scalar tops are always present so the field set is uniform
    # across every NTOP value; only the first EFFECTIVE_NTOP are live after
    # constant folding. t0 is the top, t{EFFECTIVE_NTOP-1} the deepest scalar.
    host.t0 = 0
    host.t1 = 0
    host.t2 = 0
    host.t3 = 0
    host.t4 = 0
    host.t5 = 0
    host.t6 = 0
    host.t7 = 0
    host.t8 = 0
    host.t9 = 0
    host.t10 = 0
    host.t11 = 0
    host.t12 = 0
    host.t13 = 0
    host.t14 = 0
    host.t15 = 0
    host.cache_depth = 0
    host.frame = [0] * FRAME_SIZE
    make_sure_not_resized(host.frame)

    # Shared spill: every cell below the cached window. Plain heap (immutable
    # reference), sized to the whole stack depth, one per VM; every fragment is
    # a window [0, spill_ptr) onto it, so nest/unnest allocates nothing.
    host.frag_ptr = 0
    host.spill = [0] * SPILL_SIZE
    make_sure_not_resized(host.spill)
    host.spill_ptr = 0


class DSCacheSnapshotN(object):
    """Immutable snapshot of the parametric cache: all sixteen scalar tops (only
    the live ones matter), cache_depth, a private copy of the frame, and the two
    pointers. The shared spill is not copied; restore rolls spill_ptr back."""

    _immutable_fields_ = ["t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7",
                          "t8", "t9", "t10", "t11", "t12", "t13", "t14", "t15",
                          "cache_depth", "frame[*]", "frag_ptr", "spill_ptr"]

    def __init__(self, tops, cache_depth, frame, frag_ptr, spill_ptr):
        self.t0 = tops[0]
        self.t1 = tops[1]
        self.t2 = tops[2]
        self.t3 = tops[3]
        self.t4 = tops[4]
        self.t5 = tops[5]
        self.t6 = tops[6]
        self.t7 = tops[7]
        self.t8 = tops[8]
        self.t9 = tops[9]
        self.t10 = tops[10]
        self.t11 = tops[11]
        self.t12 = tops[12]
        self.t13 = tops[13]
        self.t14 = tops[14]
        self.t15 = tops[15]
        self.cache_depth = cache_depth
        self.frame = frame
        self.frag_ptr = frag_ptr
        self.spill_ptr = spill_ptr


@unroll_safe
def snapshot_cache(host):
    """Capture host's active cache. Copies the fixed-size frame so later cache
    writes cannot disturb the snapshot."""
    frame_copy = [0] * FRAME_SIZE
    i = 0
    while i < FRAME_SIZE:
        frame_copy[i] = host.frame[i]
        i += 1
    make_sure_not_resized(frame_copy)
    tops = [host.t0, host.t1, host.t2, host.t3, host.t4, host.t5, host.t6,
            host.t7, host.t8, host.t9, host.t10, host.t11, host.t12, host.t13,
            host.t14, host.t15]
    return DSCacheSnapshotN(tops, host.cache_depth, frame_copy,
                            host.frag_ptr, host.spill_ptr)


@unroll_safe
def restore_cache(host, snap):
    """Roll host's stack back to a snapshot, discarding everything pushed since.
    Restores the cache and the cache/spill pointers; the spill cells below the
    saved spill pointer are left in place."""
    host.t0 = snap.t0
    host.t1 = snap.t1
    host.t2 = snap.t2
    host.t3 = snap.t3
    host.t4 = snap.t4
    host.t5 = snap.t5
    host.t6 = snap.t6
    host.t7 = snap.t7
    host.t8 = snap.t8
    host.t9 = snap.t9
    host.t10 = snap.t10
    host.t11 = snap.t11
    host.t12 = snap.t12
    host.t13 = snap.t13
    host.t14 = snap.t14
    host.t15 = snap.t15
    host.cache_depth = snap.cache_depth
    i = 0
    while i < FRAME_SIZE:
        host.frame[i] = snap.frame[i]
        i += 1
    host.frag_ptr = snap.frag_ptr
    host.spill_ptr = snap.spill_ptr


class DSIntMetaStackN(DSMetaStack):
    """Integer data stack in the parametric-NTOP three-tier layout:
    t0..t{EFFECTIVE_NTOP-1} scalars | frame[FRAME_SIZE] (both virtualizable) | one
    shared spill array. The spill is allocated once per VM; a fragment is just the
    window [0, spill_ptr) onto it, so nest/unnest allocates nothing."""

    def init_fields(self):
        init_fields(self)

    # ------------------------------------------------------------------
    # Scalar-tops helpers. Written as UNROLLED chains gated by constant
    # comparisons against EFFECTIVE_NTOP; the translator folds each branch to a
    # constant, leaving exactly EFFECTIVE_NTOP live scalars. A vable array cannot be
    # indexed by a runtime offset, so the scalars are addressed only through
    # these constant-index chains.
    # ------------------------------------------------------------------
    def _get_scalar(self, k):
        # Return the k'th scalar top (0 = top). k must be < EFFECTIVE_NTOP.
        if k == 0:
            return self.t0
        if k == 1:
            return self.t1
        if k == 2:
            return self.t2
        if k == 3:
            return self.t3
        if k == 4:
            return self.t4
        if k == 5:
            return self.t5
        if k == 6:
            return self.t6
        if k == 7:
            return self.t7
        if k == 8:
            return self.t8
        if k == 9:
            return self.t9
        if k == 10:
            return self.t10
        if k == 11:
            return self.t11
        if k == 12:
            return self.t12
        if k == 13:
            return self.t13
        if k == 14:
            return self.t14
        return self.t15

    def _set_scalar(self, k, v):
        if k == 0:
            self.t0 = v
        elif k == 1:
            self.t1 = v
        elif k == 2:
            self.t2 = v
        elif k == 3:
            self.t3 = v
        elif k == 4:
            self.t4 = v
        elif k == 5:
            self.t5 = v
        elif k == 6:
            self.t6 = v
        elif k == 7:
            self.t7 = v
        elif k == 8:
            self.t8 = v
        elif k == 9:
            self.t9 = v
        elif k == 10:
            self.t10 = v
        elif k == 11:
            self.t11 = v
        elif k == 12:
            self.t12 = v
        elif k == 13:
            self.t13 = v
        elif k == 14:
            self.t14 = v
        else:
            self.t15 = v

    @unroll_safe
    def _push_scalar(self, v):
        # Shift the scalar tops down by one and drop v into t0. The scalar that
        # falls off the deepest slot (t{EFFECTIVE_NTOP-1}) is handled by the caller,
        # which reads it before this shift (push_on) -- here we only slide.
        if EFFECTIVE_NTOP > 15:
            self.t15 = self.t14
        if EFFECTIVE_NTOP > 14:
            self.t14 = self.t13
        if EFFECTIVE_NTOP > 13:
            self.t13 = self.t12
        if EFFECTIVE_NTOP > 12:
            self.t12 = self.t11
        if EFFECTIVE_NTOP > 11:
            self.t11 = self.t10
        if EFFECTIVE_NTOP > 10:
            self.t10 = self.t9
        if EFFECTIVE_NTOP > 9:
            self.t9 = self.t8
        if EFFECTIVE_NTOP > 8:
            self.t8 = self.t7
        if EFFECTIVE_NTOP > 7:
            self.t7 = self.t6
        if EFFECTIVE_NTOP > 6:
            self.t6 = self.t5
        if EFFECTIVE_NTOP > 5:
            self.t5 = self.t4
        if EFFECTIVE_NTOP > 4:
            self.t4 = self.t3
        if EFFECTIVE_NTOP > 3:
            self.t3 = self.t2
        if EFFECTIVE_NTOP > 2:
            self.t2 = self.t1
        if EFFECTIVE_NTOP > 1:
            self.t1 = self.t0
        self.t0 = v

    @unroll_safe
    def _pop_scalar_shift(self, refill):
        # Slide the scalar tops up by one (t0 = t1, t1 = t2, ...) and drop
        # ``refill`` into the deepest live scalar t{EFFECTIVE_NTOP-1}. The old t0 is
        # read by the caller before this call.
        if EFFECTIVE_NTOP > 1:
            self.t0 = self.t1
        if EFFECTIVE_NTOP > 2:
            self.t1 = self.t2
        if EFFECTIVE_NTOP > 3:
            self.t2 = self.t3
        if EFFECTIVE_NTOP > 4:
            self.t3 = self.t4
        if EFFECTIVE_NTOP > 5:
            self.t4 = self.t5
        if EFFECTIVE_NTOP > 6:
            self.t5 = self.t6
        if EFFECTIVE_NTOP > 7:
            self.t6 = self.t7
        if EFFECTIVE_NTOP > 8:
            self.t7 = self.t8
        if EFFECTIVE_NTOP > 9:
            self.t8 = self.t9
        if EFFECTIVE_NTOP > 10:
            self.t9 = self.t10
        if EFFECTIVE_NTOP > 11:
            self.t10 = self.t11
        if EFFECTIVE_NTOP > 12:
            self.t11 = self.t12
        if EFFECTIVE_NTOP > 13:
            self.t12 = self.t13
        if EFFECTIVE_NTOP > 14:
            self.t13 = self.t14
        if EFFECTIVE_NTOP > 15:
            self.t14 = self.t15
        self._set_scalar(EFFECTIVE_NTOP - 1, refill)

    # ------------------------------------------------------------------
    # Hot path. The top EFFECTIVE_NTOP cells are scalars; the frame array is reached
    # only past depth EFFECTIVE_NTOP, and the spill only past NTOP_ACTIVE_MAX.
    # ------------------------------------------------------------------
    def push_on(self, v):
        dd = self.cache_depth
        if dd >= NTOP_ACTIVE_MAX:
            self._spill_bottom()
            dd = self.cache_depth
        if dd >= EFFECTIVE_NTOP:
            si = dd - EFFECTIVE_NTOP
            assert si >= 0
            self.frame[si] = self._get_scalar(EFFECTIVE_NTOP - 1)
        self._push_scalar(v)
        self.cache_depth = dd + 1

    def pop_on(self):
        dd = self.cache_depth
        if dd <= 0:
            return self._pop_from_spill()
        r = self.t0
        if dd > EFFECTIVE_NTOP:
            si = dd - EFFECTIVE_NTOP - 1
            assert si >= 0
            self._pop_scalar_shift(self.frame[si])
        else:
            self._pop_scalar_shift(0)
        self.cache_depth = dd - 1
        return r

    def _pop_from_spill(self):
        # Cache empty: the top now lives in the spill (a callee consumed past its
        # imported window into parent cells). Underflow is an interpreter bug,
        # checked by assert (no branch on the hot path).
        ap = self.spill_ptr - 1
        assert ap >= 0
        r = self.spill[ap]
        self.spill_ptr = ap
        return r

    @unroll_safe
    def _spill_bottom(self):
        # Move the deepest cached cell (frame[0]) to the spill and slide the
        # frame down, leaving one free slot. Cold: only when the stack is deeper
        # than NTOP_ACTIVE_MAX.
        ap = self.spill_ptr
        if ap >= SPILL_SIZE:
            raise DataStackOverflow()
        assert ap >= 0
        self.spill[ap] = self.frame[0]
        self.spill_ptr = ap + 1
        i = 0
        while i < FRAME_SIZE - 1:
            self.frame[i] = self.frame[i + 1]
            i += 1
        self.cache_depth = self.cache_depth - 1

    @unroll_safe
    def _park_one(self):
        # Evacuate the single deepest cached cell to the spill and slide the
        # whole cache (frame then scalars) down by one, so the surviving cells
        # end packed at the shallow end (t0, t1, ...). Deepest cell is frame[0]
        # when the frame holds cells, else the deepest live scalar. Reuses only
        # the constant-index scalar chains and the frame[i]=frame[i+1] slide, so
        # no runtime-offset index ever touches the virtualizable arrays.
        ap = self.spill_ptr
        if ap >= SPILL_SIZE:
            raise DataStackOverflow()
        assert ap >= 0
        dd = self.cache_depth
        if dd > EFFECTIVE_NTOP:
            self.spill[ap] = self.frame[0]
            i = 0
            while i < FRAME_SIZE - 1:
                self.frame[i] = self.frame[i + 1]
                i += 1
            si = dd - EFFECTIVE_NTOP - 1
            assert si >= 0
            self.frame[si] = self._get_scalar(EFFECTIVE_NTOP - 1)
        else:
            self.spill[ap] = self._get_scalar(dd - 1)
        self.spill_ptr = ap + 1
        self.cache_depth = dd - 1

    def peek_on(self, depth):
        depth = promote(depth)
        dd = self.cache_depth
        if depth < dd:
            if depth < EFFECTIVE_NTOP:
                return self._get_scalar(depth)
            si = dd - 1 - depth
            assert si >= 0
            return self.frame[si]
        ai = self.spill_ptr - 1 - (depth - dd)
        assert ai >= 0
        return self.spill[ai]

    def poke_on(self, depth, v):
        depth = promote(depth)
        dd = self.cache_depth
        if depth < dd:
            if depth < EFFECTIVE_NTOP:
                self._set_scalar(depth, v)
                return
            si = dd - 1 - depth
            assert si >= 0
            self.frame[si] = v
            return
        ai = self.spill_ptr - 1 - (depth - dd)
        assert ai >= 0
        self.spill[ai] = v

    def depth_on(self):
        return self.cache_depth + self.spill_ptr

    @unroll_safe
    def reset_on(self):
        self.t0 = 0
        self.t1 = 0
        self.t2 = 0
        self.t3 = 0
        self.t4 = 0
        self.t5 = 0
        self.t6 = 0
        self.t7 = 0
        self.t8 = 0
        self.t9 = 0
        self.t10 = 0
        self.t11 = 0
        self.t12 = 0
        self.t13 = 0
        self.t14 = 0
        self.t15 = 0
        self.cache_depth = 0
        self.frag_ptr = 0
        self.spill_ptr = 0

    # ------------------------------------------------------------------
    # Call entry / return. On a call the caller's below-CALL_WINDOW cells are
    # parked in the spill and the active depth is normalized to the CALL_WINDOW
    # top cells (2, regardless of EFFECTIVE_NTOP -- calling-convention parity
    # across every NTOP value). Return is O(1): the spill already holds the
    # caller's cells.
    # ------------------------------------------------------------------
    @unroll_safe
    def push_fragment_on(self):
        self.frag_ptr = self.frag_ptr + 1
        dd = self.cache_depth
        # Park below-window cells one at a time, deepest first, so each ends at
        # spill[ap+k] in the same order the flagship parks and the surviving
        # window cells finish packed at t0..t{CALL_WINDOW-1}.
        while dd > CALL_WINDOW:
            self._park_one()
            dd -= 1

    def pop_fragment_commit_on(self):
        # O(1): the callee's net result is already the cache top and the caller's
        # parked cells are already the spill top, contiguously below it. Nothing
        # to move -- just unwind the call counter. Every poppable call-stack
        # entry was pushed with a paired push_fragment_on (and ABORT zeroes both
        # counters together), so the counter cannot underflow; assert instead of
        # branching on the hot return path.
        fp = self.frag_ptr - 1
        assert fp >= 0
        self.frag_ptr = fp

    # ------------------------------------------------------------------
    # Public, test-facing wrappers.
    # ------------------------------------------------------------------
    def __init__(self):
        self.init_fields()

    def push(self, v):
        self.push_on(v)

    def pop(self):
        return self.pop_on()

    def peek(self, depth):
        return self.peek_on(depth)

    def poke(self, depth, v):
        self.poke_on(depth, v)

    def size(self):
        return self.depth_on()

    def clear(self):
        self.reset_on()

    def push_fragment(self):
        self.push_fragment_on()

    def pop_fragment_commit(self):
        self.pop_fragment_commit_on()

    def snapshot(self):
        return snapshot_cache(self)

    def restore(self, snap):
        restore_cache(self, snap)
