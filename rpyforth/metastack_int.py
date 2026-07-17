from rpython.rlib.jit import promote, unroll_safe
from rpython.rlib.debug import make_sure_not_resized

from rpyforth.metastack import (
    NTOP,
    FRAME_SIZE,
    ACTIVE_MAX,
    SPILL_SIZE,
    DataStackOverflow,
    DSFragment,
    DSMetaStack,
)


class DSIntFragment(DSFragment):
    """Legacy value object, retained only for import compatibility."""

    _immutable_fields_ = ["parent"]

    def __init__(self, parent):
        self.parent = parent


def init_fields(host):
    """Install the host-resident active-fragment + metastack spill state."""
    # The active fragment caches the TOP of the data stack:
    #   * the top NTOP cells in scalar fields t0, t1 (hottest, in registers)
    #   * the next cells in the small virtualizable spill array ``frame``
    # ``cache_depth`` is the number of cells currently cached (0..ACTIVE_MAX).
    host.t0 = 0
    host.t1 = 0
    host.cache_depth = 0
    host.frame = [0] * FRAME_SIZE
    make_sure_not_resized(host.frame)

    # The shared spill holds every cell BELOW the cached fragment -- the caller
    # frames parked on a call plus the rare single-word overflow ("other
    # places"). Contiguous: spill[spill_ptr-1] is the cell just under the cache,
    # spill[0] the bottom of the stack. Plain heap (immutable reference): it is
    # sized to the whole stack depth, too large to virtualize. One spill per VM,
    # allocated once here; every fragment is a window [0, spill_ptr) onto it, so
    # nest/unnest allocates nothing. frag_ptr just tracks call nesting.
    host.frag_ptr = 0
    host.spill = [0] * SPILL_SIZE
    make_sure_not_resized(host.spill)
    host.spill_ptr = 0


class DSCacheSnapshot(object):
    """Immutable snapshot of the cache: scalar tops, cache_depth, a private copy
    of the frame, and the two pointers. The shared spill is not copied; restore
    rolls spill_ptr back and relies on the cells below it being undisturbed."""

    _immutable_fields_ = ["t0", "t1", "cache_depth", "frame[*]", "frag_ptr", "spill_ptr"]

    def __init__(self, t0, t1, cache_depth, frame, frag_ptr, spill_ptr):
        self.t0 = t0
        self.t1 = t1
        self.cache_depth = cache_depth
        self.frame = frame
        self.frag_ptr = frag_ptr
        self.spill_ptr = spill_ptr


def snapshot_cache(host):
    """Capture host's active-fragment state. Copies the fixed-size frame so later
    cache writes cannot disturb the snapshot."""
    frame_copy = [0] * FRAME_SIZE
    i = 0
    while i < FRAME_SIZE:
        frame_copy[i] = host.frame[i]
        i += 1
    make_sure_not_resized(frame_copy)
    return DSCacheSnapshot(host.t0, host.t1, host.cache_depth, frame_copy,
                           host.frag_ptr, host.spill_ptr)


def restore_cache(host, snap):
    """Roll host's stack back to a snapshot, discarding everything pushed since.
    Restores the cache and the cache/spill pointers; the spill cells below the
    saved spill pointer are left in place."""
    host.t0 = snap.t0
    host.t1 = snap.t1
    host.cache_depth = snap.cache_depth
    i = 0
    while i < FRAME_SIZE:
        host.frame[i] = snap.frame[i]
        i += 1
    host.frag_ptr = snap.frag_ptr
    host.spill_ptr = snap.spill_ptr


class DSIntMetaStack(DSMetaStack):
    """Integer data stack in the three-tier layout: t0, t1 scalars |
    frame[FRAME_SIZE] (both virtualizable) | one shared spill array. The spill is
    allocated once per VM; a fragment is just the window [0, spill_ptr) onto it,
    so nest/unnest allocates nothing."""

    def init_fields(self):
        init_fields(self)

    # ------------------------------------------------------------------
    # Hot path. The top NTOP cells are scalars, so dup/swap/+/- (which touch
    # only the top 1-2 cells) compile to pure register arithmetic. The frame
    # array is reached only past depth NTOP, and the spill only past ACTIVE_MAX.
    # ------------------------------------------------------------------
    def push_on(self, v):
        dd = self.cache_depth
        if dd >= ACTIVE_MAX:
            self._spill_bottom()
            dd = self.cache_depth
        if dd >= NTOP:
            si = dd - NTOP
            assert si >= 0
            self.frame[si] = self.t1
        self.t1 = self.t0
        self.t0 = v
        self.cache_depth = dd + 1

    def pop_on(self):
        dd = self.cache_depth
        if dd <= 0:
            return self._pop_from_spill()
        r = self.t0
        self.t0 = self.t1
        if dd > NTOP:
            si = dd - NTOP - 1
            assert si >= 0
            self.t1 = self.frame[si]
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
        # than ACTIVE_MAX.
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

    def peek_on(self, depth):
        depth = promote(depth)
        dd = self.cache_depth
        if depth < dd:
            if depth == 0:
                return self.t0
            if depth == 1:
                return self.t1
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
            if depth == 0:
                self.t0 = v
            elif depth == 1:
                self.t1 = v
            else:
                si = dd - 1 - depth
                assert si >= 0
                self.frame[si] = v
            return
        ai = self.spill_ptr - 1 - (depth - dd)
        assert ai >= 0
        self.spill[ai] = v

    def depth_on(self):
        return self.cache_depth + self.spill_ptr

    def reset_on(self):
        self.t0 = 0
        self.t1 = 0
        self.cache_depth = 0
        self.frag_ptr = 0
        self.spill_ptr = 0

    # ------------------------------------------------------------------
    # Call entry / return. On a call the caller's below-NTOP cells are parked in
    # the spill and the active depth is normalized to the NTOP scalar tops, so
    # the callee runs with a small, call-local fragment. The tops themselves
    # flow into the callee for free (they are the conservative argument window).
    # Return is O(1): the spill already holds the caller's cells in place.
    # ------------------------------------------------------------------
    @unroll_safe
    def push_fragment_on(self):
        self.frag_ptr = self.frag_ptr + 1
        dd = self.cache_depth
        if dd > NTOP:
            n = dd - NTOP
            ap = self.spill_ptr
            if ap + n > SPILL_SIZE:
                raise DataStackOverflow()
            assert ap >= 0
            # park the below-NTOP frame cells; frame[i] (deepest at i=0) lands at
            # spill[ap+i], so the shallowest stays at the spill top (depth NTOP).
            i = 0
            while i < n:
                self.spill[ap + i] = self.frame[i]
                i += 1
            self.spill_ptr = ap + n
            self.cache_depth = NTOP

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
