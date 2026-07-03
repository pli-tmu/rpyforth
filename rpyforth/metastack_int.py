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
    STACK_FRAGMENT_STRICT,
)


class DSIntFragment(DSFragment):
    """Legacy value object, retained only for import compatibility."""

    _immutable_fields_ = ["parent"]

    def __init__(self, parent):
        self.parent = parent


def init_fields(host):
    """Install the host-resident active-fragment + metastack arena state."""
    # The active fragment caches the TOP of the data stack:
    #   * the top NTOP cells in scalar fields t0, t1 (hottest, in registers)
    #   * the next cells in the small virtualizable spill array ``frame``
    # ``d`` is the number of cells currently cached (0..ACTIVE_MAX).
    host.t0 = 0
    host.t1 = 0
    host.d = 0
    host.frame = [0] * FRAME_SIZE
    make_sure_not_resized(host.frame)

    # The metastack arena holds every cell BELOW the cached fragment -- the
    # caller frames parked on a call plus the rare single-word overflow ("other
    # places"). Contiguous: spill[spill_ptr-1] is the cell just under the cache,
    # spill[0] the bottom of the stack. Plain heap (immutable reference): it is
    # sized to the whole stack depth, too large to virtualize. frag_ptr just
    # tracks call nesting for the metastack.
    host.frag_ptr = 0
    host.spill = [0] * SPILL_SIZE
    make_sure_not_resized(host.spill)
    host.spill_ptr = 0


class DSCacheSnapshot(object):
    """Immutable capture of the active-fragment cache, for saving and restoring
    the data stack. Holds the scalar tops, the cached depth, a private copy of the
    frame array, and the cache/arena pointers. The arena buffer is not copied:
    restore rolls the spill pointer back (discarding cells parked above it) and
    relies on the cells below it being undisturbed."""

    _immutable_fields_ = ["t0", "t1", "d", "frame[*]", "frag_ptr", "spill_ptr"]

    def __init__(self, t0, t1, d, frame, frag_ptr, spill_ptr):
        self.t0 = t0
        self.t1 = t1
        self.d = d
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
    return DSCacheSnapshot(host.t0, host.t1, host.d, frame_copy,
                           host.frag_ptr, host.spill_ptr)


def restore_cache(host, snap):
    """Roll host's stack back to a snapshot, discarding everything pushed since.
    Restores the cache and the cache/arena pointers; the arena cells below the
    saved spill pointer are left in place."""
    host.t0 = snap.t0
    host.t1 = snap.t1
    host.d = snap.d
    i = 0
    while i < FRAME_SIZE:
        host.frame[i] = snap.frame[i]
        i += 1
    host.frag_ptr = snap.frag_ptr
    host.spill_ptr = snap.spill_ptr


class DSIntMetaStack(DSMetaStack):
    def init_fields(self):
        init_fields(self)

    # ------------------------------------------------------------------
    # Hot path. The top NTOP cells are scalars, so dup/swap/+/- (which touch
    # only the top 1-2 cells) compile to pure register arithmetic. The frame
    # array is reached only past depth NTOP, and the arena only past ACTIVE_MAX.
    # ------------------------------------------------------------------
    def push_on(self, v):
        dd = self.d
        if dd >= ACTIVE_MAX:
            # cache full: evacuate its deepest cell to the arena to free a slot
            self._spill_bottom()
            dd = self.d
        if dd >= NTOP:
            si = dd - NTOP
            assert si >= 0
            self.frame[si] = self.t1
        self.t1 = self.t0
        self.t0 = v
        self.d = dd + 1

    def pop_on(self):
        dd = self.d
        if dd <= 0:
            return self._pop_from_arena()
        r = self.t0
        self.t0 = self.t1
        if dd > NTOP:
            si = dd - NTOP - 1
            assert si >= 0
            self.t1 = self.frame[si]
        self.d = dd - 1
        return r

    def _pop_from_arena(self):
        # Cache empty: the top now lives in the arena (a callee consumed past its
        # imported window into parent cells).
        ap = self.spill_ptr - 1
        if ap < 0:
            assert not STACK_FRAGMENT_STRICT
            assert False, "integer stack underflow"
        assert ap >= 0
        r = self.spill[ap]
        self.spill_ptr = ap
        return r

    @unroll_safe
    def _spill_bottom(self):
        # Move the deepest cached cell (frame[0]) to the arena and slide the
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
        self.d = self.d - 1

    def peek_on(self, depth):
        depth = promote(depth)
        dd = self.d
        if depth < dd:
            if depth == 0:
                return self.t0
            if depth == 1:
                return self.t1
            si = dd - 1 - depth
            assert si >= 0
            return self.frame[si]
        ai = self.spill_ptr - 1 - (depth - dd)
        if ai < 0:
            assert not STACK_FRAGMENT_STRICT
            assert False, "integer stack underflow"
        assert ai >= 0
        return self.spill[ai]

    def poke_on(self, depth, v):
        depth = promote(depth)
        dd = self.d
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
        if ai < 0:
            assert not STACK_FRAGMENT_STRICT
            assert False, "integer stack underflow"
        assert ai >= 0
        self.spill[ai] = v

    def depth_on(self):
        # Full logical depth: cached cells plus everything parked in the arena.
        return self.d + self.spill_ptr

    def reset_on(self):
        self.t0 = 0
        self.t1 = 0
        self.d = 0
        self.frag_ptr = 0
        self.spill_ptr = 0

    # ------------------------------------------------------------------
    # Call entry / return. On a call the caller's below-NTOP cells are parked in
    # the arena and the active depth is normalized to the NTOP scalar tops, so
    # the callee runs with a small, call-local fragment. The tops themselves
    # flow into the callee for free (they are the conservative argument window).
    # Return is O(1): the arena already holds the caller's cells in place.
    # ------------------------------------------------------------------
    @unroll_safe
    def push_fragment_on(self):
        self.frag_ptr = self.frag_ptr + 1
        dd = self.d
        if dd > NTOP:
            n = dd - NTOP
            ap = self.spill_ptr
            if ap + n > SPILL_SIZE:
                raise DataStackOverflow()
            assert ap >= 0
            # park the below-NTOP frame cells; frame[i] (deepest at i=0) lands at
            # spill[ap+i], so the shallowest stays at the arena top (depth NTOP).
            i = 0
            while i < n:
                self.spill[ap + i] = self.frame[i]
                i += 1
            self.spill_ptr = ap + n
            self.d = NTOP

    def pop_fragment_commit_on(self):
        # O(1): the callee's net result is already the cache top and the caller's
        # parked cells are already the arena top, contiguously below it. Nothing
        # to move -- just unwind the call counter.
        if self.frag_ptr > 0:
            self.frag_ptr = self.frag_ptr - 1

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
