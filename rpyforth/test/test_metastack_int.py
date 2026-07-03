from rpyforth.metastack import (
    NTOP,
    FRAME_SIZE,
    ACTIVE_MAX,
    CALL_WINDOW,
    SPILL_SIZE,
    STACK_FRAGMENT_VIRTUALIZABLES,
)
from rpyforth.metastack_int import DSIntMetaStack, init_fields


class _Host(object):
    pass


def test_init_fields_sets_virtualizable_host_slots():
    host = _Host()
    init_fields(host)
    assert host.t0 == 0
    assert host.t1 == 0
    assert host.d == 0
    assert host.frag_ptr == 0
    assert host.spill_ptr == 0
    assert len(host.frame) == FRAME_SIZE
    assert len(host.spill) == SPILL_SIZE


def test_stack_fragment_virtualizables_include_tops_and_frame():
    # Tops, cache depth, frame array and the changing stack pointers.
    for name in (
        "t0", "t1", "d", "frame[*]",
        "frag_ptr", "spill_ptr",
        "rs_ptr", "cs_tids", "cs_ips", "cs_ptr",
    ):
        assert name in STACK_FRAGMENT_VIRTUALIZABLES
    # The arena and the immutable array references stay out.
    for name in ("spill", "spill[*]", "rs", "ds_locals"):
        assert name not in STACK_FRAGMENT_VIRTUALIZABLES


def test_int_push_pop_peek():
    s = DSIntMetaStack()
    n = ACTIVE_MAX
    for v in range(n):
        s.push(v)
    assert s.size() == n
    assert s.d == n
    assert s.peek(0) == n - 1
    assert s.peek(n - 1) == 0
    assert [s.pop() for _ in range(n)] == list(range(n - 1, -1, -1))
    assert s.size() == 0


def test_int_deep_peek_poke():
    # Fill tops + frame, then poke/peek across the scalar/array boundary.
    s = DSIntMetaStack()
    n = ACTIVE_MAX
    for v in range(n):
        s.push(v)
    s.poke(n - 1, 424242)        # deepest (in the frame array)
    s.poke(0, 99)                # top (a scalar)
    assert s.peek(n - 1) == 424242
    assert s.peek(0) == 99
    out = [s.pop() for _ in range(n)]
    assert out[0] == 99
    assert out[-1] == 424242


def test_int_arena_spill_beyond_active_max():
    # Pushing past the cache spills the deepest cells into the arena -- no error,
    # and the values survive peek/pop through the cache/arena boundary.
    s = DSIntMetaStack()
    n = 2 * ACTIVE_MAX + 3
    for v in range(n):
        s.push(v)
    assert s.size() == n
    assert s.d == ACTIVE_MAX
    assert s.spill_ptr == n - ACTIVE_MAX
    assert s.peek(0) == n - 1            # top, in the cache
    assert s.peek(ACTIVE_MAX) == n - 1 - ACTIVE_MAX   # just below the cache
    assert s.peek(n - 1) == 0            # bottom, deep in the arena
    assert [s.pop() for _ in range(n)] == list(range(n - 1, -1, -1))
    assert s.size() == 0
    assert s.spill_ptr == 0


def test_int_clear_resets():
    s = DSIntMetaStack()
    for v in range(5):
        s.push(v)
    s.clear()
    assert s.size() == 0
    assert s.d == 0
    assert s.frag_ptr == 0
    assert s.spill_ptr == 0
    s.push(42)
    s.push(43)
    assert s.size() == 2


def test_int_fragment_call_commit_within_window():
    # Caller depth fits the NTOP scalar tops: nothing is parked, the tops flow
    # into the callee for free.
    s = DSIntMetaStack()
    for v in range(NTOP):
        s.push(v)
    s.push_fragment()
    assert s.frag_ptr == 1
    assert s.spill_ptr == 0
    assert s.d == NTOP
    assert s.size() == NTOP
    assert s.peek(0) == NTOP - 1
    s.push(99)
    s.pop_fragment_commit()
    assert s.frag_ptr == 0
    assert s.size() == NTOP + 1
    assert s.peek(0) == 99
    assert [s.pop() for _ in range(NTOP + 1)] == [99] + list(range(NTOP - 1, -1, -1))


def test_int_fragment_call_commit_beyond_window():
    # Caller deeper than NTOP: the below-top cells are parked in the arena; the
    # callee runs with the cache normalized to NTOP, the total depth preserved.
    s = DSIntMetaStack()
    n = NTOP + 2
    for v in range(n):
        s.push(v)
    s.push_fragment()
    parked = n - NTOP
    assert s.frag_ptr == 1
    assert s.d == NTOP                   # cache normalized to the tops
    assert s.spill_ptr == parked         # below-top cells parked in the arena
    assert s.size() == n                 # total depth preserved
    # callee sees the top NTOP cells in the cache
    assert s.peek(0) == n - 1
    assert s.peek(NTOP - 1) == n - NTOP
    # below the tops: fall through into the parked arena
    assert s.peek(NTOP) == n - NTOP - 1
    assert s.peek(n - 1) == 0
    s.pop_fragment_commit()
    assert s.frag_ptr == 0
    assert s.size() == n
    assert [s.pop() for _ in range(n)] == list(range(n - 1, -1, -1))
    assert s.spill_ptr == 0


def test_int_below_window_poke_then_commit():
    # Poking below the cache writes the parked arena, and the write must survive
    # commit.
    s = DSIntMetaStack()
    n = NTOP + 2
    for v in range(n):
        s.push(v)
    s.push_fragment()
    s.poke(n - 1, 777)           # deepest caller cell, parked in the arena
    assert s.peek(n - 1) == 777
    s.pop_fragment_commit()
    assert s.peek(n - 1) == 777


def test_int_nested_fragments_bounded_entry_depth():
    # Deep nesting: every entry normalizes the cache to at most NTOP, so the
    # active (virtualized) depth stays tiny while the arena absorbs the rest.
    s = DSIntMetaStack()
    depth = 40
    for v in range(depth):
        s.push(v)
        s.push_fragment()
        assert s.d <= NTOP
    assert s.frag_ptr == depth
    assert s.size() == depth
    for _ in range(depth):
        s.pop_fragment_commit()
    assert s.frag_ptr == 0
    # the full stack is still intact, served from cache + arena
    assert s.size() == depth
    assert [s.pop() for _ in range(depth)] == list(range(depth - 1, -1, -1))
    assert s.spill_ptr == 0


def test_int_fragment_consume_all_args():
    # A callee that consumes its whole window and produces nothing leaves the
    # stack empty.
    s = DSIntMetaStack()
    for v in range(NTOP):
        s.push(v)
    s.push_fragment()
    for _ in range(NTOP):
        s.pop()                  # callee drains the cache
    assert s.size() == 0
    s.pop_fragment_commit()
    assert s.size() == 0
    assert s.frag_ptr == 0


# ------------------------------------------------------------------
# snapshot / restore. snapshot() captures the active cache (scalar tops + frame
# + the cache/arena pointers); restore() puts the stack back to that captured
# state, discarding anything pushed since. The arena cells below the saved spill
# pointer are assumed undisturbed (true for code that does not pop below the
# snapshot depth).
# ------------------------------------------------------------------
def test_snapshot_restore_identity():
    s = DSIntMetaStack()
    for v in range(5):
        s.push(v)
    snap = s.snapshot()
    s.restore(snap)
    assert s.size() == 5
    assert [s.pop() for _ in range(5)] == [4, 3, 2, 1, 0]


def test_snapshot_restore_within_cache():
    # Depth inside tops + frame (no arena). Mutate after snapshot, then restore.
    s = DSIntMetaStack()
    for v in range(NTOP + 3):
        s.push(v)
    snap = s.snapshot()
    before = [s.peek(i) for i in range(s.size())]
    s.push(100)
    s.push(101)
    s.pop()
    s.poke(0, 999)
    s.restore(snap)
    assert s.size() == NTOP + 3
    assert [s.peek(i) for i in range(s.size())] == before


def test_restore_discards_pushed_values():
    s = DSIntMetaStack()
    for v in range(NTOP + 1):
        s.push(v)
    d0 = s.size()
    snap = s.snapshot()
    for v in range(50):
        s.push(v)
    s.restore(snap)
    assert s.size() == d0
    assert s.peek(0) == NTOP


def test_snapshot_is_independent_copy():
    # Overwriting every cached cell after the snapshot must not affect it.
    s = DSIntMetaStack()
    for v in range(ACTIVE_MAX):
        s.push(v)
    snap = s.snapshot()
    for i in range(ACTIVE_MAX):
        s.poke(i, -1)
    s.restore(snap)
    assert [s.peek(i) for i in range(ACTIVE_MAX)] == [ACTIVE_MAX - 1 - i for i in range(ACTIVE_MAX)]


def test_snapshot_restore_across_arena():
    # Depth past ACTIVE_MAX so cells live in the arena. Churn (more spill + pops)
    # then restore must rebuild the exact stack through the cache/arena boundary.
    s = DSIntMetaStack()
    n = 2 * ACTIVE_MAX + 5
    for v in range(n):
        s.push(v)
    snap = s.snapshot()
    before = [s.peek(i) for i in range(s.size())]
    for v in range(ACTIVE_MAX + 7):
        s.push(1000 + v)
    for _ in range(3):
        s.pop()
    s.restore(snap)
    assert s.size() == n
    assert [s.peek(i) for i in range(s.size())] == before
    assert [s.pop() for _ in range(n)] == list(range(n - 1, -1, -1))
    assert s.spill_ptr == 0


def test_snapshot_restore_nested():
    # Nested CATCH frames: snapshots form a stack, each restoring its own depth.
    s = DSIntMetaStack()
    for v in range(NTOP + 1):
        s.push(v)
    outer = s.snapshot()
    for v in range(ACTIVE_MAX):
        s.push(50 + v)
    inner = s.snapshot()
    for v in range(20):
        s.push(900 + v)
    s.restore(inner)
    assert s.size() == NTOP + 1 + ACTIVE_MAX
    assert s.peek(0) == 50 + ACTIVE_MAX - 1
    s.restore(outer)
    assert s.size() == NTOP + 1
    assert s.peek(0) == NTOP
