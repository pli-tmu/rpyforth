# Integer data-stack layout: a scalar-tops cache over a shared spill

The integer data stack (enabled with `RPYFORTH_STACK_FRAGMENT`) is split into
three tiers. The top few cells live in CPU registers, the next few in a small
array, and everything deeper in one plain-heap "spill" arena shared by the whole
call chain. Only the top two tiers are virtualized, so the hot stack operations
compile to register arithmetic with no memory traffic.

```
   NTOP = 2          FRAME_SIZE = 8        ACTIVE_MAX = NTOP + FRAME_SIZE = 10
   CALL_WINDOW = NTOP = 2                  SPILL_SIZE = 16384 cells
```

A "fragment" here is not an allocated object. It is the implicit window
`[0, spill_ptr)` onto the single shared spill array, plus the cached tops above
it. Nesting a call parks some cells and bumps two integer pointers; it allocates
nothing. An earlier design allocated a fragment node per call, linked to its
parent; the *History* section below covers why that changed.

## Where and how fragments are managed

Three layers, from policy down to mechanics:

1. When to push and pop lives in `rpyforth/inner_interp.py`, at call boundaries.
   `execute_thread` calls `push_ds_fragments(self)` on portal entry and on every
   colon-word call; `pop_call` calls `pop_ds_fragments_commit(self)` on every
   return. All of this is gated by `USE_STACK_FRAGMENT`.
2. The dispatch wrappers are in `rpyforth/metastack.py`: `push_ds_fragments`,
   `pop_ds_fragments_commit`, and `reset_ds_fragments` forward to the active
   metastack's methods and also drive the float fragment when it is enabled.
3. The mechanics are in `rpyforth/metastack_int.py`: `push_fragment_on`,
   `pop_fragment_commit_on`, and the `push_on` / `pop_on` / `peek_on` / `poke_on`
   hot path. Ablation variants live in `metastack_int_ntop.py` (parametric NTOP)
   and `metastack_int_frameonly.py` (NTOP=0).

The stack state is a flat set of host fields, installed by `init_fields`
(`metastack_int.py`): the scalar tops `t0`, `t1`; the cached depth `cache_depth`;
the fixed array `frame[FRAME_SIZE]`; the shared `spill[SPILL_SIZE]`; and two
pointers, `spill_ptr` and `frag_ptr`. The spill is allocated once per VM.

The two pointers have distinct jobs, and keeping them separate is what keeps
nest and unnest cheap:

- `spill_ptr` is the only pointer that decides where cells live. It counts how
  many cells are parked in the spill, so it marks the boundary between the parked
  caller cells (`spill[0 .. spill_ptr-1]`) and the live cache above it. Every deep
  read, every park, and every restore is written in terms of it.
- `frag_ptr` is just a nesting counter, the call depth. `push_fragment_on`
  increments it, `pop_fragment_commit_on` decrements it (with a `>= 0` balance
  assert), and CATCH snapshots it. It never indexes the spill and never affects
  where a cell is stored.

So managing a fragment is not managing an object. A fragment is the window
`[0, spill_ptr)` onto the shared spill, plus the cached tops above it. Entering a
call parks the caller's below-window cells and normalizes the cache; returning
just unwinds the counter. Both directions move only these two pointers, plus a
copy of a few cells when a call overflows the window. No node is allocated or
freed, which is why deep recursion stays cheap.

## The three tiers

A stack holding 14 values (pushed `0,1,...,13`, so `13` is on top):

```
        TOP OF STACK  (depth 0)
              │
   depth │  location      value      tier
   ──────┼─────────────────────────────────────────────────────────────
     0   │  t0       =     13     ┐ scalar tops           ┐
     1   │  t1       =     12     ┘ (CPU registers)       │  ACTIVE CACHE
   ──────┼───────────────────────                         │  virtualized:
     2   │  frame[7] =     11     ┐                        │  kept in registers
     3   │  frame[6] =     10     │ small frame array      │  across the JIT loop,
     …   │    …                   │ (virtualizable)        │  reloaded only at
     8   │  frame[1] =      5     │                        │  the loop header
     9   │  frame[0] =      4     ┘ cache_depth = 10       ┘
   ──────┼───────────────────────────────────────────────────────────────
    10   │  spill[3] =      3     ┐                        ┐
    11   │  spill[2] =      2     │ contiguous shared      │  SPILL ARENA
    12   │  spill[1] =      1     │ arena (plain heap)     │  "other places":
    13   │  spill[0] =      0     ┘ spill_ptr = 4          │  NOT virtualized
   ──────┴───────────────────────────────────────────────┘  (too large)
        BOTTOM OF STACK
```

Invariants:

```
   logical depth      = cache_depth + spill_ptr           (here 10 + 4 = 14)
   t0                 = top of stack (TOS)
   t1                 = next of stack (NOS)
   frame[0]           = deepest CACHED cell
   frame[fsp-1]       = shallowest cached frame cell   (fsp = cache_depth - NTOP)
   spill[spill_ptr-1] = the cell just BELOW the cache (contiguous, no gap)
```

`cache_depth` is the number of cells currently held in the cache (0..ACTIVE_MAX).
It is a virtualizable field, so inside a trace it is a loop-carried SSA value
rather than a memory load. In JIT logs it shows up as `inst_cache_depth`, and the
depth-tier comparisons on it (`cache_depth <= 0`, `cache_depth > NTOP`) are the
guards you see at a trace's first stack op, before range analysis folds the rest.

## Where does the cell at `depth` live?

`peek(depth)` / `poke(depth)` pick a tier by comparing against `cache_depth`:

```
   depth == 0                          ->  t0                       (register)
   depth == 1                          ->  t1                       (register)
   2 <= depth <  cache_depth           ->  frame[cache_depth-1-depth]   (array)
   depth >= cache_depth                ->  spill[spill_ptr-1-(depth-cache_depth)]
```

The first two branches fold to a constant in the JIT because `dup`, `+`, `-`,
`swap`, `<`, `1+`, and friends carry a constant `depth`, so they never touch the
array.

## push / pop (the hot path)

`push` shifts everything down one slot (depth `k` becomes `k+1`); `pop` shifts up.
Only the scalar tops move in the common shallow case:

```
   push v   (cache not full):                    pop  ->  r
   ┌─────────────────────────────┐               ┌────────────────────────────────┐
   │ if cache_depth >= NTOP:      │               │ r  = t0                        │
   │     frame[cache_depth-NTOP]  │  displaced    │ t0 = t1                        │
   │              = t1            │  NOS spills   │ if cache_depth > NTOP:          │
   │ t1 = t0                      │  into frame   │     t1 = frame[cache_depth      │
   │ t0 = v                       │               │              -NTOP-1]          │
   │ cache_depth += 1             │               │ cache_depth -= 1 ; return r    │
   └─────────────────────────────┘               └────────────────────────────────┘
```

Overflow is cold: when `cache_depth == ACTIVE_MAX`, `push` first evacuates the
deepest cached cell (`frame[0]`) into the spill (`spill[spill_ptr++] = frame[0]`,
then slide the frame down) before it proceeds. Underflow past `cache_depth == 0`
reads straight from the spill top. Neither happens for stacks that stay within 10
cells.

## A call: park + normalize (and an O(1) return)

On a non-tail call the caller's below-window cells are parked in the spill and
the cache is normalized to the `NTOP` scalar tops. The tops themselves *are* the
callee's argument window, so they flow in for free, with no copy:

```
   BEFORE call  (caller, cache_depth = 5)      AFTER push_fragment (normalized)
   ─────────────────────────────────────      ─────────────────────────────────
   depth  cell                                 depth  cell
     0    t0       = e   ┐ tops                  0    t0       = e   ┐ callee's
     1    t1       = d   ┘                        1    t1       = d   ┘ args (free)
     2    frame[2] = c   ┐                      ───────────────────
     3    frame[1] = b   ├ frame                  2    spill[2] = c   ┐ caller cells,
     4    frame[0] = a   ┘                        3    spill[1] = b   ├ parked in the
   ───────────────────                            4    spill[0] = a   ┘ spill
   spill empty                                  ───────────────────
   cache_depth = 5, spill_ptr = 0               cache_depth = 2 (=NTOP), spill_ptr = 3
```

`push_fragment_on` (`metastack_int.py`) increments `frag_ptr`. Then, only if
`cache_depth > NTOP`, it copies the below-NTOP frame cells down into the spill,
advances `spill_ptr`, and sets `cache_depth = NTOP`. The callee now runs with a
tiny, call-local cache. Anything it reads below the tops falls through to the
spill; anything it pushes grows the cache again.

Return needs no copy-back. The spill already holds the caller's cells contiguously
below the callee's result, so `pop_fragment_commit` is O(1): it just decrements
`frag_ptr`. That is why deep recursion costs no per-call allocation.

`frag_ptr` is a plain nesting counter (call depth), not a pointer into a linked
structure. It is virtualized, balance-checked on commit, and snapshotted by
CATCH; it does not index the spill, since the spill window is defined by
`spill_ptr`.

## CATCH and snapshot / restore

CATCH saves and restores the whole cache alongside the call stack. In
`catch_push_frame` (`inner_interp.py`) the scalar tops, `cache_depth`, `frag_ptr`,
`spill_ptr`, and the frame row are copied into per-frame arrays (`ca_t0`, `ca_t1`,
`ca_cache_depth`, `ca_frag`, `ca_spill`, `ca_frames`). The spill contents below
`spill_ptr` are left in place and stay valid, because nothing overwrites them.
`throw_unwind` restores those fields, which rolls the logical stack back to the
CATCH point no matter how deep the protected word grew.

`snapshot_cache` / `restore_cache` (`metastack_int.py`) give the same
save/restore as a value object (`DSCacheSnapshot`) for non-CATCH uses. They copy
the fixed-size frame and record the two pointers, and restore rolls `spill_ptr`
back without touching the cells below it.

## History: from a linked list of fragments to a shared spill

The original design made each call allocate a *fragment* node that pointed at its
parent, forming a linked list down the call chain. That cost a heap allocation per
call (GC traffic) and a pointer chase per deep access. It also could not be
virtualized at all: a linked list of heap nodes is opaque to the meta-tracing
JIT, which was the decisive problem.

The current design keeps the conceptual fragment, a call-local window, but backs
every fragment with one shared, pre-allocated `spill` array. A fragment is just
the range `[0, spill_ptr)`, and nest/unnest only moves `spill_ptr` and `frag_ptr`.
Nothing is allocated per call, and the live cache is a small fixed set of
virtualizable fields the JIT can hold in registers across a trace.

The linked-list types survive only as vestigial stubs kept for import
compatibility: `DSFragment` (`metastack.py`), `DSIntFragment` (`metastack_int.py`,
with a `parent` field), and `DSObjFragment` (`metastack_obj.py`). None are
instantiated on the live integer path.

## Configuration and variants

Flag resolution lives at the top of `rpyforth/metastack.py`. `USE_STACK_FRAGMENT`
is the master switch; with it off, the interpreter uses the plain array-backed
stacks. Above it, the int cache has three mutually exclusive layouts, resolved in
priority order (frame-only, then parametric-NTOP, then default), plus one optional
addon:

- `default`: the two scalar tops `t0`, `t1` plus `frame[*]`, as described above.
- `frame-only` (`RPYFORTH_FRAME_ONLY=1`, conceptually NTOP=0,
  `metastack_int_frameonly.py`): no scalar tops. Every cached cell lives in the
  virtualizable `frame[*]` array, so push/pop move no data, just an index. This
  mirrors PyPy's own `fastlocals_w[*]`.
- `parametric-NTOP` (`RPYFORTH_NTOP` set to 2/4/8/16, `metastack_int_ntop.py`):
  generalizes the scalar-tops count to `EFFECTIVE_NTOP`. The call window stays
  `CALL_WINDOW = NTOP = 2` so the calling convention matches across every NTOP
  value.
- `float fragment` (`RPYFORTH_FLOAT_FRAGMENT=1`): mirrors this layout for the
  float stack. It rides only on the default int layout and measured a net loss, so
  it stays opt-in for ablation.

`FRAME_SIZE` is configurable via `RPYFORTH_FRAME_SIZE` (default 8, clamped to
[1, 64]).

The virtualizable field set the JIT tracks is assembled in `metastack.py` as
`STACK_FRAGMENT_VIRTUALIZABLES`: the int cache fields (`t0`, `t1`, `cache_depth`,
`frame[*]`, `frag_ptr`, `spill_ptr`), the return- and call-stack pointers, and the
loop and cell-size state. The `spill` array itself is not in this set. It is one
immutable-reference array, too large to virtualize, and it is touched only when
the stack goes deeper than the roughly 10-cell cache.

## Why this shape

```
   register tops  ──►  hot ops (top 1-2 cells) are register-only, zero memory
   small frame[8] ──►  a bit more working depth, still a small virtualizable
                       array the trace optimizer can track cheaply
   shared spill   ──►  unbounded depth lives here; a LARGE virtualizable array
                       poisons the optimizer, so the deep store stays plain heap
```

Two design constants were tuned against the trace:

- `FRAME_SIZE` is small (8, not 64+): a virtualizable `[*]` array is tracked
  slot-by-slot and reloaded at the loop header, so a large one is pathological.
- `NTOP` is 2, not 3: a third scalar top widened the push/pop shift and added a
  guard-spilled register, which measurably slowed push/pop-heavy DO-loop setup,
  for a gain only words reaching the 3rd cell (such as `rot`) would ever see.

The net effect: the shallow, hot part of the stack is pure register arithmetic
inside a trace, deep and recursive stacks cost no per-call allocation, and only
the rare deep access falls through to plain-heap memory.
