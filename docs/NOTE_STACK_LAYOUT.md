# Integer data-stack layout: a scalar-tops cache over a metastack arena

The integer data stack (enabled with `RPYFORTH_STACK_FRAGMENT`) is split into
three tiers. The top few cells live in CPU registers, the next few in a small
array, and everything deeper in a plain-heap "metastack" arena. Only the top two
tiers are virtualized, so the hot stack operations compile to register
arithmetic with no memory traffic.

```
   NTOP = 2          FRAME_SIZE = 8        ACTIVE_MAX = NTOP + FRAME_SIZE = 10
```

## The three tiers

A stack holding 14 values (pushed `0,1,...,13`, so `13` is on top):

```
        TOP OF STACK  (depth 0)
              │
   depth │  location      value      tier
   ──────┼─────────────────────────────────────────────────────────────
     0   │  t0       =     13     ┐ scalar tops          ┐
     1   │  t1       =     12     ┘ (CPU registers)      │  ACTIVE FRAGMENT
   ──────┼───────────────────────                        │  virtualized:
     2   │  frame[7] =     11     ┐                       │  kept in registers
     3   │  frame[6] =     10     │ small spill array     │  across the JIT loop,
     …   │    …                   │ (virtualizable)       │  reloaded only at
     8   │  frame[1] =      5     │                       │  the loop header
     9   │  frame[0] =      4     ┘ d = 10  (cache full)  ┘
   ──────┼───────────────────────────────────────────────────────────────
    10   │  spill[3] =      3     ┐                       ┐
    11   │  spill[2] =      2     │ contiguous arena       │  METASTACK
    12   │  spill[1] =      1     │ (plain heap)           │  "other places":
    13   │  spill[0] =      0     ┘ spill_ptr = 4          │  NOT virtualized
   ──────┴───────────────────────────────────────────────┘  (too large)
        BOTTOM OF STACK
```

Invariants:

```
   logical depth   = d + spill_ptr                 (here 10 + 4 = 14)
   t0              = top of stack (TOS)
   t1              = next of stack (NOS)
   frame[0]        = deepest CACHED cell
   frame[fsp-1]    = shallowest cached frame cell   (fsp = d - NTOP)
   spill[spill_ptr-1] = the cell just BELOW the cache (contiguous, no gap)
```

## Where does cell at `depth` live?

`peek(depth)` / `poke(depth)` pick a tier by comparing against `d`:

```
   depth == 0                 ->  t0                                  (register)
   depth == 1                 ->  t1                                  (register)
   2 <= depth <  d            ->  frame[d - 1 - depth]                (small array)
   depth >= d                 ->  spill[spill_ptr - 1 - (depth - d)]  (heap arena)
```

The first two branches fold to a constant in the JIT because `dup`, `+`, `-`,
`swap`, `<`, `1+`, … carry a constant `depth`, so they never touch the array.

## push / pop (the hot path)

`push` shifts everything down one slot (depth `k` -> `k+1`); `pop` shifts up.
Only the scalar tops move in the common (shallow) case:

```
   push v   (cache not full):                 pop  ->  r
   ┌──────────────────────────┐               ┌──────────────────────────┐
   │ if d >= NTOP:             │               │ r  = t0                  │
   │     frame[d-NTOP] = t1    │  displaced    │ t0 = t1                  │
   │ t1 = t0                   │  NOS spills   │ if d > NTOP:             │
   │ t0 = v                    │  into frame   │     t1 = frame[d-NTOP-1] │
   │ d += 1                    │               │ d -= 1 ; return r        │
   └──────────────────────────┘               └──────────────────────────┘
```

Overflow is cold: when `d == ACTIVE_MAX`, `push` first evacuates the deepest
cached cell (`frame[0]`) into the arena (`spill[spill_ptr++] = frame[0]`, slide
the frame down), then proceeds. Underflow past `d == 0` reads straight from the
arena top. Neither happens for stacks that stay within 10 cells.

## A call: park + normalize (and an O(1) return)

On a non-tail call the caller's below-top cells are parked in the arena and the
cache is normalized to the `NTOP` scalar tops. The tops themselves *are* the
callee's argument window, so they flow in for free (no copy):

```
   BEFORE call  (caller, d = 5)                AFTER push_fragment (normalized)
   ───────────────────────────                ─────────────────────────────────
   depth  cell                                 depth  cell
     0    t0       = e   ┐ tops                  0    t0       = e   ┐ callee's
     1    t1       = d   ┘                        1    t1       = d   ┘ args (free)
     2    frame[2] = c   ┐                      ───────────────────
     3    frame[1] = b   ├ frame                  2    spill[2] = c   ┐ caller frame,
     4    frame[0] = a   ┘                        3    spill[1] = b   ├ parked in the
   ───────────────────                            4    spill[0] = a   ┘ arena
   arena empty                                  ───────────────────
   d = 5, spill_ptr = 0                         d = 2 (= NTOP), spill_ptr = 3
```

The callee now runs with a tiny, call-local fragment (`d` starts at `NTOP`).
Anything it reads below the tops falls through to the arena; anything it pushes
grows the cache again.

Return needs no copy-back. The arena already holds the caller's cells
contiguously below the callee's result, so `pop_fragment_commit` is **O(1)** — it
just unwinds the call counter. This is the property that removed the per-call
allocation that used to make recursion slow.

## Why this shape

```
   register tops  ──►  hot ops (top 1–2 cells) are register-only, zero memory
   small frame[8] ──►  a bit more working depth, still a small virtualizable
                       array the trace optimizer can track cheaply
   heap arena     ──►  unbounded depth lives here; a LARGE virtualizable array
                       segfaults the optimizer, so the deep store stays plain heap
```

Two design constants were tuned against the trace:

- **`FRAME_SIZE` small** (8, not 64+): a virtualizable `[*]` array is tracked
  slot-by-slot and reloaded at the loop header, so a large one is pathological.
- **`NTOP = 2`, not 3**: a third scalar top widened the push/pop shift and added
  a guard-spilled register, which measurably slowed push/pop-heavy DO-loop setup
  for a gain only words reaching the 3rd cell (e.g. `rot`) would see.

Result: every shootout benchmark runs faster than the default virtualized build,
recursion included (`fibo` ~1.06×, `ack` ~1.55×, `nestedloop` ~1.8×).
