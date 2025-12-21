"""
Virtual stack implementation for RPython JIT allocation removal.

The key insight: RPython can only virtualize small, fixed-size arrays.
A 4096-element array defeats allocation removal because:
1. The JIT can't track all elements
2. Dynamic indexing causes escape

Solution: Use a small "virtual window" (e.g., 8 elements) that the JIT
can fully virtualize, with overflow/underflow to a backing store.

The window acts as a cache for the top of stack - most Forth operations
only touch the top 2-4 elements, so this is highly effective.
"""

from rpython.rlib.jit import JitDriver, promote, elidable, unroll_safe, hint

# Virtual window size - small enough for full virtualization
# Must be power of 2 for efficient modulo
VSTACK_WINDOW_SIZE = 8

# Backing store size
VSTACK_BACKING_SIZE = 4096


class VirtualIntStack(object):
    """
    Integer stack with virtual window for JIT allocation removal.

    Layout:
    - window[0..VSTACK_WINDOW_SIZE-1]: virtualized top-of-stack cache
    - window_ptr: how many elements in window (0 to VSTACK_WINDOW_SIZE)
    - backing[]: overflow storage
    - backing_ptr: elements in backing store

    Total stack depth = backing_ptr + window_ptr

    Note: _virtualizable_ is NOT set here because RPython JIT requires
    virtualizables to be direct fields of the object in the jitdriver's reds.
    The virtualization happens at the InnerInterpreter level.
    """

    _immutable_fields_ = []

    def __init__(self):
        # The virtualized window - small fixed size
        self.window = [0] * VSTACK_WINDOW_SIZE
        self.window_ptr = 0

        # Non-virtualized backing store
        self.backing = [0] * VSTACK_BACKING_SIZE
        self.backing_ptr = 0

    def _virtualize_window(self):
        """Hint to ensure window is accessed virtually."""
        # This is called at trace entry points
        hint(self, access_directly=False)

    @unroll_safe
    def _spill_to_backing(self):
        """Move bottom half of window to backing store."""
        # Spill VSTACK_WINDOW_SIZE // 2 elements
        spill_count = VSTACK_WINDOW_SIZE // 2

        # Copy bottom elements to backing
        for i in range(spill_count):
            self.backing[self.backing_ptr + i] = self.window[i]
        self.backing_ptr += spill_count

        # Shift remaining elements down in window
        for i in range(spill_count, VSTACK_WINDOW_SIZE):
            self.window[i - spill_count] = self.window[i]

        self.window_ptr -= spill_count

    @unroll_safe
    def _fill_from_backing(self):
        """Refill window from backing store."""
        if self.backing_ptr == 0:
            return  # Nothing to fill from

        # How many to restore (up to half window)
        fill_count = min(self.backing_ptr, VSTACK_WINDOW_SIZE // 2)

        # Shift current window elements up
        for i in range(self.window_ptr - 1, -1, -1):
            self.window[i + fill_count] = self.window[i]

        # Copy from backing to bottom of window
        for i in range(fill_count):
            self.window[i] = self.backing[self.backing_ptr - fill_count + i]

        self.backing_ptr -= fill_count
        self.window_ptr += fill_count

    def push(self, value):
        """Push integer onto stack."""
        ptr = self.window_ptr

        # Check for window overflow
        if ptr >= VSTACK_WINDOW_SIZE:
            self._spill_to_backing()
            ptr = self.window_ptr

        self.window[ptr] = value
        self.window_ptr = ptr + 1

    def pop(self):
        """Pop integer from stack."""
        ptr = self.window_ptr - 1

        # Check for window underflow
        if ptr < 0:
            self._fill_from_backing()
            ptr = self.window_ptr - 1

        if ptr < 0:
            raise IndexError("Stack underflow")

        value = self.window[ptr]
        self.window_ptr = ptr
        return value

    def peek(self, depth=0):
        """Peek at stack element without removing. depth=0 is top."""
        depth = promote(depth)  # Hint that depth is often constant

        ptr = self.window_ptr - 1 - depth
        if ptr >= 0:
            # Element is in window
            return self.window[ptr]
        else:
            # Element is in backing store
            backing_idx = self.backing_ptr + ptr
            if backing_idx < 0:
                raise IndexError("Stack underflow in peek")
            return self.backing[backing_idx]

    def poke(self, depth, value):
        """Set element at depth without push/pop. depth=0 is top."""
        depth = promote(depth)

        ptr = self.window_ptr - 1 - depth
        if ptr >= 0:
            self.window[ptr] = value
        else:
            backing_idx = self.backing_ptr + ptr
            if backing_idx < 0:
                raise IndexError("Stack underflow in poke")
            self.backing[backing_idx] = value

    def depth(self):
        """Return total stack depth."""
        return self.backing_ptr + self.window_ptr

    def is_empty(self):
        """Check if stack is empty."""
        return self.window_ptr == 0 and self.backing_ptr == 0

    def clear(self):
        """Clear the stack."""
        self.window_ptr = 0
        self.backing_ptr = 0

    @unroll_safe
    def top2(self):
        """Optimized: pop two elements (y then x, so x was deeper)."""
        # Common case: both in window
        ptr = self.window_ptr
        if ptr >= 2:
            y = self.window[ptr - 1]
            x = self.window[ptr - 2]
            self.window_ptr = ptr - 2
            return x, y
        else:
            # Slow path
            y = self.pop()
            x = self.pop()
            return x, y

    @unroll_safe
    def dup(self):
        """Optimized DUP: duplicate top element."""
        ptr = self.window_ptr
        if ptr > 0 and ptr < VSTACK_WINDOW_SIZE:
            val = self.window[ptr - 1]
            self.window[ptr] = val
            self.window_ptr = ptr + 1
        else:
            # Slow path
            val = self.peek(0)
            self.push(val)

    @unroll_safe
    def swap(self):
        """Optimized SWAP: exchange top two elements."""
        ptr = self.window_ptr
        if ptr >= 2:
            tmp = self.window[ptr - 1]
            self.window[ptr - 1] = self.window[ptr - 2]
            self.window[ptr - 2] = tmp
        else:
            # Slow path - need to involve backing
            y = self.pop()
            x = self.pop()
            self.push(y)
            self.push(x)

    @unroll_safe
    def over(self):
        """Optimized OVER: copy second element to top."""
        ptr = self.window_ptr
        if ptr >= 2 and ptr < VSTACK_WINDOW_SIZE:
            val = self.window[ptr - 2]
            self.window[ptr] = val
            self.window_ptr = ptr + 1
        else:
            # Slow path
            val = self.peek(1)
            self.push(val)

    @unroll_safe
    def rot(self):
        """Optimized ROT: rotate top 3 (x1 x2 x3 -- x2 x3 x1)."""
        ptr = self.window_ptr
        if ptr >= 3:
            x3 = self.window[ptr - 1]
            x2 = self.window[ptr - 2]
            x1 = self.window[ptr - 3]
            self.window[ptr - 3] = x2
            self.window[ptr - 2] = x3
            self.window[ptr - 1] = x1
        else:
            # Slow path
            x3 = self.pop()
            x2 = self.pop()
            x1 = self.pop()
            self.push(x2)
            self.push(x3)
            self.push(x1)


class VirtualFloatStack(object):
    """Float stack with same virtual window pattern."""

    def __init__(self):
        self.window = [0.0] * VSTACK_WINDOW_SIZE
        self.window_ptr = 0
        self.backing = [0.0] * VSTACK_BACKING_SIZE
        self.backing_ptr = 0

    @unroll_safe
    def _spill_to_backing(self):
        spill_count = VSTACK_WINDOW_SIZE // 2
        for i in range(spill_count):
            self.backing[self.backing_ptr + i] = self.window[i]
        self.backing_ptr += spill_count
        for i in range(spill_count, VSTACK_WINDOW_SIZE):
            self.window[i - spill_count] = self.window[i]
        self.window_ptr -= spill_count

    @unroll_safe
    def _fill_from_backing(self):
        if self.backing_ptr == 0:
            return
        fill_count = min(self.backing_ptr, VSTACK_WINDOW_SIZE // 2)
        for i in range(self.window_ptr - 1, -1, -1):
            self.window[i + fill_count] = self.window[i]
        for i in range(fill_count):
            self.window[i] = self.backing[self.backing_ptr - fill_count + i]
        self.backing_ptr -= fill_count
        self.window_ptr += fill_count

    def push(self, value):
        ptr = self.window_ptr
        if ptr >= VSTACK_WINDOW_SIZE:
            self._spill_to_backing()
            ptr = self.window_ptr
        self.window[ptr] = value
        self.window_ptr = ptr + 1

    def pop(self):
        ptr = self.window_ptr - 1
        if ptr < 0:
            self._fill_from_backing()
            ptr = self.window_ptr - 1
        if ptr < 0:
            raise IndexError("Float stack underflow")
        value = self.window[ptr]
        self.window_ptr = ptr
        return value

    def peek(self, depth=0):
        depth = promote(depth)
        ptr = self.window_ptr - 1 - depth
        if ptr >= 0:
            return self.window[ptr]
        else:
            backing_idx = self.backing_ptr + ptr
            if backing_idx < 0:
                raise IndexError("Float stack underflow in peek")
            return self.backing[backing_idx]

    def depth(self):
        """Return total stack depth."""
        return self.backing_ptr + self.window_ptr

    def clear(self):
        """Clear the stack."""
        self.window_ptr = 0
        self.backing_ptr = 0


class VirtualReturnStack(object):
    """
    Return stack with virtual window for JIT allocation removal.

    Also supports loop operations where each loop frame is 2 cells:
    - limit (bottom)
    - counter (top)

    Loop depth 0 = innermost loop, depth 1 = next outer loop, etc.
    """

    def __init__(self):
        self.window = [0] * VSTACK_WINDOW_SIZE
        self.window_ptr = 0
        self.backing = [0] * VSTACK_BACKING_SIZE
        self.backing_ptr = 0

    @unroll_safe
    def _spill_to_backing(self):
        """Move bottom half of window to backing store."""
        spill_count = VSTACK_WINDOW_SIZE // 2
        for i in range(spill_count):
            self.backing[self.backing_ptr + i] = self.window[i]
        self.backing_ptr += spill_count
        for i in range(spill_count, VSTACK_WINDOW_SIZE):
            self.window[i - spill_count] = self.window[i]
        self.window_ptr -= spill_count

    @unroll_safe
    def _fill_from_backing(self):
        """Refill window from backing store."""
        if self.backing_ptr == 0:
            return
        fill_count = min(self.backing_ptr, VSTACK_WINDOW_SIZE // 2)
        for i in range(self.window_ptr - 1, -1, -1):
            self.window[i + fill_count] = self.window[i]
        for i in range(fill_count):
            self.window[i] = self.backing[self.backing_ptr - fill_count + i]
        self.backing_ptr -= fill_count
        self.window_ptr += fill_count

    def push(self, value):
        """Push value onto return stack."""
        ptr = self.window_ptr
        if ptr >= VSTACK_WINDOW_SIZE:
            self._spill_to_backing()
            ptr = self.window_ptr
        self.window[ptr] = value
        self.window_ptr = ptr + 1

    def pop(self):
        """Pop value from return stack."""
        ptr = self.window_ptr - 1
        if ptr < 0:
            self._fill_from_backing()
            ptr = self.window_ptr - 1
        if ptr < 0:
            raise IndexError("Return stack underflow")
        value = self.window[ptr]
        self.window_ptr = ptr
        return value

    def peek(self, depth=0):
        """Peek at stack element without removing. depth=0 is top."""
        depth = promote(depth)
        ptr = self.window_ptr - 1 - depth
        if ptr >= 0:
            return self.window[ptr]
        else:
            backing_idx = self.backing_ptr + ptr
            if backing_idx < 0:
                raise IndexError("Return stack underflow in peek")
            return self.backing[backing_idx]

    def poke(self, depth, value):
        """Set element at depth without push/pop. depth=0 is top."""
        depth = promote(depth)
        ptr = self.window_ptr - 1 - depth
        if ptr >= 0:
            self.window[ptr] = value
        else:
            backing_idx = self.backing_ptr + ptr
            if backing_idx < 0:
                raise IndexError("Return stack underflow in poke")
            self.backing[backing_idx] = value

    def depth(self):
        """Return total stack depth."""
        return self.backing_ptr + self.window_ptr

    def clear(self):
        """Clear the stack."""
        self.window_ptr = 0
        self.backing_ptr = 0

    # Loop operations - each loop frame is 2 cells: limit (bottom), counter (top)

    def push_loop(self, limit, counter):
        """Push loop parameters: limit first, then counter on top."""
        self.push(limit)
        self.push(counter)

    def pop_loop(self):
        """Pop loop parameters, returns (limit, counter)."""
        counter = self.pop()
        limit = self.pop()
        return limit, counter

    @unroll_safe
    def peek_loop_counter(self, loop_depth=0):
        """
        Get loop counter at given loop depth (0 = innermost).
        Counter is at top of each 2-cell loop frame.
        """
        loop_depth = promote(loop_depth)
        # Each loop is 2 cells, counter is at offset 0 from top of frame
        cell_depth = loop_depth * 2
        return self.peek(cell_depth)

    @unroll_safe
    def peek_loop_limit(self, loop_depth=0):
        """
        Get loop limit at given loop depth (0 = innermost).
        Limit is below counter in each 2-cell loop frame.
        """
        loop_depth = promote(loop_depth)
        # Each loop is 2 cells, limit is at offset 1 from top of frame
        cell_depth = loop_depth * 2 + 1
        return self.peek(cell_depth)

    @unroll_safe
    def set_loop_counter(self, loop_depth, value):
        """Set loop counter at given loop depth (0 = innermost)."""
        loop_depth = promote(loop_depth)
        cell_depth = loop_depth * 2
        self.poke(cell_depth, value)


class VirtualCallStack(object):
    """
    Call stack (thread, ip pairs) with virtual window.
    Uses two parallel windows for thread refs and IPs.
    """

    def __init__(self):
        self.window_threads = [None] * VSTACK_WINDOW_SIZE
        self.window_ips = [0] * VSTACK_WINDOW_SIZE
        self.window_ptr = 0

        self.backing_threads = [None] * VSTACK_BACKING_SIZE
        self.backing_ips = [0] * VSTACK_BACKING_SIZE
        self.backing_ptr = 0

    @unroll_safe
    def _spill_to_backing(self):
        spill_count = VSTACK_WINDOW_SIZE // 2
        for i in range(spill_count):
            self.backing_threads[self.backing_ptr + i] = self.window_threads[i]
            self.backing_ips[self.backing_ptr + i] = self.window_ips[i]
        self.backing_ptr += spill_count
        for i in range(spill_count, VSTACK_WINDOW_SIZE):
            self.window_threads[i - spill_count] = self.window_threads[i]
            self.window_ips[i - spill_count] = self.window_ips[i]
        self.window_ptr -= spill_count

    def push(self, thread, ip):
        ptr = self.window_ptr
        if ptr >= VSTACK_WINDOW_SIZE:
            self._spill_to_backing()
            ptr = self.window_ptr
        self.window_threads[ptr] = thread
        self.window_ips[ptr] = ip
        self.window_ptr = ptr + 1

    def pop(self):
        ptr = self.window_ptr - 1
        if ptr < 0:
            # Fill from backing
            if self.backing_ptr > 0:
                fill_count = min(self.backing_ptr, VSTACK_WINDOW_SIZE // 2)
                for i in range(self.window_ptr - 1, -1, -1):
                    self.window_threads[i + fill_count] = self.window_threads[i]
                    self.window_ips[i + fill_count] = self.window_ips[i]
                for i in range(fill_count):
                    self.window_threads[i] = self.backing_threads[self.backing_ptr - fill_count + i]
                    self.window_ips[i] = self.backing_ips[self.backing_ptr - fill_count + i]
                self.backing_ptr -= fill_count
                self.window_ptr += fill_count
                ptr = self.window_ptr - 1

        if ptr < 0:
            raise IndexError("Call stack underflow")

        thread = self.window_threads[ptr]
        ip = self.window_ips[ptr]
        self.window_threads[ptr] = None  # Clear reference
        self.window_ptr = ptr
        return thread, ip

    def is_empty(self):
        return self.window_ptr == 0 and self.backing_ptr == 0

    def clear(self):
        """Clear the stack."""
        self.window_ptr = 0
        self.backing_ptr = 0
        # Clear thread references to avoid memory leaks
        for i in range(len(self.window_threads)):
            self.window_threads[i] = None
