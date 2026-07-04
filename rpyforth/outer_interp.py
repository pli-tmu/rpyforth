from rpyforth.objects import (
    W_StringObject, Word, CodeThread, W_IntObject, W_PtrObject, W_FloatObject, W_WordObject, ZERO, TRUE,
    word_from_wid)
from rpyforth.inner_interp import Abort
from rpyforth.primitives import install_primitives
from rpyforth.util import to_upper, split_whitespace, split_whitespace_stateful

from rpython.rlib.rfile import create_stdio
from rpython.rlib.streamio import open_file_as_stream
from rpython.rlib.jit import elidable, unroll_safe, promote

INTERPRET = 0
COMPILE   = 1

# Max callee body length (excluding trailing EXIT) spliced inline at a call site.
MAX_INLINE_BODY = 32

# Control stack entry kinds
CTRL_IF   = 0
CTRL_ELSE = 1
CTRL_DO   = 2
CTRL_BEGIN = 3
CTRL_WHILE = 4
CTRL_CASE = 5
CTRL_OF   = 6

# Control-structure kinds POSTPONE can defer (IF/ELSE/THEN are parser tokens, not
# dictionary words, so POSTPONE routes them through runtime_compile_control).
CONTROL_IF   = 0
CONTROL_ELSE = 1
CONTROL_THEN = 2

class CtrlEntry(object):
    """Control stack entry for compilation-time control structures.

    RPython-friendly class to avoid tuple unpacking and string comparisons.
    """
    def __init__(self, kind, index):
        self.kind = kind    # int: CTRL_IF, CTRL_ELSE, or CTRL_DO
        self.index = index  # int: position in scurrent_code for patching
        self.leave_addrs = []  # list of LEAVE positions to patch (for DO loops)

class OuterInterpreter(object):
    _immutable_fields_ = ['wBR', 'w0BR', 'wLIT', 'wEXIT', 'wDO', 'wQDO', 'wLOOP', 'wPLUSLOOP', 'wLEAVE', 'wTYPE', 'wUNLOOP', 'wABORTQUOTE']

    def __init__(self, inner):
        self.inner = inner
        inner.outer = self    # Back-reference for LITERAL/FLITERAL primitives
        self.dict = {}         # dictionary is owned here (case-insensitive by uppercase keys)
        self.state = INTERPRET # state for compilation
        self.comment = False
        self.current_name = ''
        self.last_word = None  # Last defined word (for IMMEDIATE)

        # Input source tracking for SOURCE and >IN
        self.source_buffer = ''  # Current input line
        self.source_index = 0    # Current parse position (>IN)
        # Line-buffered source of the file currently being INCLUDEd, so REFILL
        # (executed from a colon body) can pull the next physical line and make
        # it the current parse buffer. include_pos indexes the next unread line.
        self.include_lines = []
        self.include_pos = 0
        # Per-spelling occurrence counters for string tokens (S", .", C") on
        # the current line: each spelling scans for its own nth occurrence.
        self.string_token_counts = {}

        # Runtime parse cursor. Defining words (CONSTANT/CREATE/VARIABLE/DEFER/')
        # can execute inside a colon body; when they do, they consume the next
        # token of the line currently being interpreted. self.toks holds that
        # line's tokens and the cursor lives in a dedicated heap cell (to_in_addr)
        # so >IN save/restore round-trips through @ / ! naturally. The cursor is a
        # token index, not a character offset -- adequate for save/restore, which
        # is all the appbench sources do with >IN.
        self.toks = []
        # Reserve a dedicated cell for the parse cursor at the top of the heap so
        # it never collides with here-managed user allocations (which grow from
        # 0). Fixed, not here-allocated, so tests that use low addresses (0, 4)
        # are undisturbed.
        from rpyforth.heap import HEAP_SIZE_BYTES
        self.to_in_addr = HEAP_SIZE_BYTES - 8
        inner.cell_store(self.to_in_addr, 0)
        # Dedicated fixed cell backing the BASE variable, next to the parse cursor
        # so it never collides with here-managed allocations. prim_BASE keeps it
        # and inner.base in sync so `BASE @` / `BASE !` round-trip the radix.
        self.base_addr = HEAP_SIZE_BYTES - 16
        inner.cell_store(self.base_addr, inner.base)
        # Dedicated fixed cell backing STATE. STATE ( -- a-addr ) refreshes this
        # cell from self.state and pushes its address, so `STATE @` reads the
        # live compilation state both when interpreting and from a colon body
        # (the state-smart POSTPONE idiom in brainless environ.fs/utils.fs).
        self.state_addr = HEAP_SIZE_BYTES - 24
        # Fixed scratch buffer for WORD's counted string, below the reserved
        # cells: WORD must not advance HERE (fcp measures dictionary growth
        # with HERE deltas around book loading).
        self.word_scratch_addr = HEAP_SIZE_BYTES - 280
        inner.cell_store(self.state_addr, 0)

        # Position of a DOES> body within the definition currently compiling
        # (-1 = none). At finalize this body is sliced into a standalone thread
        # and bound as the runtime action of the words the definition CREATEs.
        self.does_ip_mark = -1

        # True when the word currently compiling was bound early (RECURSE /
        # RECURSIVE) so its dict Word must be reused at finalize. A plain
        # redefinition instead installs a fresh Word, leaving earlier references
        # (e.g. the process chain in cd16sim) pointing at the prior definition.
        self.current_predefined = False

        self.reset_code()

        self.ctrl = []         # control stack at compilation

        # install minimal core words into dictionary
        install_primitives(self)

        self.wBR = self.dict["BRANCH"]
        self.w0BR = self.dict["0BRANCH"]
        self.wLIT = self.dict["LIT"]
        self.wEXIT = self.dict["EXIT"]
        self.wDO = self.dict["(DO)"]
        self.wQDO = self.dict["(?DO)"]
        self.wLOOP = self.dict["(LOOP)"]
        self.wPLUSLOOP = self.dict["(+LOOP)"]
        self.wUNLOOP = self.dict["UNLOOP"]
        self.wLEAVE = self.dict["LEAVE"]
        self.wTYPE = self.dict["TYPE"]
        self.wABORTQUOTE = self.dict['(ABORT")']

        # Cell address backing each VALUE name (for TO to locate its storage).
        self.value_addrs = {}

        # Slot id backing each DEFER name (for IS to locate its binding).
        self.defer_ids = {}

        # Search order (A5). Wordlist 0 is the FORTH-WORDLIST -- the base dict
        # installed above, so ordinary lookups keep hitting self.dict directly.
        # A wordlist is just a name->Word dict; wordlist_ids indexes them.
        # search_order lists wordlist ids searched front-first; current_wl is
        # where new definitions land. While the order is the lone default list
        # (order_is_default), lookups stay on the single-dict fast path.
        self.forth_wl = self.dict
        self.wordlists = [self.dict]
        self.search_order = [0]
        self.current_wl = 0
        self.order_is_default = True

        # True while compiling a :NONAME definition (push xt instead of naming it).
        self.noname_mode = False

        # [IF]/[ELSE]/[THEN] conditional-compilation skip state (spans lines).
        self.cond_skipping = False
        self.cond_skip_depth = 0
        self.cond_skip_to_else = False

        # Open ( ) comment nesting carried across lines. An unterminated '('
        # comment in an INCLUDEd file continues onto following lines (gforth
        # behaviour); this counts how many are still open at line start.
        self.paren_depth = 0

        # Define LITERAL as an immediate word (for POSTPONE to find it)
        self._define_literal_word()
        # Define FLITERAL as an immediate word for floating point literals
        self._define_fliteral_word()

    def reset_code(self):
        self.current_code = [None] * 128
        self.current_lits = [None] * 128
        self.cc_ptr = 0
        self.lit_ptr = 0

    def push_code(self, w):
        if self.cc_ptr >= len(self.current_code):
            self.current_code = self.current_code + [None] * len(self.current_code)
        self.current_code[self.cc_ptr] = w
        self.cc_ptr += 1

    def pop_code(self):
        assert self.cc_ptr > 0
        self.cc_ptr -= 1
        return self.current_code[self.cc_ptr]

    def push_lit(self, w):
        if self.lit_ptr >= len(self.current_lits):
            self.current_lits = self.current_lits + [None] * len(self.current_lits)
        self.current_lits[self.lit_ptr] = w
        self.lit_ptr += 1

    def pop_lit(self):
        assert self.lit_ptr > 0
        self.lit_ptr -= 1
        return self.current_lits[self.lit_ptr]

    def define_prim(self, name, func):
        w = Word(name, prim=func, immediate=False, thread=None)
        self.dict[to_upper(name)] = w
        return w

    def define_colon(self, name, thread):
        w = Word(name, prim=None, immediate=False, thread=thread)
        self.dict[to_upper(name)] = w
        self.last_word = w
        return w

    def _define_literal_word(self):
        """Define LITERAL as an immediate word that compiles a literal."""
        # LITERAL is IMMEDIATE and uses prim_LITERAL from primitives.py
        from rpyforth.primitives import prim_LITERAL
        w = Word("LITERAL", prim=prim_LITERAL, immediate=True, thread=None)
        self.dict["LITERAL"] = w
        self.last_word = w

    def _define_fliteral_word(self):
        """Define FLITERAL as an immediate word that compiles a float literal."""
        # FLITERAL is IMMEDIATE and uses prim_FLITERAL from primitives.py
        from rpyforth.primitives import prim_FLITERAL
        w = Word("FLITERAL", prim=prim_FLITERAL, immediate=True, thread=None)
        self.dict["FLITERAL"] = w
        self.last_word = w

    def _emit_word(self, w):
        self.push_code(w)
        self.push_lit(ZERO)

    def _emit_lit(self, w_n):
        self.push_code(self.wLIT)
        self.push_lit(w_n)

    @elidable
    def _is_number(self, s):
        length = len(s)
        if length == 0:
            return False
        start_idx = 0
        if s[0] == '-':
            start_idx = 1
            if length == 1:
                return False
        # Unroll the check for better performance
        for i in range(start_idx, length):
            ch = s[i]
            if ch < '0' or ch > '9':
                return False
        return True

    @elidable
    def _to_number(self, s):
        """Convert string to integer. Optimized for JIT."""
        sign = 1
        start_idx = 0
        length = len(s)
        if s[0] == '-':
            sign = -1
            start_idx = 1
        n = 0
        for i in range(start_idx, length):
            n = n * 10 + (ord(s[i]) - ord('0'))
        result = sign * n
        return result

    def _digit_value(self, ch):
        o = ord(ch)
        if o >= 48 and o <= 57:      # 0-9
            return o - 48
        if o >= 65 and o <= 90:      # A-Z
            return o - 55
        if o >= 97 and o <= 122:     # a-z
            return o - 87
        return -1

    def _prefix_base(self, s):
        """Recognize gforth number-prefix specifiers that override BASE: '$' hex,
        '#' decimal, '%' binary. Returns the effective base, or the passed-in
        default when there is no prefix."""
        if len(s) == 0:
            return -1, 0
        c = s[0]
        if c == '$':
            return 16, 1
        if c == '#':
            return 10, 1
        if c == '%':
            return 2, 1
        return 0, 0

    def _is_number_base(self, s, base):
        """True if s is an integer literal in the given base (leading '-' ok).
        A leading '$'/'#'/'%' prefix overrides the base (gforth specifiers)."""
        length = len(s)
        if length == 0:
            return False
        pbase, pskip = self._prefix_base(s)
        if pbase > 0:
            base = pbase
        start_idx = pskip
        if start_idx < length and s[start_idx] == '-':
            start_idx += 1
        if start_idx >= length:
            return False
        for i in range(start_idx, length):
            d = self._digit_value(s[i])
            if d < 0 or d >= base:
                return False
        return True

    def _to_number_base(self, s, base):
        length = len(s)
        pbase, pskip = self._prefix_base(s)
        if pbase > 0:
            base = pbase
        sign = 1
        start_idx = pskip
        if start_idx < length and s[start_idx] == '-':
            sign = -1
            start_idx += 1
        n = 0
        for i in range(start_idx, length):
            n = n * base + self._digit_value(s[i])
        return sign * n

    @unroll_safe
    def _is_float(self, s):
        length = len(s)
        if length == 0:
            return False

        # Handle negative sign
        idx = 0
        if s[idx] == '-':
            idx += 1
            if idx >= length:
                return False

        # Must have at least one digit or decimal point
        has_digit = False
        has_dot = False
        has_e = False

        while idx < length:
            ch = s[idx]
            if ch == '.':
                if has_dot or has_e:
                    return False
                has_dot = True
            elif ch == 'E' or ch == 'e':
                if has_e or not has_digit:
                    return False
                has_e = True
                # Check for optional sign after E
                if idx + 1 < length and (s[idx + 1] == '+' or s[idx + 1] == '-'):
                    idx += 1
            elif '0' <= ch <= '9':
                has_digit = True
            else:
                return False
            idx += 1

        # Must have at least a dot or E to be a float
        return has_digit and (has_dot or has_e)

    def _to_float(self, s):
        """Convert string to float"""
        # Handle Forth-style notation like '1e' (meaning 1e0)
        # Python requires a digit after 'e', so append '0' if needed
        if s.endswith('e') or s.endswith('E'):
            s = s + '0'
        elif s.endswith('e+') or s.endswith('E+') or s.endswith('e-') or s.endswith('E-'):
            s = s + '0'
        return float(s)

    def _emit_with_target(self, w, target_index):
        self.push_code(w)
        self.push_lit(W_IntObject(target_index))

    def _patch_here(self, at_index):
        self.current_lits[at_index] = W_IntObject(self.cc_ptr)

    def _read_tok(self, toks, i):
        t = toks[i]
        return t, i+1

    # Helper methods for interpret_line refactoring

    @unroll_safe
    def _find_nth_occurrence(self, line, token, n):
        """Find the Nth occurrence (0-indexed) of token in line. Returns position after token, or -1."""
        line_len = len(line)
        token_len = len(token)
        count = 0
        pos = 0

        while pos < line_len:
            # Skip whitespace
            while pos < line_len and line[pos] in ' \t\n\r\v\f':
                pos += 1
            if pos >= line_len:
                break

            # Read a token
            tok_start = pos
            while pos < line_len and line[pos] not in ' \t\n\r\v\f':
                pos += 1
            tok = line[tok_start:pos]

            if tok == token:
                if count == n:
                    return pos
                count += 1

        return -1

    @unroll_safe
    def _parse_string_at_occurrence(self, start_token, occurrence):
        """Parse string from the Nth occurrence of start_token in source_buffer.

        Returns the string content (without quotes).
        """
        line = self.source_buffer
        line_len = len(line)

        pos = self._find_nth_occurrence(line, start_token, occurrence)
        if pos < 0:
            return ''

        # Skip one space after token
        if pos < line_len and line[pos] == ' ':
            pos += 1

        # Read the string content until closing quote
        str_start = pos
        while pos < line_len and line[pos] != '"':
            pos += 1

        return line[str_start:pos]

    @unroll_safe
    def _skip_tokens_until_quote(self, toks, i):
        """Skip tokens until we find one ending with quote. Returns new index."""
        toks_len = len(toks)
        while i < toks_len:
            t = toks[i]
            i += 1
            t_len = len(t)
            if t_len > 0 and t[t_len - 1] == '"':
                break
        return i

    @unroll_safe
    def _parse_string_until_quote(self, toks, i):
        """Parse tokens until closing quote, return (parsed_string, new_index).

        Note: This is kept for ABORT" in compile mode where we need token-based parsing.
        """
        toks_len = len(toks)
        parts = []
        while i < toks_len:
            t, i = self._read_tok(toks, i)
            t_len = len(t)
            if t_len > 0 and t[t_len - 1] == '"':
                stop = t_len - 1
                assert 0 <= stop <= len(t)
                parts.append(t[:stop])
                break
            parts.append(t)
        return ' '.join(parts), i

    def _next_string_occurrence(self, start_token):
        n = self.string_token_counts.get(start_token, 0)
        self.string_token_counts[start_token] = n + 1
        return n

    def _copy_string_counts(self):
        d = {}
        for k in self.string_token_counts:
            d[k] = self.string_token_counts[k]
        return d

    def _handle_s_quote(self, toks, i, start_token):
        """Handle S" / s" - parse a string. In interpret mode push ( c-addr u );
        in compile mode allocate the buffer now and emit two literals so the
        compiled word pushes the same pair at runtime."""
        parsed_str = self._parse_string_at_occurrence(
            start_token, self._next_string_occurrence(start_token))
        i = self._skip_tokens_until_quote(toks, i)
        size = len(parsed_str)
        c_addr = self.inner.alloc_buf(parsed_str, size)
        if self.state == INTERPRET:
            self.inner.push_ds_int(c_addr)
            self.inner.push_ds_int(size)
        else:
            self._emit_lit(W_IntObject(c_addr))
            self._emit_lit(W_IntObject(size))
        return i

    def _handle_dot_quote(self, toks, i, start_token):
        """Handle ." / ." - parse string and print or compile."""
        parsed_str = self._parse_string_at_occurrence(
            start_token, self._next_string_occurrence(start_token))
        i = self._skip_tokens_until_quote(toks, i)
        w_str = W_StringObject(parsed_str)
        if self.state == INTERPRET:
            self.inner.print_str(w_str)
        else:
            self._emit_lit(w_str)
            self._emit_word(self.wTYPE)
        return i

    def _find_close_paren(self, line, start):
        """Return the position of the next ')' at or after start, else len(line)."""
        pos = start
        n = len(line)
        while pos < n and line[pos] != ')':
            pos += 1
        return pos

    def _handle_dot_paren(self, toks, i, start_token):
        """.( ccc ) -- print ccc immediately during parsing (up to ')')."""
        line = self.source_buffer
        occ = self._find_nth_occurrence(line, start_token, 0)
        if occ < 0:
            return i
        if occ < len(line) and line[occ] == ' ':
            occ += 1
        end = self._find_close_paren(line, occ)
        assert 0 <= occ <= end
        self.inner.print_str(W_StringObject(line[occ:end]))
        # advance the token index past the ')'
        toks_len = len(toks)
        while i < toks_len:
            tok = toks[i]
            i += 1
            if ')' in tok:
                break
        return i

    def _define_simple_word(self, name, val):
        """Define a word that pushes a single value (for VARIABLE, CONSTANT, etc.)."""
        code = [self.wLIT, self.wEXIT]
        lits = [val, ZERO]
        thread = CodeThread(code, lits)
        self.define_colon(name, thread)

    def _handle_variable(self, toks, i):
        """Handle VARIABLE and FVARIABLE."""
        toks_len = len(toks)
        if i >= toks_len:
            print "VARIABLE/FVARIABLE requires a name"
            return -1
        name, i = self._read_tok(toks, i)
        addr = self.inner.here
        self.inner.here += self.inner.cell_size_bytes
        self._define_simple_word(name, W_IntObject(addr))
        return i

    def _handle_2variable(self, toks, i):
        """Handle 2VARIABLE."""
        toks_len = len(toks)
        if i >= toks_len:
            print "2VARIABLE requires a name"
            return -1
        name, i = self._read_tok(toks, i)
        addr = self.inner.here
        self.inner.here += self.inner.cell_size_bytes
        self.inner.here += self.inner.cell_size_bytes  # allocate 2 cells
        self._define_simple_word(name, W_IntObject(addr))
        return i

    def _handle_constant(self, toks, i):
        """Handle CONSTANT and FCONSTANT."""
        toks_len = len(toks)
        if i >= toks_len:
            print "CONSTANT requires a name"
            return -1
        name, i = self._read_tok(toks, i)
        # Check which stack has data and pop from it
        if self.inner.ds_int_size() > 0:
            # Unboxed integer
            intval = self.inner.pop_ds_int()
            val = W_IntObject(intval)
        elif self.inner.ds_ptr_floats > 0:
            # Unboxed float
            floatval = self.inner.pop_ds_float()
            val = W_FloatObject(floatval)
        elif self.inner.ds_ptr_locals > 0:
            # Boxed object
            val = self.inner.pop_ds()
        else:
            print "CONSTANT: stack underflow"
            return -1
        self._define_simple_word(name, val)
        return i

    def _naligned(self, addr, align):
        if align <= 1:
            return addr
        return (addr + align - 1) & ~(align - 1)

    def _handle_field(self, toks, i):
        """FIELD ( align1 offset1 align size "name" -- align2 offset2 ). Defines
        name as ( addr -- addr+field-offset ) and advances the running offset."""
        toks_len = len(toks)
        if i >= toks_len:
            print "FIELD requires a name"
            return -1
        name, i = self._read_tok(toks, i)
        if self.inner.ds_int_size() < 4:
            print "FIELD: stack underflow"
            return -1
        size = self.inner.pop_ds_int()
        align = self.inner.pop_ds_int()
        offset1 = self.inner.pop_ds_int()
        align1 = self.inner.pop_ds_int()
        field_offset = self._naligned(offset1, align)
        # name execution: ( addr -- addr + field_offset )
        code = [self.wLIT, self.forth_wl["+"], self.wEXIT]
        lits = [W_IntObject(field_offset), ZERO, ZERO]
        self.define_colon(name, CodeThread(code, lits))
        self.inner.push_ds_int(align1)
        self.inner.push_ds_int(field_offset + size)
        return i

    def _handle_end_struct(self, toks, i):
        """END-STRUCT ( align size "name" -- ). Defines name to push ( align size )."""
        toks_len = len(toks)
        if i >= toks_len:
            print "END-STRUCT requires a name"
            return -1
        name, i = self._read_tok(toks, i)
        if self.inner.ds_int_size() < 2:
            print "END-STRUCT: stack underflow"
            return -1
        size = self.inner.pop_ds_int()
        align = self.inner.pop_ds_int()
        size = self._naligned(size, align)
        code = [self.wLIT, self.wLIT, self.wEXIT]
        lits = [W_IntObject(align), W_IntObject(size), ZERO]
        self.define_colon(name, CodeThread(code, lits))
        return i

    def _handle_2constant(self, toks, i):
        """Handle 2CONSTANT ( x1 x2 "name" -- ): name pushes x1 x2."""
        toks_len = len(toks)
        if i >= toks_len:
            print "2CONSTANT requires a name"
            return -1
        name, i = self._read_tok(toks, i)
        if self.inner.ds_int_size() < 2:
            print "2CONSTANT: stack underflow"
            return -1
        x2 = self.inner.pop_ds_int()
        x1 = self.inner.pop_ds_int()
        code = [self.wLIT, self.wLIT, self.wEXIT]
        lits = [W_IntObject(x1), W_IntObject(x2), ZERO]
        self.define_colon(name, CodeThread(code, lits))
        return i

    def _handle_create(self, toks, i):
        """Handle CREATE."""
        toks_len = len(toks)
        if i >= toks_len:
            print "CREATE requires a name"
            return -1
        name, i = self._read_tok(toks, i)
        addr = self.inner.here
        self._define_simple_word(name, W_IntObject(addr))
        return i

    def _handle_value(self, toks, i):
        """Handle VALUE: a cell-backed mutable constant. The word fetches its cell
        (LIT addr @); TO stores into the cell."""
        toks_len = len(toks)
        if i >= toks_len:
            print "VALUE requires a name"
            return -1
        name, i = self._read_tok(toks, i)
        if self.inner.ds_int_size() <= 0:
            print "VALUE: stack underflow"
            return -1
        intval = self.inner.pop_ds_int()
        addr = self.inner.here
        self.inner.here += self.inner.cell_size_bytes
        self.inner.cell_store(addr, intval)
        self.value_addrs[to_upper(name)] = addr
        code = [self.wLIT, self.forth_wl["@"], self.wEXIT]
        lits = [W_IntObject(addr), ZERO, ZERO]
        self.define_colon(name, CodeThread(code, lits))
        return i

    def _handle_to(self, toks, i):
        """Handle TO (interpret mode): store the top value into the named VALUE."""
        toks_len = len(toks)
        if i >= toks_len:
            print "TO requires a name"
            return -1
        name, i = self._read_tok(toks, i)
        key = to_upper(name)
        if key not in self.value_addrs:
            print "TO: not a VALUE"
            return -1
        if self.inner.ds_int_size() <= 0:
            print "TO: stack underflow"
            return -1
        self.inner.cell_store(self.value_addrs[key], self.inner.pop_ds_int())
        return i

    def _include_file(self, path):
        """Read a Forth source file and interpret it line by line, preserving the
        caller's source-scan state across the nested interpretation."""
        f = open_file_as_stream(path)
        content = f.readall()
        f.close()
        saved_buf = self.source_buffer
        saved_idx = self.source_index
        saved_cnt = self._copy_string_counts()
        saved_toks = self.toks
        saved_cur = self.inner.cell_fetch_int(self.to_in_addr)
        saved_lines = self.include_lines
        saved_pos = self.include_pos
        saved_paren = self.paren_depth
        self.paren_depth = 0
        self.include_lines = content.split('\n')
        self.include_pos = 0
        n = len(self.include_lines)
        while self.include_pos < n:
            line = self.include_lines[self.include_pos]
            self.include_pos += 1
            self.interpret_line(line)
        self.paren_depth = saved_paren
        self.include_lines = saved_lines
        self.include_pos = saved_pos
        self.source_buffer = saved_buf
        self.source_index = saved_idx
        self.string_token_counts = saved_cnt
        self.toks = saved_toks
        self.inner.cell_store(self.to_in_addr, saved_cur)

    def _handle_include(self, toks, i):
        """Handle INCLUDE / REQUIRE: the filename is the next token."""
        toks_len = len(toks)
        if i >= toks_len:
            print "INCLUDE requires a filename"
            return -1
        name, i = self._read_tok(toks, i)
        self._include_file(name)
        return i

    def _handle_included(self):
        """Handle INCLUDED / REQUIRED ( c-addr u -- ): the filename is a string."""
        self.inner.pop_ds_int()           # length (unused: the cell holds the string)
        c_addr = self.inner.pop_ds_int()
        w = self.inner.buf_get(c_addr)
        assert isinstance(w, W_StringObject)
        self._include_file(w.strval)

    def _handle_defer(self, toks, i):
        """Handle DEFER: define a word that executes a later-bound xt."""
        toks_len = len(toks)
        if i >= toks_len:
            print "DEFER requires a name"
            return -1
        name, i = self._read_tok(toks, i)
        slot = len(self.inner.deferred_words)
        self.inner.deferred_words.append(None)
        self.defer_ids[to_upper(name)] = slot
        code = [self.wLIT, self.forth_wl["(DEFER)"], self.wEXIT]
        lits = [W_IntObject(slot), ZERO, ZERO]
        self.define_colon(name, CodeThread(code, lits))
        return i

    def _handle_is(self, toks, i):
        """Handle IS (interpret mode): bind the xt on the stack to a DEFER word."""
        toks_len = len(toks)
        if i >= toks_len:
            print "IS requires a name"
            return -1
        name, i = self._read_tok(toks, i)
        key = to_upper(name)
        if key not in self.defer_ids:
            print "IS: not a DEFER"
            return -1
        self.inner.deferred_words[self.defer_ids[key]] = \
            word_from_wid(self.inner.pop_ds_int())
        return i

    # Control structure compilation helpers

    def _compile_if(self):
        """Compile IF."""
        orig = self.cc_ptr
        self._emit_with_target(self.w0BR, 0)
        self.ctrl.append(CtrlEntry(CTRL_IF, orig))

    def _compile_else(self):
        """Compile ELSE."""
        if not self.ctrl:
            print "ELSE without IF"
            return False
        entry = self.ctrl.pop()
        if entry.kind != CTRL_IF:
            print "ELSE without IF"
            return False
        self._patch_here(entry.index)
        orig2 = self.cc_ptr
        self._emit_with_target(self.wBR, 0)
        self.ctrl.append(CtrlEntry(CTRL_ELSE, orig2))
        self._patch_here(entry.index)
        return True

    def _compile_then(self):
        """Compile THEN. Resolves the forward branch left by IF/ELSE, or by a
        WHILE whose loop was closed with UNTIL (BEGIN ... WHILE ... UNTIL THEN)."""
        if not self.ctrl:
            print "THEN without IF/ELSE"
            return False
        entry = self.ctrl.pop()
        if (entry.kind != CTRL_IF and entry.kind != CTRL_ELSE
                and entry.kind != CTRL_WHILE):
            print "THEN without IF/ELSE"
            return False
        self._patch_here(entry.index)
        return True

    def _compile_do(self):
        """Compile DO."""
        self._emit_word(self.wDO)
        do_body_start = self.cc_ptr
        self.ctrl.append(CtrlEntry(CTRL_DO, do_body_start))

    def _compile_qdo(self):
        """Compile ?DO. Like DO, but emits a forward branch (patched to the loop
        end by LOOP/+LOOP) taken at runtime when limit == start."""
        qdo_addr = self.cc_ptr
        self._emit_with_target(self.wQDO, 0)
        do_body_start = self.cc_ptr
        entry = CtrlEntry(CTRL_DO, do_body_start)
        entry.leave_addrs.append(qdo_addr)
        self.ctrl.append(entry)

    def _compile_loop(self):
        """Compile LOOP."""
        if not self.ctrl:
            print "LOOP without DO"
            return False
        entry = self.ctrl.pop()
        if entry.kind != CTRL_DO:
            print "LOOP without DO"
            return False
        self._emit_with_target(self.wLOOP, entry.index)
        loop_end = self.cc_ptr
        for leave_addr in entry.leave_addrs:
            self.current_lits[leave_addr] = W_IntObject(loop_end)
        return True

    def _compile_begin(self):
        """Compile BEGIN."""
        begin_addr = self.cc_ptr
        self.ctrl.append(CtrlEntry(CTRL_BEGIN, begin_addr))

    def _compile_while(self):
        """Compile WHILE."""
        if not self.ctrl:
            print "WHILE without BEGIN"
            return False
        entry = self.ctrl.pop()
        if entry.kind != CTRL_BEGIN:
            print "WHILE without BEGIN"
            return False
        while_addr = self.cc_ptr
        self._emit_with_target(self.w0BR, 0)
        self.ctrl.append(CtrlEntry(CTRL_BEGIN, entry.index))
        self.ctrl.append(CtrlEntry(CTRL_WHILE, while_addr))
        return True

    def _compile_repeat(self):
        """Compile REPEAT."""
        if len(self.ctrl) < 2:
            print "REPEAT without BEGIN...WHILE"
            return False
        while_entry = self.ctrl.pop()
        begin_entry = self.ctrl.pop()
        if while_entry.kind != CTRL_WHILE or begin_entry.kind != CTRL_BEGIN:
            print "REPEAT without proper BEGIN...WHILE"
            return False
        self._emit_with_target(self.wBR, begin_entry.index)
        self._patch_here(while_entry.index)
        return True

    def _compile_plusloop(self):
        """Compile +LOOP."""
        if not self.ctrl:
            print "+LOOP without DO"
            return False
        entry = self.ctrl.pop()
        if entry.kind != CTRL_DO:
            print "+LOOP without DO"
            return False
        self._emit_with_target(self.wPLUSLOOP, entry.index)
        loop_end = self.cc_ptr
        for leave_addr in entry.leave_addrs:
            self.current_lits[leave_addr] = W_IntObject(loop_end)
        return True

    def _compile_again(self):
        """Compile AGAIN (unconditional branch back to BEGIN)."""
        if not self.ctrl:
            print "AGAIN without BEGIN"
            return False
        entry = self.ctrl.pop()
        if entry.kind != CTRL_BEGIN:
            print "AGAIN without BEGIN"
            return False
        self._emit_with_target(self.wBR, entry.index)
        return True

    def _compile_until(self):
        """Compile UNTIL (conditional branch back to BEGIN if false).

        Also supports BEGIN ... WHILE ... UNTIL THEN (used by fcp's >goodVar): the
        WHILE entry sits above its BEGIN, so pop the WHILE, branch back to the
        BEGIN, then re-push the WHILE so the following THEN resolves its forward
        exit branch."""
        if not self.ctrl:
            print "UNTIL without BEGIN"
            return False
        entry = self.ctrl.pop()
        if entry.kind == CTRL_WHILE:
            if not self.ctrl:
                print "UNTIL without BEGIN"
                self.ctrl.append(entry)
                return False
            begin_entry = self.ctrl.pop()
            if begin_entry.kind != CTRL_BEGIN:
                print "UNTIL without BEGIN"
                self.ctrl.append(begin_entry)
                self.ctrl.append(entry)
                return False
            self._emit_with_target(self.w0BR, begin_entry.index)
            self.ctrl.append(entry)
            return True
        if entry.kind != CTRL_BEGIN:
            print "UNTIL without BEGIN"
            return False
        self._emit_with_target(self.w0BR, entry.index)
        return True

    def _compile_leave(self):
        """Compile LEAVE."""
        # Find the innermost DO loop on the control stack
        for i in range(len(self.ctrl) - 1, -1, -1):
            entry = self.ctrl[i]
            if entry.kind == CTRL_DO:
                # Emit LEAVE with a placeholder target (will be patched by LOOP/+LOOP)
                leave_addr = self.cc_ptr
                self._emit_with_target(self.wLEAVE, 0)  # 0 is placeholder
                entry.leave_addrs.append(leave_addr)
                return True
        print "LEAVE without DO"
        return False

    def _compile_case(self):
        """Compile CASE. The entry collects each ENDOF's forward branch, all
        patched past the selector DROP at ENDCASE."""
        self.ctrl.append(CtrlEntry(CTRL_CASE, 0))

    def _compile_of(self):
        """Compile OF: OVER = 0BRANCH<next-clause> DROP."""
        if not self.ctrl or self.ctrl[len(self.ctrl) - 1].kind != CTRL_CASE:
            print "OF without CASE"
            return False
        self._emit_word(self.forth_wl["OVER"])
        self._emit_word(self.forth_wl["="])
        of_addr = self.cc_ptr
        self._emit_with_target(self.w0BR, 0)
        self._emit_word(self.forth_wl["DROP"])
        self.ctrl.append(CtrlEntry(CTRL_OF, of_addr))
        return True

    def _compile_endof(self):
        """Compile ENDOF: BRANCH<endcase>; patch the OF 0BRANCH to the next clause."""
        if not self.ctrl or self.ctrl[len(self.ctrl) - 1].kind != CTRL_OF:
            print "ENDOF without OF"
            return False
        of_entry = self.ctrl.pop()
        endof_addr = self.cc_ptr
        self._emit_with_target(self.wBR, 0)
        self.ctrl[len(self.ctrl) - 1].leave_addrs.append(endof_addr)
        self._patch_here(of_entry.index)
        return True

    def _compile_endcase(self):
        """Compile ENDCASE: DROP the selector, then patch every ENDOF branch past it."""
        if not self.ctrl or self.ctrl[len(self.ctrl) - 1].kind != CTRL_CASE:
            print "ENDCASE without CASE"
            return False
        entry = self.ctrl.pop()
        self._emit_word(self.forth_wl["DROP"])
        end = self.cc_ptr
        for endof_addr in entry.leave_addrs:
            self.current_lits[endof_addr] = W_IntObject(end)
        return True
        return False

    def _compile_char(self, toks, i):
        """Compile [CHAR]."""
        toks_len = len(toks)
        if i >= toks_len:
            print "[CHAR] requires a following character"
            return i
        char_tok = toks[i]
        i += 1
        if len(char_tok) > 0:
            self._emit_lit(W_IntObject(ord(char_tok[0])))
        else:
            print "[CHAR] got empty token"
        return i

    # Word execution/compilation helpers

    def _execute_or_push(self, w, t):
        """Execute word or push number in INTERPRET mode."""
        base = self.inner.base
        if w is not None:
            self.inner.execute_word_now(w)
        elif base == 10 and self._is_float(t):
            self.inner.push_ds_float(self._to_float(t))
        elif self._is_number_base(t, base):
            self.inner.push_ds_int(self._to_number_base(t, base))
        else:
            print "UNKNOWN: " + t

    def _value_word_literal(self, w):
        """Return the literal x if w is a value-word (body LIT x ; EXIT), else None.

        Lets a VARIABLE/CONSTANT/CREATE reference compile to an immediate push
        instead of a call. The word stays in the dictionary, so ' / >BODY /
        EXECUTE are unaffected."""
        if w.prim is not None:
            return None
        if w.immediate:
            return None
        if w.does_ip != -1:          # CREATE ... DOES> has custom runtime behavior
            return None
        thread = w.thread
        if thread is None:
            return None
        code = thread.code
        if len(code) != 2:
            return None
        if code[0] is not self.wLIT:
            return None
        if code[1] is not self.wEXIT:
            return None
        return thread.lits[0]

    def _inlinable_colon_body(self, w):
        """Return w's CodeThread if it is a small straight-line colon word that can
        be spliced inline, else None.

        Straight-line = ends in EXIT and contains none of the ip-altering words
        (BRANCH, 0BRANCH, (DO), (LOOP), (+LOOP), LEAVE, UNLOOP, interior EXIT,
        TAILCALL). The branch/loop ones encode an absolute target into their own
        thread and cannot be relocated; everything else is position independent."""
        if w.prim is not None:
            return None
        if w.immediate:
            return None
        if w.does_ip != -1:            # CREATE ... DOES> has custom runtime behavior
            return None
        thread = w.thread
        if thread is None:
            return None
        if thread.does_word is not None:
            return None
        code = thread.code
        n = len(code)
        if n < 2:
            return None
        if code[n - 1] is not self.wEXIT:  # tail-call-optimized words end in TAILCALL
            return None
        body_len = n - 1
        if body_len > MAX_INLINE_BODY:
            return None
        i = 0
        while i < body_len:
            cw = code[i]
            if (cw is self.wEXIT or cw is self.wBR or cw is self.w0BR or
                    cw is self.wDO or cw is self.wQDO or cw is self.wLOOP or
                    cw is self.wPLUSLOOP or cw is self.wLEAVE or cw is self.wUNLOOP):
                return None
            i += 1
        return thread

    def _emit_inline(self, thread):
        """Splice a callee body (all but its trailing EXIT) into the current def."""
        code = thread.code
        lits = thread.lits
        body_len = len(code) - 1
        i = 0
        while i < body_len:
            self.push_code(code[i])
            self.push_lit(lits[i])
            i += 1

    def _compile_word_or_literal(self, w, t):
        """Compile word or literal in COMPILE mode."""
        if w is not None:
            # Check if word is immediate - if so, execute it now
            if w.immediate:
                # Immediate words (including LITERAL, FLITERAL) are executed immediately
                # LITERAL and FLITERAL have primitives that pop from the stack and emit literals
                self.inner.execute_word_now(w)
            else:
                lit = self._value_word_literal(w)
                if lit is not None:
                    self._emit_lit(lit)
                else:
                    body = self._inlinable_colon_body(w)
                    if body is not None:
                        self._emit_inline(body)
                    else:
                        self._emit_word(w)
        elif self.inner.base == 10 and self._is_float(t):
            self._emit_lit(W_FloatObject(self._to_float(t)))
        elif self._is_number_base(t, self.inner.base):
            self._emit_lit(W_IntObject(self._to_number_base(t, self.inner.base)))
        else:
            print "UNKNOWN: " + t

    # System word handlers

    def _lookup(self, name_upper):
        """Resolve a name to a Word through the search order. On the default
        single-wordlist order this is exactly self.dict.get (the fast path)."""
        if self.order_is_default:
            return self.dict.get(name_upper, None)
        for idx in range(len(self.search_order)):
            wl = self.wordlists[self.search_order[idx]]
            w = wl.get(name_upper, None)
            if w is not None:
                return w
        return None

    def _recompute_order_default(self):
        self.order_is_default = (
            len(self.search_order) == 1
            and self.search_order[0] == 0
            and self.current_wl == 0
        )

    def _set_current_wl(self, wl_id):
        self.current_wl = wl_id
        self.dict = self.wordlists[wl_id]
        self._recompute_order_default()

    # WORDLIST ( -- wid )
    def _handle_wordlist(self):
        wid = len(self.wordlists)
        self.wordlists.append({})
        self.inner.push_ds_int(wid)

    # GET-CURRENT ( -- wid )
    def _handle_get_current(self):
        self.inner.push_ds_int(self.current_wl)

    # SET-CURRENT ( wid -- )
    def _handle_set_current(self):
        wid = self.inner.pop_ds_int()
        if 0 <= wid < len(self.wordlists):
            self._set_current_wl(wid)

    # FORTH-WORDLIST ( -- wid )
    def _handle_forth_wordlist(self):
        self.inner.push_ds_int(0)

    # DEFINITIONS ( -- ) make the top of the search order the current wordlist
    def _handle_definitions(self):
        if len(self.search_order) > 0:
            self._set_current_wl(self.search_order[0])

    # FORTH ( -- ) replace the top of the search order with FORTH-WORDLIST
    def _handle_forth(self):
        if len(self.search_order) == 0:
            self.search_order = [0]
        else:
            self.search_order[0] = 0
        self._recompute_order_default()

    # ONLY ( -- ) minimal search order (just FORTH here)
    def _handle_only(self):
        self.search_order = [0]
        self._recompute_order_default()

    # ALSO ( -- ) duplicate the top of the search order
    def _handle_also(self):
        if len(self.search_order) > 0:
            self.search_order.insert(0, self.search_order[0])
        else:
            self.search_order = [0]
        self._recompute_order_default()

    # >ORDER ( wid -- ) push wid onto the top of the search order (gforth)
    def _handle_to_order(self):
        wid = self.inner.pop_ds_int()
        if 0 <= wid < len(self.wordlists):
            self.search_order.insert(0, wid)
            self._recompute_order_default()

    # PREVIOUS ( -- ) drop the top of the search order
    def _handle_previous(self):
        if len(self.search_order) > 1:
            del self.search_order[0]
        self._recompute_order_default()

    # GET-ORDER ( -- wid_n ... wid_1 n )
    def _handle_get_order(self):
        # push bottom-first so wid_1 (top) ends up just under the count
        n = len(self.search_order)
        for k in range(n - 1, -1, -1):
            self.inner.push_ds_int(self.search_order[k])
        self.inner.push_ds_int(n)

    # SET-ORDER ( wid_n ... wid_1 n -- )
    def _handle_set_order(self):
        n = self.inner.pop_ds_int()
        if n < 0:
            # standard: restore the implementation default
            self.search_order = [0]
            self._recompute_order_default()
            return
        order = [0] * n
        for k in range(n):
            order[k] = self.inner.pop_ds_int()
        self.search_order = order
        self._recompute_order_default()

    # SEARCH-WORDLIST ( c-addr u wid -- 0 | xt 1 | xt -1 )
    def _handle_search_wordlist(self):
        wid = self.inner.pop_ds_int()
        self.inner.pop_ds_int()            # length (name held whole in the buf slot)
        c_addr = self.inner.pop_ds_int()
        buf_entry = self.inner.buf_get(c_addr)
        if buf_entry is None or not isinstance(buf_entry, W_StringObject):
            self.inner.push_ds_int(0)
            return
        name_upper = to_upper(buf_entry.strval)
        if 0 <= wid < len(self.wordlists):
            wl = self.wordlists[wid]
            w = wl.get(name_upper, None)
        else:
            w = None
        if w is None:
            self.inner.push_ds_int(0)
            return
        self.inner.push_ds_int(w.wid)
        if w.immediate:
            self.inner.push_ds_int(1)
        else:
            self.inner.push_ds_int(-1)

    def _counted_string_at(self, c_addr):
        """Read a counted string ( length byte then chars ) at c_addr. If a buf
        slot holds a W_StringObject there (an S"-style whole-string address), use
        it; otherwise decode the char-memory counted string that BL WORD builds."""
        buf_entry = self.inner.buf_get(c_addr)
        if isinstance(buf_entry, W_StringObject):
            return buf_entry.strval
        length = self.inner.char_fetch(c_addr)
        if length < 0:
            length = 0
        chars = []
        for k in range(length):
            chars.append(chr(self.inner.char_fetch(c_addr + 1 + k)))
        return "".join(chars)

    # FIND ( c-addr -- c-addr 0 | xt 1 | xt -1 )
    def _handle_find(self):
        w_caddr = self.inner.pop_ds_int()
        name = self._counted_string_at(w_caddr)
        word = self._lookup(to_upper(name))
        if word is not None:
            self.inner.push_ds_int(word.wid)
            if word.immediate:
                self.inner.push_ds_int(1)
            else:
                self.inner.push_ds_int(-1)
        else:
            self.inner.push_ds_int(w_caddr)
            self.inner.push_ds_int(0)

    # SOURCE ( -- c-addr u )
    def _handle_source(self):
        size = len(self.source_buffer)
        c_addr = self.inner.alloc_buf(self.source_buffer, size)
        self.inner.push_ds_int(c_addr)
        self.inner.push_ds_int(size)

    # >IN ( -- a-addr ) address of the runtime parse cursor (a token index).
    def _handle_to_in(self):
        self.inner.push_ds_int(self.to_in_addr)

    def parse_next_token(self):
        """Consume and return the next token from the line being interpreted,
        advancing the runtime parse cursor. Returns '' at end of line."""
        idx = self.inner.cell_fetch_int(self.to_in_addr)
        if idx < 0 or idx >= len(self.toks):
            return ''
        tok = self.toks[idx]
        self.inner.cell_store(self.to_in_addr, idx + 1)
        return tok

    def runtime_refill(self):
        """REFILL ( -- flag ): consume the next physical line of the file being
        INCLUDEd, make it the current parse buffer (resetting the parse cursor),
        and push true. If there is no next line, push false. Words like squares:
        use this to read table rows and parse names from them at run time."""
        if self.include_pos < len(self.include_lines):
            line = self.include_lines[self.include_pos]
            self.include_pos += 1
            self.source_buffer = line
            self.source_index = 0
            self.string_token_counts = {}
            self.toks = split_whitespace(line)
            self.inner.cell_store(self.to_in_addr, 0)
            self.inner.push_ds_int(-1)
        else:
            self.inner.push_ds_int(0)

    def runtime_postpone(self, word):
        """Append word to the definition currently being compiled. Invoked by the
        (POSTPONE) primitive that POSTPONE compiles for a non-immediate target,
        so an immediate word built with POSTPONE defers the target instead of
        executing it (e.g. hash! = POSTPONE ! must compile !, not run it)."""
        self._emit_word(word)

    def _control_postpone_kind(self, name_upper):
        """Map a control parser-token name to its CONTROL_* kind for POSTPONE, or
        -1 if it is not a POSTPONE-able control token."""
        if name_upper == "IF":
            return CONTROL_IF
        if name_upper == "ELSE":
            return CONTROL_ELSE
        if name_upper == "THEN":
            return CONTROL_THEN
        return -1

    def runtime_compile_control(self, kind):
        """Run a control-structure compiler against the definition currently
        being compiled. IF/ELSE/THEN are parser tokens rather than dictionary
        words, so POSTPONE IF (brainless tmovegen's ?single-move) compiles a call
        to this hook; when the enclosing immediate word runs at the outer word's
        compile time, it splices the control structure there."""
        if kind == CONTROL_IF:
            self._compile_if()
        elif kind == CONTROL_ELSE:
            self._compile_else()
        elif kind == CONTROL_THEN:
            self._compile_then()

    def _runtime_pop_value(self):
        """Pop one value off whichever data stack holds it, boxed for storage in
        a defined word's literal (mirrors _handle_constant)."""
        if self.inner.ds_int_size() > 0:
            return W_IntObject(self.inner.pop_ds_int())
        elif self.inner.ds_ptr_floats > 0:
            return W_FloatObject(self.inner.pop_ds_float())
        elif self.inner.ds_ptr_locals > 0:
            return self.inner.pop_ds()
        return ZERO

    def runtime_constant(self):
        """CONSTANT executed from a colon body: name the next token, bind value."""
        name = self.parse_next_token()
        if name == '':
            print "CONSTANT requires a name"
            return
        self._define_simple_word(name, self._runtime_pop_value())

    def runtime_variable(self):
        name = self.parse_next_token()
        if name == '':
            print "VARIABLE requires a name"
            return
        addr = self.inner.here
        self.inner.here += self.inner.cell_size_bytes
        self._define_simple_word(name, W_IntObject(addr))

    def runtime_create(self, does_word):
        """CREATE executed from a colon body. The child word pushes its data
        field address; if the enclosing definition had a DOES>, does_word runs
        after the push."""
        name = self.parse_next_token()
        if name == '':
            print "CREATE requires a name"
            return
        addr = self.inner.here
        if does_word is None:
            code = [self.wLIT, self.wEXIT]
            lits = [W_IntObject(addr), ZERO]
        else:
            code = [self.wLIT, does_word, self.wEXIT]
            lits = [W_IntObject(addr), ZERO, ZERO]
        self.define_colon(name, CodeThread(code, lits))

    def runtime_defer(self):
        name = self.parse_next_token()
        if name == '':
            print "DEFER requires a name"
            return
        slot = len(self.inner.deferred_words)
        self.inner.deferred_words.append(None)
        self.defer_ids[to_upper(name)] = slot
        code = [self.wLIT, self.forth_wl["(DEFER)"], self.wEXIT]
        lits = [W_IntObject(slot), ZERO, ZERO]
        self.define_colon(name, CodeThread(code, lits))

    def runtime_word(self):
        """WORD executed from a colon body: consume the next token of the line
        being interpreted (via the shared parse cursor) and store it as a counted
        string at HERE, leaving its address. The delimiter char is popped and
        ignored -- tokens are already whitespace-split, which covers BL WORD."""
        self.inner.pop_ds_int()
        word_str = self.parse_next_token()
        length = len(word_str)
        addr = self.word_scratch_addr
        self.inner.char_store(addr, length)
        i = 0
        while i < length:
            self.inner.char_store(addr + 1 + i, ord(word_str[i]))
            i += 1
        self.inner.push_ds_int(addr)

    def runtime_parse(self):
        """PARSE executed from a colon body ( char "ccc<char>" -- c-addr u ).
        Consume the next token of the line via the shared cursor and store it in
        char memory, returning its address and length. The token-based cursor
        means the delimiter char is honored only as whitespace, which covers the
        BL PARSE fcp uses."""
        self.inner.pop_ds_int()
        word_str = self.parse_next_token()
        length = len(word_str)
        addr = self.inner.here
        for ch in word_str:
            self.inner.char_store(self.inner.here, ord(ch))
            self.inner.here += 1
        self.inner.push_ds_int(addr)
        self.inner.push_ds_int(length)

    def runtime_marker(self):
        """MARKER executed from a colon body ( "name" -- ): define a word that, on
        execution, would forget everything defined after it. Dictionary rollback
        is not modeled here, so the marker word is a harmless no-op -- adequate
        for a single benchmark load."""
        name = self.parse_next_token()
        if name == '':
            print "MARKER requires a name"
            return
        code = [self.wEXIT]
        lits = [ZERO]
        self.define_colon(name, CodeThread(code, lits))

    def runtime_included(self):
        """INCLUDED executed from a colon body ( c-addr u -- ). Every string is
        byte-backed in data space (alloc_buf writes S" strings there too), so
        the filename is always read from char memory. Loading goes through the
        same nested source path as INCLUDE, which saves/restores the parse
        cursor."""
        u = self.inner.pop_ds_int()
        c_addr = self.inner.pop_ds_int()
        if u < 0:
            u = 0
        chars = []
        for k in range(u):
            chars.append(chr(self.inner.char_fetch(c_addr + k)))
        self._include_file("".join(chars))

    def runtime_find(self):
        """FIND executed from a colon body ( c-addr -- c-addr 0 | xt 1 | xt -1 )."""
        self._handle_find()

    def runtime_base(self):
        """BASE executed from a colon body ( -- a-addr )."""
        self._handle_base()

    def runtime_tick(self):
        name = self.parse_next_token()
        if name == '':
            print "' requires a following word"
            return
        word = self._lookup(to_upper(name))
        if word is not None:
            self.inner.push_ds_int(word.wid)
        else:
            print "' cannot find word:", name

    # ' (tick) ( "<spaces>name" -- xt )
    def _handle_tick(self, toks, i):
        toks_len = len(toks)
        if i >= toks_len:
            print "' requires a following word"
            return i
        name, i = self._read_tok(toks, i)
        name_upper = to_upper(name)
        word = self._lookup(name_upper)
        if word is not None:
            self.inner.push_ds_int(word.wid)
        else:
            print "' cannot find word:", name
        return i

    # ( - skip until )
    def _handle_paren_comment(self, toks, i):
        toks_len = len(toks)
        while i < toks_len:
            tok, i = self._read_tok(toks, i)
            if ')' in tok:
                break
        return i

    # COUNT ( c-addr1 -- c-addr2 u )
    def _handle_count(self):
        c_addr1 = self.inner.pop_ds_int()
        count = self.inner.cell_fetch(c_addr1)
        assert isinstance(count, W_IntObject)
        c_addr2 = c_addr1 + self.inner.cell_size_bytes
        self.inner.push_ds_int(c_addr2)
        self.inner.push_ds_int(count.intval)

    # WORD ( char "<chars>ccc<char>" -- c-addr )
    def _handle_word(self, toks, i):
        toks_len = len(toks)
        char_obj = self.inner.pop_ds_int()

        if i >= toks_len:
            word_str = ''
        else:
            word_str, i = self._read_tok(toks, i)

        length = len(word_str)
        addr = self.word_scratch_addr
        # Store as a counted string in character (byte) space so COUNT / C@ read
        # it back consistently. The fixed scratch buffer keeps HERE untouched.
        self.inner.char_store(addr, length)
        k = 0
        while k < length:
            self.inner.char_store(addr + 1 + k, ord(word_str[k]))
            k += 1
        self.inner.push_ds_int(addr)
        return i

    # STATE ( -- a-addr )
    def _handle_state(self):
        self.runtime_state()

    def runtime_state(self):
        """STATE ( -- a-addr ): refresh the dedicated state cell from the live
        compilation state and push its address. Works from interpret mode and
        from a colon body (state-smart words compiled with POSTPONE)."""
        state_val = -1 if self.state == COMPILE else 0
        self.inner.cell_store(self.state_addr, state_val)
        self.inner.push_ds_int(self.state_addr)

    def runtime_save_input(self):
        """SAVE-INPUT ( -- xn..x1 n ): save enough of the input-source state to
        allow RESTORE-INPUT to rewind. Here the only mutable position is the
        token parse cursor (>IN), so we push it plus a count of 1. brainless
        option-exists? saves, parses+finds the option name, then restores so the
        name can be re-parsed by create-option."""
        cur = self.inner.cell_fetch_int(self.to_in_addr)
        self.inner.push_ds_int(cur)
        self.inner.push_ds_int(1)

    def runtime_restore_input(self):
        """RESTORE-INPUT ( xn..x1 n -- flag ): restore the input position saved
        by SAVE-INPUT and push false (success). If the saved spec is not the one
        we produce (n != 1) push true (failure) leaving the cursor unchanged."""
        n = self.inner.pop_ds_int()
        if n != 1:
            self.inner.push_ds_int(-1)
            return
        cur = self.inner.pop_ds_int()
        self.inner.cell_store(self.to_in_addr, cur)
        self.inner.push_ds_int(0)

    # EVALUATE ( c-addr u -- )
    def _handle_evaluate(self):
        length = self.inner.pop_ds_int()
        c_addr = self.inner.pop_ds_int()
        buf_str = self.inner.buf_get(c_addr)
        assert isinstance(buf_str, W_StringObject)
        saved_buf = self.source_buffer
        saved_cnt = self._copy_string_counts()
        saved_toks = self.toks
        saved_cur = self.inner.cell_fetch_int(self.to_in_addr)
        self.interpret_line(buf_str.strval)
        self.source_buffer = saved_buf
        self.string_token_counts = saved_cnt
        self.toks = saved_toks
        self.inner.cell_store(self.to_in_addr, saved_cur)

    # >NUMBER ( ud1 c-addr1 u1 -- ud2 c-addr2 u2 )
    def _handle_to_number(self):
        """Convert string to number according to BASE."""
        u1 = self.inner.pop_ds_int()
        c_addr1 = self.inner.pop_ds_int()
        ud1_hi = self.inner.pop_ds_int()
        ud1_lo = self.inner.pop_ds_int()

        base = self.inner.base
        addr = c_addr1
        length = u1
        value = ud1_lo

        # Process each character
        chars_processed = 0
        for j in range(length):
            ch = chr(self.inner.char_fetch(addr + j))

            # Convert character to digit value
            digit = -1
            if '0' <= ch <= '9':
                digit = ord(ch) - ord('0')
            elif 'A' <= ch <= 'Z':
                digit = ord(ch) - ord('A') + 10
            elif 'a' <= ch <= 'z':
                digit = ord(ch) - ord('a') + 10

            if digit < 0 or digit >= base:
                # Invalid character, stop conversion
                break

            value = value * base + digit
            chars_processed += 1

        # Return updated values
        self.inner.push_ds_int(value)
        self.inner.push_ds_int(0)  # ud2 high
        self.inner.push_ds_int(addr + chars_processed)
        self.inner.push_ds_int(length - chars_processed)

    # ENVIRONMENT? ( c-addr u -- false | i*x true )
    def _handle_environment_query(self):
        """Query environmental information."""
        u = self.inner.pop_ds_int()
        c_addr = self.inner.pop_ds_int()

        # Extract the query string from buffer
        buf_entry = self.inner.buf_get(c_addr)
        assert isinstance(buf_entry, W_StringObject)
        strval = buf_entry.strval
        assert 0 <= u <= len(strval)
        query = strval[:u]
        query_upper = to_upper(query)

        # Handle known queries
        if query_upper == "/COUNTED-STRING":
            self.inner.push_ds_int(255)  # max counted string length
            self.inner.push_ds_int(-1)
        elif query_upper == "/HOLD":
            self.inner.push_ds_int(128)  # hold buffer size
            self.inner.push_ds_int(-1)
        elif query_upper == "/PAD":
            self.inner.push_ds_int(256)  # pad buffer size
            self.inner.push_ds_int(-1)
        elif query_upper == "ADDRESS-UNIT-BITS":
            self.inner.push_ds_int(8)  # bits per address unit
            self.inner.push_ds_int(-1)
        elif query_upper == "CORE":
            self.inner.push_ds_int(-1)  # CORE wordset is present
            self.inner.push_ds_int(-1)
        elif query_upper == "CORE-EXT":
            self.inner.push_ds_int(0)  # CORE-EXT not fully present
            self.inner.push_ds_int(-1)
        elif query_upper == "FLOORED":
            self.inner.push_ds_int(0)  # division is symmetric, not floored
            self.inner.push_ds_int(-1)
        elif query_upper == "MAX-CHAR":
            self.inner.push_ds_int(255)  # max character value
            self.inner.push_ds_int(-1)
        elif query_upper == "MAX-N":
            self.inner.push_ds_int((1 << 62) - 1 + (1 << 62))  # max signed cell
            self.inner.push_ds_int(-1)
        elif query_upper == "MAX-U":
            self.inner.push_ds_int(-1)  # max unsigned (all bits set)
            self.inner.push_ds_int(-1)
        elif query_upper == "MAX-D":
            self.inner.push_ds_int(-1)  # max signed double: low cell
            self.inner.push_ds_int((1 << 62) - 1 + (1 << 62))  # high cell
            self.inner.push_ds_int(-1)
        elif query_upper == "MAX-UD":
            self.inner.push_ds_int(-1)  # max unsigned double: low cell
            self.inner.push_ds_int(-1)  # high cell
            self.inner.push_ds_int(-1)
        elif query_upper == "RETURN-STACK-CELLS":
            self.inner.push_ds_int(64)  # return stack size
            self.inner.push_ds_int(-1)
        elif query_upper == "STACK-CELLS":
            self.inner.push_ds_int(64)  # data stack size
            self.inner.push_ds_int(-1)
        else:
            self.inner.push_ds_int(0)  # false - unknown query

    # BASE ( -- a-addr )
    def _handle_base(self):
        """Return the address of the BASE variable ( -- a-addr ). The cell is the
        source of truth (HEX/DECIMAL/BASE! and a direct `BASE !` all write it), so
        adopt it into inner.base -- which drives number parsing -- before handing
        out the address for a following `BASE @` / `BASE !`."""
        cell_val = self.inner.cell_fetch_int(self.base_addr)
        if cell_val >= 2:
            self.inner.base = cell_val
        self.inner.push_ds_int(self.base_addr)

    # ABORT ( -- )
    def _handle_abort(self):
        self.inner.ds_ptr = 0
        self.inner.rs_ptr = 0
        self.state = INTERPRET
        print "ABORT"

    # ABORT" ( flag "ccc<quote>" -- )
    @unroll_safe
    def _handle_abort_quote(self, toks, i):
        toks_len = len(toks)
        abort_msg_parts = []
        while i < toks_len:
            tok, i = self._read_tok(toks, i)
            tok_len = len(tok)
            if tok_len > 0 and tok[tok_len - 1] == '"':
                stop = tok_len - 1
                if stop > 0:
                    abort_msg_parts.append(tok[:stop])
                break
            abort_msg_parts.append(tok)

        abort_msg = ' '.join(abort_msg_parts)
        flag = self.inner.pop_ds_int()

        if flag != 0:
            print "ABORT:", abort_msg
            self.inner.clear_ds_int()
            self.inner.ds_ptr_floats = 0
            self.inner.ds_ptr_locals = 0
            self.inner.rs_ptr = 0
            self.state = INTERPRET
            return i, True  # Signal to return from interpret_line
        return i, False

    # QUIT ( -- )
    def _handle_quit(self):
        self.inner.ds_ptr = 0
        self.inner.rs_ptr = 0
        self.state = INTERPRET

    # IMMEDIATE ( -- )
    def _handle_immediate(self):
        """Mark the last defined word as immediate."""
        self.runtime_immediate()

    def runtime_immediate(self):
        """Mark the most recently defined word immediate. Called both from the
        interpret-mode lexical handler and from the (IMMEDIATE) primitive that
        IMMEDIATE compiles into a colon body, so a defining word can run
        CREATE IMMEDIATE ... DOES> to build immediate children at runtime."""
        if self.last_word is not None:
            self.last_word.immediate = True
        else:
            print "IMMEDIATE: no word to mark"

    # Dispatch methods for interpret and compile modes

    def _dispatch_interpret(self, tkey, toks, i, toks_len):
        """Dispatch interpret-mode words. Returns (handled, new_i, should_return)."""
        if tkey == "2VARIABLE":
            result = self._handle_2variable(toks, i)
            if result < 0:
                return True, i, True
            return True, result, False

        if tkey == "2CONSTANT":
            result = self._handle_2constant(toks, i)
            if result < 0:
                return True, i, True
            return True, result, False

        if tkey == "FIELD":
            result = self._handle_field(toks, i)
            if result < 0:
                return True, i, True
            return True, result, False

        if tkey == "END-STRUCT":
            result = self._handle_end_struct(toks, i)
            if result < 0:
                return True, i, True
            return True, result, False

        if tkey == "VALUE":
            result = self._handle_value(toks, i)
            if result < 0:
                return True, i, True
            return True, result, False

        if tkey == "TO":
            result = self._handle_to(toks, i)
            if result < 0:
                return True, i, True
            return True, result, False

        if tkey == "INCLUDE" or tkey == "REQUIRE":
            result = self._handle_include(toks, i)
            if result < 0:
                return True, i, True
            return True, result, False

        if tkey == "INCLUDED" or tkey == "REQUIRED":
            self._handle_included()
            return True, i, False

        if tkey == "IS":
            result = self._handle_is(toks, i)
            if result < 0:
                return True, i, True
            return True, result, False

        if tkey == "FIND":
            self._handle_find()
            return True, i, False

        if tkey == "SOURCE":
            self._handle_source()
            return True, i, False

        if tkey == "(":
            i = self._handle_paren_comment(toks, i)
            return True, i, False

        if tkey == "WORD":
            i = self._handle_word(toks, i)
            return True, i, False

        if tkey == "STATE":
            self._handle_state()
            return True, i, False

        if tkey == 'ABORT"':
            i, should_return = self._handle_abort_quote(toks, i)
            return True, i, should_return

        if tkey == "QUIT":
            self._handle_quit()
            return True, i, True

        if tkey == "IMMEDIATE":
            self._handle_immediate()
            return True, i, False

        if tkey == "]":
            self.state = COMPILE
            return True, i, False

        if tkey == "BASE":
            self._handle_base()
            return True, i, False

        if tkey == ">NUMBER":
            self._handle_to_number()
            return True, i, False

        if tkey == "ENVIRONMENT?":
            self._handle_environment_query()
            return True, i, False

        # Search order / wordlists (A5)
        if tkey == "WORDLIST":
            self._handle_wordlist()
            return True, i, False
        if tkey == "GET-CURRENT":
            self._handle_get_current()
            return True, i, False
        if tkey == "SET-CURRENT":
            self._handle_set_current()
            return True, i, False
        if tkey == "FORTH-WORDLIST":
            self._handle_forth_wordlist()
            return True, i, False
        if tkey == "DEFINITIONS":
            self._handle_definitions()
            return True, i, False
        if tkey == "FORTH":
            self._handle_forth()
            return True, i, False
        if tkey == "ONLY":
            self._handle_only()
            return True, i, False
        if tkey == "ALSO":
            self._handle_also()
            return True, i, False
        if tkey == "PREVIOUS":
            self._handle_previous()
            return True, i, False
        if tkey == ">ORDER":
            self._handle_to_order()
            return True, i, False
        if tkey == "GET-ORDER":
            self._handle_get_order()
            return True, i, False
        if tkey == "SET-ORDER":
            self._handle_set_order()
            return True, i, False
        if tkey == "SEARCH-WORDLIST":
            self._handle_search_wordlist()
            return True, i, False

        return False, i, False

    def _dispatch_compile(self, tkey, toks, i, toks_len):
        """Dispatch compile-mode words. Returns (handled, new_i, should_return)."""
        if tkey == "IMMEDIATE":
            # IMMEDIATE inside a colon body (CREATE IMMEDIATE ... DOES>) compiles
            # a call that marks the runtime-defined word immediate when the
            # defining word executes.
            self._emit_word(self.forth_wl["(IMMEDIATE)"])
            return True, i, False

        if tkey == "IF":
            self._compile_if()
            return True, i, False

        if tkey == "ELSE":
            if not self._compile_else():
                return True, i, True
            return True, i, False

        if tkey == "THEN":
            if not self._compile_then():
                return True, i, True
            return True, i, False

        if tkey == "DO":
            self._compile_do()
            return True, i, False

        if tkey == "?DO":
            self._compile_qdo()
            return True, i, False

        if tkey == "TO":
            if i >= toks_len:
                print "TO requires a name"
                return True, i, True
            name = toks[i]
            i += 1
            key = to_upper(name)
            if key not in self.value_addrs:
                print "TO: not a VALUE"
                return True, i, True
            self._emit_lit(W_IntObject(self.value_addrs[key]))
            self._emit_word(self.forth_wl["!"])
            return True, i, False

        if tkey == "IS":
            if i >= toks_len:
                print "IS requires a name"
                return True, i, True
            name = toks[i]
            i += 1
            key = to_upper(name)
            if key not in self.defer_ids:
                print "IS: not a DEFER"
                return True, i, True
            self._emit_lit(W_IntObject(self.defer_ids[key]))
            self._emit_word(self.forth_wl["(IS!)"])
            return True, i, False

        if tkey == "LOOP":
            if not self._compile_loop():
                return True, i, True
            return True, i, False

        if tkey == "+LOOP":
            if not self._compile_plusloop():
                return True, i, True
            return True, i, False

        if tkey == "BEGIN":
            self._compile_begin()
            return True, i, False

        if tkey == "WHILE":
            if not self._compile_while():
                return True, i, True
            return True, i, False

        if tkey == "REPEAT":
            if not self._compile_repeat():
                return True, i, True
            return True, i, False

        if tkey == "AGAIN":
            if not self._compile_again():
                return True, i, True
            return True, i, False

        if tkey == "UNTIL":
            if not self._compile_until():
                return True, i, True
            return True, i, False

        if tkey == "LEAVE":
            if not self._compile_leave():
                return True, i, True
            return True, i, False

        if tkey == "CASE":
            self._compile_case()
            return True, i, False

        if tkey == "OF":
            if not self._compile_of():
                return True, i, True
            return True, i, False

        if tkey == "ENDOF":
            if not self._compile_endof():
                return True, i, True
            return True, i, False

        if tkey == "ENDCASE":
            if not self._compile_endcase():
                return True, i, True
            return True, i, False

        if tkey == "[CHAR]":
            i = self._compile_char(toks, i)
            return True, i, False

        if tkey == "RECURSIVE":
            if self.current_name:
                thread = CodeThread([], [])
                self.define_colon(self.current_name, thread)
                self.current_predefined = True
            return True, i, False

        if tkey == "RECURSE":
            if self.current_name:
                name_upper = to_upper(self.current_name)
                if name_upper in self.dict and self.current_predefined:
                    self._emit_word(self.dict[name_upper])
                else:
                    thread = CodeThread([], [])
                    word = self.define_colon(self.current_name, thread)
                    self._emit_word(word)
                    self.current_predefined = True
            else:
                print "RECURSE outside of definition"
            return True, i, False

        if tkey == 'ABORT"':
            parsed_str, i = self._parse_string_until_quote(toks, i)
            w_str = W_StringObject(parsed_str)
            self._emit_lit(w_str)
            self._emit_lit(W_IntObject(len(parsed_str)))
            self._emit_word(self.wABORTQUOTE)
            return True, i, False

        if tkey == "[":
            self.state = INTERPRET
            return True, i, False

        if tkey == "STATE":
            self._emit_word(self.forth_wl["(STATE)"])
            return True, i, False

        if tkey == "POSTPONE":
            if i >= toks_len:
                print "POSTPONE requires a following word"
                return True, i, True
            name, i = self._read_tok(toks, i)
            name_upper = to_upper(name)
            control_kind = self._control_postpone_kind(name_upper)
            if control_kind >= 0:
                self._emit_lit(W_IntObject(control_kind))
                self._emit_word(self.forth_wl["(POSTPONE-CONTROL)"])
                return True, i, False
            word = self._lookup(name_upper)
            if word is not None:
                if word.immediate:
                    self._emit_word(word)
                else:
                    self._emit_lit(W_IntObject(word.wid))
                    self._emit_word(self.forth_wl["(POSTPONE)"])
            else:
                print "POSTPONE: word not found:", name
            return True, i, False

        if tkey == "[']":
            if i >= toks_len:
                print "['] requires a following word"
                return True, i, True
            name, i = self._read_tok(toks, i)
            name_upper = to_upper(name)
            word = self._lookup(name_upper)
            if word is not None:
                self._emit_lit(W_IntObject(word.wid))
            else:
                print "['] word not found:", name
            return True, i, False

        if tkey == "DOES>":
            # End the CREATE portion, then mark where the DOES> body begins. The
            # body (up to the trailing EXIT) becomes the runtime action bound to
            # each child word by CREATE.
            self._emit_word(self.wEXIT)
            self.does_ip_mark = self.cc_ptr
            return True, i, False

        return False, i, False

    # main outer interpreter
    @unroll_safe
    def interpret_line(self, line):
        try:
            self._interpret_line_tokens(line)
        except Abort:
            # QUIT / ABORT" abandon the rest of the input line; interpretation
            # resumes with the next line. The reset runs here, outside any
            # portal, so compiled frames never unwind over half-cleared state.
            self.inner.reset_after_abort()
            self.state = INTERPRET

    def _interpret_line_tokens(self, line):
        # Store the source line for SOURCE word
        self.source_buffer = line
        self.source_index = 0
        self.string_token_counts = {}

        toks, self.paren_depth = split_whitespace_stateful(line, self.paren_depth)
        toks_len = len(toks)
        self.toks = toks
        i = 0
        while i < toks_len:
            t, i = self._read_tok(toks, i)

            # Conditional compilation. While skipping, only [IF]/[ELSE]/[THEN]
            # adjust nesting -- everything else (including : ; definitions) is
            # dropped. This runs before any other handling and spans lines.
            ckey = to_upper(t)
            if self.cond_skipping:
                if ckey == "[IF]":
                    self.cond_skip_depth += 1
                elif ckey == "[ELSE]":
                    if self.cond_skip_depth == 0 and self.cond_skip_to_else:
                        self.cond_skipping = False
                elif ckey == "[THEN]":
                    if self.cond_skip_depth == 0:
                        self.cond_skipping = False
                    else:
                        self.cond_skip_depth -= 1
                continue
            if ckey == "[IF]":
                if self.inner.pop_ds_int() == 0:
                    self.cond_skipping = True
                    self.cond_skip_depth = 0
                    self.cond_skip_to_else = True
                continue
            if ckey == "[ELSE]":
                self.cond_skipping = True
                self.cond_skip_depth = 0
                self.cond_skip_to_else = False
                continue
            if ckey == "[THEN]":
                continue
            if ckey == "[DEFINED]" or ckey == "[UNDEFINED]":
                nm, i = self._read_tok(toks, i)
                present = self._lookup(to_upper(nm)) is not None
                if ckey == "[UNDEFINED]":
                    present = not present
                if present:
                    self.inner.push_ds_int(-1)
                else:
                    self.inner.push_ds_int(0)
                continue

            # Handle string / parse specials (case-insensitive dispatch).
            tup = to_upper(t)
            if tup == 'S"' or tup == 'C"':
                i = self._handle_s_quote(toks, i, t)
                continue

            if tup == '."':
                i = self._handle_dot_quote(toks, i, t)
                continue

            if tup == '.(':
                i = self._handle_dot_paren(toks, i, t)
                continue

            if tup == "CHAR":
                s, i = self._read_tok(toks, i)
                self.inner.push_ds_int(ord(s[0]))
                continue

            # :INLINE (fcp) defines an inlining word. Its true implementation
            # metaprograms with : PARSE SLITERAL POSTPONE EVALUATE, which needs a
            # compilable ':'; that machinery is absent here. fcp's own profiling
            # fallback is ': :inline : ;' -- i.e. a plain colon definition -- so
            # treat :INLINE as ':' (structurally identical, only the inlining perf
            # optimization is dropped). A dict stub for :INLINE is installed so
            # fcp's [UNDEFINED] :inline guard skips its metaprogramming variant.
            if tup == ":INLINE":
                if i >= toks_len:
                    print ":inline requires a name"
                    return
                self.current_name, i = self._read_tok(toks, i)
                self.noname_mode = False
                self.state = COMPILE
                self.does_ip_mark = -1
                self.current_predefined = False
                self.reset_code()
                continue

            # Handle ':' / ':NONAME' / ';' lexically (not as immediate words).
            if t == ':' or to_upper(t) == ":NONAME":
                if to_upper(t) == ":NONAME":
                    self.current_name = "NONAME"
                    self.noname_mode = True
                else:
                    if i >= toks_len:
                        print ": requires a name"
                        return
                    self.current_name, i = self._read_tok(toks, i)
                    self.noname_mode = to_upper(self.current_name) == "NONAME"
                self.state = COMPILE
                self.does_ip_mark = -1
                self.current_predefined = False
                self.reset_code()
                continue

            if t == ';':
                if self.state != COMPILE:
                    print "; outside definition"
                    continue
                self._finalize_definition()
                continue

            # Dispatch based on current state
            tkey = to_upper(t)

            if self.state == INTERPRET:
                handled, i, should_return = self._dispatch_interpret(tkey, toks, i, toks_len)
                if handled:
                    if should_return:
                        return
                    continue

            if self.state == COMPILE:
                handled, i, should_return = self._dispatch_compile(tkey, toks, i, toks_len)
                if handled:
                    if should_return:
                        return
                    continue

            # Default: look up word in dictionary and execute/compile.
            # order_is_default keeps the common case on the single-dict fast path.
            if self.order_is_default:
                w = self.dict.get(tkey, None)
            else:
                w = self._lookup(tkey)
            w = promote(w)
            if self.state == INTERPRET:
                # Publish the parse cursor so a defining word executed here can
                # consume the following token(s); pick up any advance afterward.
                self.inner.cell_store(self.to_in_addr, i)
                self._execute_or_push(w, t)
                i = self.inner.cell_fetch_int(self.to_in_addr)
            elif self.state == COMPILE:
                # Immediate words executed during compilation may parse too
                # (brainless's [DEF?] runs BL WORD FIND) -- keep the runtime
                # parse cursor in sync here as well.
                self.inner.cell_store(self.to_in_addr, i)
                self._compile_word_or_literal(w, t)
                i = self.inner.cell_fetch_int(self.to_in_addr)
            else:
                assert 0, "unreachable state"

    def _finalize_definition(self):
        tail_call_applied = False
        if self.cc_ptr > 0 and self.does_ip_mark < 0:
            last_word = self.current_code[self.cc_ptr - 1]
            # Check if it's a colon definition (not a primitive) and not a control word
            if last_word is not None and last_word.prim is None and last_word.thread is not None:
                # Replace the last word with TAILCALL and store the word in its literal slot
                self.cc_ptr -= 1  # Remove the last word
                self.lit_ptr -= 1  # Remove its literal
                # Emit TAILCALL with the word as literal
                wTAILCALL = self.dict.get("TAILCALL", None)
                if wTAILCALL is not None:
                    self.push_code(wTAILCALL)
                    self.push_lit(W_WordObject(last_word))
                    tail_call_applied = True

        if not tail_call_applied:
            self._emit_word(self.wEXIT)

        code = [self.current_code[idx] for idx in range(self.cc_ptr)]
        lits = [self.current_lits[idx] for idx in range(self.lit_ptr)]

        # If this definition contains a DOES>, carve out the DOES> body as a
        # standalone word; CREATE binds it as each child's runtime action.
        does_word = None
        if self.does_ip_mark >= 0:
            mark = self.does_ip_mark
            assert mark >= 0
            dlen = len(code) - mark
            dcode = [None] * dlen
            dlits = [ZERO] * dlen
            j = 0
            while j < dlen:
                w = code[mark + j]
                cl = lits[mark + j]
                # Branch-target literals (and LEAVE's patched loop-end) are
                # absolute indices into the parent code; rebase them onto the
                # carved body.
                if self._is_branch_target_word(w) or w is self.wLEAVE:
                    assert isinstance(cl, W_IntObject)
                    cl = W_IntObject(cl.intval - mark)
                dcode[j] = w
                dlits[j] = cl
                j += 1
            does_thread = CodeThread(dcode, dlits)
            does_word = Word("", prim=None, immediate=False, thread=does_thread)
            self.does_ip_mark = -1

        if self.noname_mode:
            thread = CodeThread(code, lits)
            thread.does_word = does_word
            self.inner.push_ds_int(Word("", thread=thread).wid)
            self.noname_mode = False
        else:
            name_upper = to_upper(self.current_name)
            if self.current_predefined and name_upper in self.dict:
                existing_word = self.dict[name_upper]
                if does_word is None:
                    # A DOES> body sits past the create-time code; splicing
                    # copies would tear it, so such words stay uninlined.
                    code, lits = self._inline_self_calls(code, lits, existing_word)
                thread = CodeThread(code, lits)
                thread.does_word = does_word
                existing_word.thread = thread
                self.last_word = existing_word
            else:
                thread = CodeThread(code, lits)
                thread.does_word = does_word
                self.define_colon(self.current_name, thread)

        self.state = INTERPRET
        self.current_name = ''
        self.reset_code()

    def _is_branch_target_word(self, w):
        """True for words whose literal is an instruction index in the current
        thread (must be relocated when code is spliced)."""
        return (w is self.wBR or w is self.w0BR or w is self.wQDO or
                w is self.wLOOP or w is self.wPLUSLOOP)

    def _inline_self_calls(self, code, lits, selfword):
        """Splice one copy of the body into each non-tail self-call site, so a
        recursive word does two levels of work per interpreter call frame. The
        copy's EXITs become branches past the copy; its self-calls stay real
        calls (which re-enter the inlined thread). Branch-target literals are
        relocated on both the original code and the copies."""
        n = len(code)
        if n == 0 or n > 48:
            return code, lits
        sites = 0
        i = 0
        while i < n:
            if code[i] is selfword:
                sites += 1
            i += 1
        # Cap the expansion: many-site words with large bodies (e.g. tak's four
        # sites) explode the trace length and run slower inlined.
        if sites == 0 or sites > 4 or n * (sites + 1) > 96:
            return code, lits
        wTAILCALL = self.dict.get("TAILCALL", None)
        # Pass 1: map every original instruction index (plus the one-past-end
        # position, a valid branch target) to its position in the new code.
        newpos = [0] * (n + 1)
        p = 0
        i = 0
        while i < n:
            newpos[i] = p
            if code[i] is selfword:
                p += n
            else:
                p += 1
            i += 1
        newpos[n] = p
        # Pass 2: emit into pre-sized lists (CodeThread requires non-resizable
        # lists), relocating branch targets.
        total = newpos[n]
        newcode = [None] * total
        newlits = [ZERO] * total
        out = 0
        i = 0
        while i < n:
            w = code[i]
            if w is selfword:
                base = newpos[i]
                copy_end = base + n
                j = 0
                while j < n:
                    cw = code[j]
                    cl = lits[j]
                    if cw is self.wEXIT:
                        newcode[out] = self.wBR
                        newlits[out] = W_IntObject(copy_end)
                    elif wTAILCALL is not None and cw is wTAILCALL:
                        assert isinstance(cl, W_WordObject)
                        if cl.word is selfword:
                            newcode[out] = self.wBR
                            newlits[out] = W_IntObject(base)
                        else:
                            # A mid-thread TAILCALL would misread its literal;
                            # demote to a plain call (it falls through to
                            # copy_end, which is the site's continuation).
                            newcode[out] = cl.word
                            newlits[out] = ZERO
                    elif self._is_branch_target_word(cw):
                        assert isinstance(cl, W_IntObject)
                        newcode[out] = cw
                        newlits[out] = W_IntObject(base + cl.intval)
                    else:
                        newcode[out] = cw
                        newlits[out] = cl
                    j += 1
                    out += 1
            elif self._is_branch_target_word(w):
                cl = lits[i]
                assert isinstance(cl, W_IntObject)
                target = cl.intval
                assert 0 <= target <= n
                newcode[out] = w
                newlits[out] = W_IntObject(newpos[target])
                out += 1
            else:
                newcode[out] = w
                newlits[out] = lits[i]
                out += 1
            i += 1
        assert out == total
        return newcode, newlits
