from rpyforth.objects import (
    W_StringObject, Word, CodeThread, W_IntObject, W_PtrObject, W_FloatObject, W_WordObject, ZERO, TRUE)
from rpyforth.primitives import install_primitives
from rpyforth.util import to_upper, split_whitespace

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
        self.string_token_count = 0  # Counter for string tokens (S", .") on current line

        # BASE variable address
        self.base_addr = 0  # Will be set after initialization

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

        # True while compiling a :NONAME definition (push xt instead of naming it).
        self.noname_mode = False

        # [IF]/[ELSE]/[THEN] conditional-compilation skip state (spans lines).
        self.cond_skipping = False
        self.cond_skip_depth = 0
        self.cond_skip_to_else = False

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

    def _handle_s_quote(self, toks, i):
        """Handle S" - parse string and push c-addr and length."""
        parsed_str = self._parse_string_at_occurrence('S"', self.string_token_count)
        self.string_token_count += 1
        i = self._skip_tokens_until_quote(toks, i)
        size = len(parsed_str)
        c_addr = self.inner.alloc_buf(parsed_str, size)
        self.inner.push_ds_int(c_addr)
        self.inner.push_ds_int(size)
        return i

    def _handle_dot_quote(self, toks, i):
        """Handle ." - parse string and print or compile."""
        parsed_str = self._parse_string_at_occurrence('."', self.string_token_count)
        self.string_token_count += 1
        i = self._skip_tokens_until_quote(toks, i)
        w_str = W_StringObject(parsed_str)
        if self.state == INTERPRET:
            self.inner.print_str(w_str)
        else:
            self._emit_lit(w_str)
            self._emit_word(self.wTYPE)
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
        code = [self.wLIT, self.dict["@"], self.wEXIT]
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
        saved_cnt = self.string_token_count
        for line in content.split('\n'):
            self.interpret_line(line)
        self.source_buffer = saved_buf
        self.source_index = saved_idx
        self.string_token_count = saved_cnt

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
        w = self.inner.buf[c_addr]
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
        code = [self.wLIT, self.dict["(DEFER)"], self.wEXIT]
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
        xt = self.inner.pop_ds()
        assert isinstance(xt, W_WordObject)
        self.inner.deferred_words[self.defer_ids[key]] = xt.word
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
        """Compile THEN."""
        if not self.ctrl:
            print "THEN without IF/ELSE"
            return False
        entry = self.ctrl.pop()
        if entry.kind != CTRL_IF and entry.kind != CTRL_ELSE:
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
        """Compile UNTIL (conditional branch back to BEGIN if false)."""
        if not self.ctrl:
            print "UNTIL without BEGIN"
            return False
        entry = self.ctrl.pop()
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
        self._emit_word(self.dict["OVER"])
        self._emit_word(self.dict["="])
        of_addr = self.cc_ptr
        self._emit_with_target(self.w0BR, 0)
        self._emit_word(self.dict["DROP"])
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
        self._emit_word(self.dict["DROP"])
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
        if w is not None:
            self.inner.execute_word_now(w)
        elif self._is_float(t):
            self.inner.push_ds_float(self._to_float(t))
        elif self._is_number(t):
            self.inner.push_ds_int(self._to_number(t))
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
        elif self._is_float(t):
            self._emit_lit(W_FloatObject(self._to_float(t)))
        elif self._is_number(t):
            self._emit_lit(W_IntObject(self._to_number(t)))
        else:
            print "UNKNOWN: " + t

    # System word handlers

    # FIND ( c-addr u -- c-addr 0 | xt 1 | xt -1 )
    def _handle_find(self):
        w_u = self.inner.pop_ds_int()
        w_caddr = self.inner.pop_ds_int()

        # Extract string from buffer
        ptr = w_caddr
        buf_entry = self.inner.buf[ptr]
        if buf_entry is not None and isinstance(buf_entry, W_StringObject):
            name = buf_entry.strval
        else:
            self.inner.push_ds_int(w_caddr)
            self.inner.push_ds_int(w_u)
            self.inner.push_ds_int(0)
            return

        name_upper = to_upper(name)
        if name_upper in self.dict:
            word = self.dict[name_upper]
            xt = W_WordObject(word)
            self.inner.push_ds(xt)
            if word.immediate:
                self.inner.push_ds_int(-1)
            else:
                self.inner.push_ds_int(1)
        else:
            self.inner.push_ds_int(w_caddr)
            self.inner.push_ds_int(w_u)
            self.inner.push_ds_int(0)

    # SOURCE ( -- c-addr u )
    def _handle_source(self):
        size = len(self.source_buffer)
        c_addr = self.inner.alloc_buf(self.source_buffer, size)
        self.inner.push_ds_int(c_addr)
        self.inner.push_ds_int(size)

    # >IN ( -- a-addr )
    def _handle_to_in(self):
        addr = self.inner.here
        self.inner.cell_store(addr, self.source_index)
        self.inner.push_ds_int(addr)

    # ' (tick) ( "<spaces>name" -- xt )
    def _handle_tick(self, toks, i):
        toks_len = len(toks)
        if i >= toks_len:
            print "' requires a following word"
            return i
        name, i = self._read_tok(toks, i)
        name_upper = to_upper(name)
        if name_upper in self.dict:
            word = self.dict[name_upper]
            xt = W_WordObject(word)
            self.inner.push_ds(xt)
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
        addr = self.inner.here
        self.inner.cell_store(addr, length)
        self.inner.here += self.inner.cell_size_bytes
        for ch in word_str:
            ch_addr = self.inner.here
            self.inner.cell_store(ch_addr, ord(ch))
            self.inner.here += 1
        self.inner.push_ds_int(addr)
        return i

    # STATE ( -- a-addr )
    def _handle_state(self):
        addr = self.inner.here
        state_val = -1 if self.state == COMPILE else 0
        self.inner.cell_store(addr, state_val)
        self.inner.push_ds_int(addr)

    # EVALUATE ( c-addr u -- )
    def _handle_evaluate(self):
        length = self.inner.pop_ds_int()
        c_addr = self.inner.pop_ds_int()
        buf_str = self.inner.buf[c_addr]
        assert isinstance(buf_str, W_StringObject)
        self.interpret_line(buf_str.strval)

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
            ch_obj = self.inner.cell_fetch(addr + j)
            assert isinstance(ch_obj, W_IntObject)
            ch = chr(ch_obj.intval)

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
        assert 0 <= c_addr < len(self.inner.buf)
        buf_entry = self.inner.buf[c_addr]
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
            self.inner.push_ds_int(2147483647)  # max signed single
            self.inner.push_ds_int(-1)
        elif query_upper == "MAX-U":
            self.inner.push_ds_int(-1)  # max unsigned (all bits set)
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
        """Return address of BASE variable."""
        if self.base_addr == 0:
            # Allocate a cell for BASE
            self.base_addr = self.inner.here
            self.inner.here += self.inner.cell_size_bytes
        # Store the current base at that address
        self.inner.cell_store(self.base_addr, self.inner.base)
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
        if self.last_word is not None:
            self.last_word.immediate = True
        else:
            print "IMMEDIATE: no word to mark"

    # Dispatch methods for interpret and compile modes

    def _dispatch_interpret(self, tkey, toks, i, toks_len):
        """Dispatch interpret-mode words. Returns (handled, new_i, should_return)."""
        if tkey == "VARIABLE" or tkey == "FVARIABLE":
            result = self._handle_variable(toks, i)
            if result < 0:
                return True, i, True
            return True, result, False

        if tkey == "2VARIABLE":
            result = self._handle_2variable(toks, i)
            if result < 0:
                return True, i, True
            return True, result, False

        if tkey == "CONSTANT" or tkey == "FCONSTANT":
            result = self._handle_constant(toks, i)
            if result < 0:
                return True, i, True
            return True, result, False

        if tkey == "CREATE":
            result = self._handle_create(toks, i)
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

        if tkey == "DEFER":
            result = self._handle_defer(toks, i)
            if result < 0:
                return True, i, True
            return True, result, False

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

        if tkey == ">IN":
            self._handle_to_in()
            return True, i, False

        if tkey == "'":
            i = self._handle_tick(toks, i)
            return True, i, False

        if tkey == "(":
            i = self._handle_paren_comment(toks, i)
            return True, i, False

        if tkey == "COUNT":
            self._handle_count()
            return True, i, False

        if tkey == "WORD":
            i = self._handle_word(toks, i)
            return True, i, False

        if tkey == "STATE":
            self._handle_state()
            return True, i, False

        if tkey == "EVALUATE":
            self._handle_evaluate()
            return True, i, False

        if tkey == "ABORT":
            self._handle_abort()
            return True, i, True

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

        return False, i, False

    def _dispatch_compile(self, tkey, toks, i, toks_len):
        """Dispatch compile-mode words. Returns (handled, new_i, should_return)."""
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
            self._emit_word(self.dict["!"])
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
            self._emit_word(self.dict["(IS!)"])
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
            return True, i, False

        if tkey == "RECURSE":
            if self.current_name:
                name_upper = to_upper(self.current_name)
                if name_upper in self.dict:
                    self._emit_word(self.dict[name_upper])
                else:
                    thread = CodeThread([], [])
                    word = self.define_colon(self.current_name, thread)
                    self._emit_word(word)
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

        if tkey == "POSTPONE":
            if i >= toks_len:
                print "POSTPONE requires a following word"
                return True, i, True
            name, i = self._read_tok(toks, i)
            name_upper = to_upper(name)
            if name_upper in self.dict:
                word = self.dict[name_upper]
                if word.immediate:
                    self._emit_word(word)
                else:
                    self._emit_lit(W_WordObject(word))
                    self._emit_word(self.dict["EXECUTE"])
            else:
                print "POSTPONE: word not found:", name
            return True, i, False

        if tkey == "[']":
            if i >= toks_len:
                print "['] requires a following word"
                return True, i, True
            name, i = self._read_tok(toks, i)
            name_upper = to_upper(name)
            if name_upper in self.dict:
                word = self.dict[name_upper]
                self._emit_lit(W_WordObject(word))
            else:
                print "['] word not found:", name
            return True, i, False

        if tkey == "DOES>":
            self._emit_word(self.wEXIT)
            does_ip = self.cc_ptr
            if self.last_word is not None:
                self.last_word.does_ip = does_ip
            if self.current_name:
                name_upper = to_upper(self.current_name)
                if name_upper not in self.dict:
                    thread = CodeThread([], [])
                    self.define_colon(self.current_name, thread)
                w = self.dict[name_upper]
                self._emit_word(w)
            else:
                print "DOES> outside definition"
            return True, i, False

        return False, i, False

    # main outer interpreter
    @unroll_safe
    def interpret_line(self, line):
        # Store the source line for SOURCE word
        self.source_buffer = line
        self.source_index = 0
        self.string_token_count = 0  # Reset counter for each new line

        toks = split_whitespace(line)
        toks_len = len(toks)
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
                present = to_upper(nm) in self.dict
                if ckey == "[UNDEFINED]":
                    present = not present
                if present:
                    self.inner.push_ds_int(-1)
                else:
                    self.inner.push_ds_int(0)
                continue

            # Handle string literals (case-sensitive)
            if t == 'S"':
                i = self._handle_s_quote(toks, i)
                continue

            if t == '."':
                i = self._handle_dot_quote(toks, i)
                continue

            if t == "CHAR":
                s, i = self._read_tok(toks, i)
                self.inner.push_ds_int(ord(s[0]))
                continue

            # Handle ':' and ';' lexically (not as immediate words)
            if t == ':':
                if i >= toks_len:
                    print ": requires a name"
                    return
                self.state = COMPILE
                self.current_name, i = self._read_tok(toks, i)
                self.noname_mode = to_upper(self.current_name) == "NONAME"
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

            # Default: look up word in dictionary and execute/compile
            w = self.dict.get(tkey, None)
            w = promote(w)
            if self.state == INTERPRET:
                self._execute_or_push(w, t)
            elif self.state == COMPILE:
                self._compile_word_or_literal(w, t)
            else:
                assert 0, "unreachable state"

    def _finalize_definition(self):
        tail_call_applied = False
        if self.cc_ptr > 0:
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
        thread = CodeThread(code, lits)

        if self.noname_mode:
            self.inner.push_ds(W_WordObject(Word("", thread=thread)))
            self.noname_mode = False
        else:
            name_upper = to_upper(self.current_name)
            if name_upper in self.dict:
                existing_word = self.dict[name_upper]
                existing_word.thread = thread
            else:
                self.define_colon(self.current_name, thread)

        self.state = INTERPRET
        self.current_name = ''
        self.reset_code()
