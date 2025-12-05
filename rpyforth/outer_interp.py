from rpyforth.objects import (
    W_StringObject, Word, CodeThread, W_IntObject, W_PtrObject, W_FloatObject, W_WordObject, ZERO, TRUE)
from rpyforth.primitives import install_primitives
from rpyforth.util import to_upper, split_whitespace

from rpython.rlib.rfile import create_stdio
from rpython.rlib.jit import elidable, unroll_safe, promote

INTERPRET = 0
COMPILE   = 1

# Control stack entry kinds
CTRL_IF   = 0
CTRL_ELSE = 1
CTRL_DO   = 2
CTRL_BEGIN = 3
CTRL_WHILE = 4

class CtrlEntry(object):
    """Control stack entry for compilation-time control structures.

    RPython-friendly class to avoid tuple unpacking and string comparisons.
    """
    def __init__(self, kind, index):
        self.kind = kind    # int: CTRL_IF, CTRL_ELSE, or CTRL_DO
        self.index = index  # int: position in scurrent_code for patching
        self.leave_addrs = []  # list of LEAVE positions to patch (for DO loops)

class OuterInterpreter(object):
    _immutable_fields_ = ['wBR', 'w0BR', 'wLIT', 'wEXIT', 'wDO', 'wLOOP', 'wPLUSLOOP', 'wLEAVE', 'wTYPE', 'wUNLOOP', 'wABORTQUOTE']

    def __init__(self, inner):
        self.inner = inner
        self.dict = {}         # dictionary is owned here (case-insensitive by uppercase keys)
        self.state = INTERPRET # state for compilation
        self.comment = False
        self.current_name = ''
        self.last_word = None  # Last defined word (for IMMEDIATE)

        # Input source tracking for SOURCE and >IN
        self.source_buffer = ''  # Current input line
        self.source_index = 0    # Current parse position (>IN)

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
        self.wLOOP = self.dict["(LOOP)"]
        self.wPLUSLOOP = self.dict["(+LOOP)"]
        self.wUNLOOP = self.dict["UNLOOP"]
        self.wLEAVE = self.dict["LEAVE"]
        self.wTYPE = self.dict["TYPE"]
        self.wABORTQUOTE = self.dict['(ABORT")']

        # Define LITERAL as an immediate word (for POSTPONE to find it)
        self._define_literal_word()

    def reset_code(self):
        self.current_code = [None] * 128
        self.current_lits = [None] * 128
        self.cc_ptr = 0
        self.lit_ptr = 0

    def push_code(self, w):
        assert self.cc_ptr < len(self.current_code)
        self.current_code[self.cc_ptr] = w
        self.cc_ptr += 1

    def pop_code(self):
        assert self.cc_ptr > 0
        self.cc_ptr -= 1
        return self.current_code[self.cc_ptr]

    def push_lit(self, w):
        assert self.lit_ptr < len(self.current_lits)
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
        # LITERAL is IMMEDIATE and should compile LIT <value>
        # We can't use a real primitive because it needs access to compilation state
        # So we use a dummy word that will be handled specially
        w = Word("LITERAL", prim=None, immediate=True, thread=None)
        self.dict["LITERAL"] = w
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
        # Python's float() handles the format we need
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
    def _parse_string_until_quote(self, toks, i):
        """Parse tokens until closing quote, return (parsed_string, new_index)."""
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
        parsed_str, i = self._parse_string_until_quote(toks, i)
        size = len(parsed_str)
        c_addr = self.inner.alloc_buf(parsed_str, size)
        self.inner.push_ds_int(c_addr)
        self.inner.push_ds_int(size)
        return i

    def _handle_dot_quote(self, toks, i):
        """Handle ." - parse string and print or compile."""
        parsed_str, i = self._parse_string_until_quote(toks, i)
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
        if self.inner.ds_ptr_ints > 0:
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

    def _compile_word_or_literal(self, w, t):
        """Compile word or literal in COMPILE mode."""
        if w is not None:
            # Check if word is immediate - if so, execute it now
            if w.immediate:
                # Special case for LITERAL: it needs compilation-time handling
                if w.name == "LITERAL" or to_upper(w.name) == "LITERAL":
                    # Pop value from stack and compile it as a literal
                    # Check the integer stack first (most common case)
                    if self.inner.ds_ptr_ints > 0:
                        # Unboxed integer
                        intval = self.inner.pop_ds_int()
                        self._emit_lit(W_IntObject(intval))
                    elif self.inner.ds_ptr_floats > 0:
                        # Unboxed float
                        floatval = self.inner.pop_ds_float()
                        self._emit_lit(W_FloatObject(floatval))
                    elif self.inner.ds_ptr_locals > 0:
                        # Boxed object
                        val = self.inner.pop_ds()
                        self._emit_lit(val)
                    else:
                        print "LITERAL: stack underflow"
                else:
                    # Other immediate words: execute them
                    self.inner.execute_word_now(w)
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
            self.inner.ds_ptr_ints = 0
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

    # main outer interpreter
    @unroll_safe
    def interpret_line(self, line):
        # Store the source line for SOURCE word
        self.source_buffer = line
        self.source_index = 0

        toks = split_whitespace(line)
        toks_len = len(toks)
        i = 0
        while i < toks_len:
            t, i = self._read_tok(toks, i)

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

            # handle ':' and ';' lexically (not as immediate words)
            if t == ':':
                if i >= toks_len:
                    print ": requires a name"
                    return
                self.state = COMPILE
                self.current_name, i = self._read_tok(toks, i)
                self.reset_code()
                continue

            if t == ';':
                if self.state != COMPILE:
                    print "; outside definition"
                    continue

                # append EXIT and install
                self._emit_word(self.wEXIT)
                # Create new lists with only the used portion (RPython needs proper list sizes)
                code = [self.current_code[idx] for idx in range(self.cc_ptr)]
                lits = [self.current_lits[idx] for idx in range(self.lit_ptr)]
                thread = CodeThread(code, lits)

                # Check if word already exists (RECURSIVE was used)
                name_upper = to_upper(self.current_name)
                if name_upper in self.dict:
                    # Update existing word's thread
                    existing_word = self.dict[name_upper]
                    existing_word.thread = thread
                else:
                    self.define_colon(self.current_name, thread)

                # reset
                self.state = INTERPRET
                self.current_name = ''
                self.reset_code()
                continue

            tkey = to_upper(t)
            if self.state == INTERPRET:
                if tkey == "VARIABLE" or tkey == "FVARIABLE":
                    result = self._handle_variable(toks, i)
                    if result < 0:
                        return
                    i = result
                    continue

                if tkey == "2VARIABLE":
                    result = self._handle_2variable(toks, i)
                    if result < 0:
                        return
                    i = result
                    continue

                if tkey == "CONSTANT" or tkey == "FCONSTANT":
                    result = self._handle_constant(toks, i)
                    if result < 0:
                        return
                    i = result
                    continue

                if tkey == "CREATE":
                    result = self._handle_create(toks, i)
                    if result < 0:
                        return
                    i = result
                    continue

                if tkey == "FIND":
                    self._handle_find()
                    continue

                if tkey == "SOURCE":
                    self._handle_source()
                    continue

                if tkey == ">IN":
                    self._handle_to_in()
                    continue

                if tkey == "'":
                    i = self._handle_tick(toks, i)
                    continue

                if tkey == "(":
                    i = self._handle_paren_comment(toks, i)
                    continue

                if tkey == "COUNT":
                    self._handle_count()
                    continue

                if tkey == "WORD":
                    i = self._handle_word(toks, i)
                    continue

                if tkey == "STATE":
                    self._handle_state()
                    continue

                if tkey == "EVALUATE":
                    self._handle_evaluate()
                    continue

                if tkey == "ABORT":
                    self._handle_abort()
                    return

                if tkey == 'ABORT"':
                    i, should_return = self._handle_abort_quote(toks, i)
                    if should_return:
                        return
                    continue

                if tkey == "QUIT":
                    self._handle_quit()
                    return

                if tkey == "IMMEDIATE":
                    if self.last_word is not None:
                        self.last_word.immediate = True
                    else:
                        print "IMMEDIATE: no word to mark"
                    continue

                if tkey == "]":
                    self.state = COMPILE
                    continue

                if tkey == "BASE":
                    # BASE returns an address; we use a special cell in memory
                    if self.base_addr == 0:
                        # Allocate a cell for BASE
                        self.base_addr = self.inner.here
                        self.inner.here += self.inner.cell_size_bytes
                    # Store the current base at that address
                    self.inner.cell_store(self.base_addr, self.inner.base)
                    self.inner.push_ds_int(self.base_addr)
                    continue

                if tkey == ">NUMBER":
                    # >NUMBER ( ud1 c-addr1 u1 -- ud2 c-addr2 u2 )
                    # Convert string to number according to BASE
                    u1 = self.inner.pop_ds_int()
                    c_addr1 = self.inner.pop_ds_int()
                    ud1_hi = self.inner.pop_ds_int()
                    ud1_lo = self.inner.pop_ds_int()

                    # assert isinstance(u1, W_IntObject)
                    # assert isinstance(c_addr1, W_IntObject)
                    # assert isinstance(ud1_hi, W_IntObject)
                    # assert isinstance(ud1_lo, W_IntObject)

                    base = self.inner.base
                    addr = c_addr1
                    length = u1
                    value = ud1_lo

                    # Process each character
                    chars_processed = 0
                    for j in range(length):
                        ch_obj = self.inner.cell_fetch(addr + j)
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
                    continue

                if tkey == "ENVIRONMENT?":
                    # ENVIRONMENT? ( c-addr u -- false | i*x true )
                    # Query environmental information
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
                    continue

            if self.state == COMPILE:
                if tkey == "IF":
                    self._compile_if()
                    continue

                if tkey == "ELSE":
                    if not self._compile_else():
                        return
                    continue

                if tkey == "THEN":
                    if not self._compile_then():
                        return
                    continue

                if tkey == "DO":
                    self._compile_do()
                    continue

                if tkey == "LOOP":
                    if not self._compile_loop():
                        return
                    continue

                if tkey == "+LOOP":
                    if not self._compile_plusloop():
                        return
                    continue

                if tkey == "BEGIN":
                    self._compile_begin()
                    continue

                if tkey == "WHILE":
                    if not self._compile_while():
                        return
                    continue

                if tkey == "REPEAT":
                    if not self._compile_repeat():
                        return
                    continue

                if tkey == "AGAIN":
                    if not self._compile_again():
                        return
                    continue

                if tkey == "UNTIL":
                    if not self._compile_until():
                        return
                    continue

                if tkey == "LEAVE":
                    if not self._compile_leave():
                        return
                    continue

                if tkey == "[CHAR]":
                    i = self._compile_char(toks, i)
                    continue

                if tkey == "RECURSIVE":
                    # Make the current word visible to itself during compilation
                    # Add word to dictionary now so it can reference itself
                    # The thread will be finalized when ; is reached
                    if self.current_name:
                        # Create word with empty thread as placeholder
                        thread = CodeThread([], [])
                        self.define_colon(self.current_name, thread)
                    continue

                if tkey == "RECURSE":
                    # Compile a call to the current word being defined
                    if self.current_name:
                        name_upper = to_upper(self.current_name)
                        if name_upper in self.dict:
                            # Word already exists (RECURSIVE was used)
                            self._emit_word(self.dict[name_upper])
                        else:
                            # Create word now so we can compile a reference to it
                            thread = CodeThread([], [])
                            word = self.define_colon(self.current_name, thread)
                            self._emit_word(word)
                    else:
                        print "RECURSE outside of definition"
                    continue

                if tkey == 'ABORT"':
                    # Compile ABORT" in compile mode
                    # Parse the message string
                    parsed_str, i = self._parse_string_until_quote(toks, i)
                    w_str = W_StringObject(parsed_str)
                    # Compile: push string, push length, call (ABORT")
                    self._emit_lit(w_str)
                    self._emit_lit(W_IntObject(len(parsed_str)))
                    self._emit_word(self.wABORTQUOTE)
                    continue

                if tkey == "[":
                    # Switch to interpret mode during compilation
                    self.state = INTERPRET
                    continue

                # Note: LITERAL is now handled as an immediate word in the dictionary
                # so it will be found by the word lookup below

                if tkey == "POSTPONE":
                    # Compile execution of the next word (even if immediate)
                    if i >= toks_len:
                        print "POSTPONE requires a following word"
                        return
                    name, i = self._read_tok(toks, i)
                    name_upper = to_upper(name)
                    if name_upper in self.dict:
                        word = self.dict[name_upper]
                        if word.immediate:
                            # For immediate words, compile code that will compile them
                            # Special case for LITERAL since it has custom handling
                            if name_upper == "LITERAL":
                                # POSTPONE LITERAL: pop value now and compile it as a literal
                                # This means when float-literal executes, it will compile (LIT <value>)
                                val = self.inner.pop_ds()
                                self._emit_lit(val)
                            else:
                                # For other immediate words, compile them to be executed
                                self._emit_word(word)
                        else:
                            # For non-immediate words, compile code to compile them
                            # Push the xt and compile EXECUTE
                            self._emit_lit(W_WordObject(word))
                            self._emit_word(self.dict["EXECUTE"])
                    else:
                        print "POSTPONE: word not found:", name
                    continue

                if tkey == "[']":
                    # Compile the xt of the next word as a literal
                    if i >= toks_len:
                        print "['] requires a following word"
                        return
                    name, i = self._read_tok(toks, i)
                    name_upper = to_upper(name)
                    if name_upper in self.dict:
                        word = self.dict[name_upper]
                        self._emit_lit(W_WordObject(word))
                    else:
                        print "['] word not found:", name
                    continue

                if tkey == "DOES>":
                    # DOES> ends the CREATE part and starts the DOES> behavior
                    # Compile EXIT to end the CREATE-time behavior
                    self._emit_word(self.wEXIT)
                    # Mark the current position as the DOES> entry point
                    # Store the does_ip for the last defined word
                    does_ip = self.cc_ptr
                    if self.last_word is not None:
                        # Store does_ip in the word for later use
                        self.last_word.does_ip = does_ip
                    # Compile a call to the word currently being defined
                    if self.current_name:
                        name_upper = to_upper(self.current_name)
                        # If not already in dictionary, add it now
                        if name_upper not in self.dict:
                            thread = CodeThread([], [])
                            self.define_colon(self.current_name, thread)
                        # Emit the current word
                        w = self.dict[name_upper]
                        self._emit_word(w)
                    else:
                        print "RECURSE outside definition"
                    continue

            w = self.dict.get(tkey, None)
            w = promote(w)
            if self.state == INTERPRET:
                self._execute_or_push(w, t)
            elif self.state == COMPILE:
                self._compile_word_or_literal(w, t)
            else:
                assert 0, "unreachable state"
