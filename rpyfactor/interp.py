"""Main interpreter loop for rpyfactor.

Flat dispatch loop with an explicit call stack. The Interpreter itself is
the virtualizable (rpyforth pattern): the fragment cache lives in fields on
this object, declared in STACK_FRAGMENT_VIRTUALIZABLES, and every stack
access flows through the portal's own red variable ``self``. There is no
separate stack object on the hot path -- re-fetching the cache through a
field would escape the virtualizable and force it on every primitive.
"""

from rpython.rlib.jit import JitDriver, elidable
from rpython.rlib.debug import make_sure_not_resized

from rpyfactor.stackrep import NaiveStack, FragmentBase
from rpyfactor.metastack import (
    USE_STACK_FRAGMENT,
    USE_STACK_VABLE,
    STACK_FRAGMENT_VIRTUALIZABLES,
)
from rpyfactor.program import Word, CallWord, LitInt, LitBool, item_to_value, is_literal_item
from rpyfactor.values import (
    W_List, W_Cons, W_Quotation,
    FactorError,
    nil_list, w_list_from_items,
)
from rpyfactor.primitives import (
    is_combinator, is_primitive, _pop_symbol,
    prim_add, prim_sub, prim_mul, prim_div, prim_rem,
    prim_lt, prim_le, prim_gt, prim_ge, prim_eq, prim_ne,
    prim_and, prim_or, prim_not,
    prim_dup, prim_2dup, prim_2drop, prim_2over,
    prim_swap, prim_nip, prim_pop, prim_over,
    prim_rot, prim_rolldown, prim_rollup, prim_dupd, prim_swapd, prim_pick, prim_3dup,
    prim_stack, prim_unstack,
    prim_cons, prim_uncons, prim_swons, prim_first, prim_rest,
    prim_concat, prim_size, prim_null, prim_small, prim_reverse,
    prim_nil, prim_succ, prim_pred, prim_dot, prim_put, prim_putchars,
    prim_intern, prim_name, prim_clock,
    prim_new_array, prim_nth, prim_set_nth,
)


CS_DEPTH = 65536

KIND_RET = 0
KIND_TIMES = 1
KIND_DIP = 2
KIND_IFTE_COND = 3
KIND_WHILE = 4
KIND_STEP = 5
KIND_MAP = 6
KIND_FOLD = 7
KIND_FILTER = 8
KIND_LINREC = 9
KIND_BINREC = 10
KIND_PRIMREC = 11

IP_RESUME_PRIMREC = -1000000
IP_RESUME_LINREC = -2000000
IP_RESUME_BINREC = -3000000


# Program lists are parse-time constants (never mutated after the lexer
# builds them), so length and item reads fold to constants inside traces:
# the whole isinstance/name-comparison dispatch disappears at trace time.
@elidable
def _prog_len(program):
    return len(program)


@elidable
def _prog_item(program, ip):
    return program[ip]


def _jit_location(ip, program):
    if program is not None and ip < len(program):
        item = program[ip]
        if isinstance(item, CallWord):
            return "ip=%d %s" % (ip, item.name)
    return "ip=%d" % ip


if USE_STACK_FRAGMENT and USE_STACK_VABLE:
    jitdriver = JitDriver(
        greens=['ip', 'program'],
        reds=['self'],
        virtualizables=['self'],
        get_printable_location=_jit_location,
    )
else:
    jitdriver = JitDriver(
        greens=['ip', 'program'],
        reds=['self'],
        get_printable_location=_jit_location,
    )


_CS_IMMUTABLE = [
    "cs_progs", "cs_ips", "cs_kinds",
    "cs_body", "cs_body2", "cs_body3", "cs_body4",
    "cs_n", "cs_i", "cs_val", "cs_val2",
    "cs_list", "cs_snap", "cs_out",
]


class Interpreter(FragmentBase):
    # Array references are allocated once and never reassigned; declaring
    # them immutable lets the JIT hoist the array-pointer loads. The frame_*
    # arrays are deliberately absent: they belong to the virtualizable set.
    if USE_STACK_FRAGMENT:
        _immutable_fields_ = _CS_IMMUTABLE + ["spill_i", "spill_t", "spill_o"]
    else:
        _immutable_fields_ = _CS_IMMUTABLE

    if USE_STACK_FRAGMENT and USE_STACK_VABLE:
        _virtualizable_ = STACK_FRAGMENT_VIRTUALIZABLES

    def __init__(self):
        self.dict = {}
        self.words = {}
        self._init_call_stack()
        if USE_STACK_FRAGMENT:
            self.init_fragment_fields()
        else:
            self.stack = NaiveStack()

    if USE_STACK_FRAGMENT:
        def st(self):
            return self
    else:
        def st(self):
            return self.stack

    def _init_call_stack(self):
        self.cs_progs = [None] * CS_DEPTH
        make_sure_not_resized(self.cs_progs)
        self.cs_ips = [0] * CS_DEPTH
        make_sure_not_resized(self.cs_ips)
        self.cs_kinds = [0] * CS_DEPTH
        make_sure_not_resized(self.cs_kinds)
        self.cs_body = [None] * CS_DEPTH
        make_sure_not_resized(self.cs_body)
        self.cs_body2 = [None] * CS_DEPTH
        make_sure_not_resized(self.cs_body2)
        self.cs_body3 = [None] * CS_DEPTH
        make_sure_not_resized(self.cs_body3)
        self.cs_body4 = [None] * CS_DEPTH
        make_sure_not_resized(self.cs_body4)
        self.cs_n = [0] * CS_DEPTH
        make_sure_not_resized(self.cs_n)
        self.cs_i = [0] * CS_DEPTH
        make_sure_not_resized(self.cs_i)
        self.cs_val = [None] * CS_DEPTH
        make_sure_not_resized(self.cs_val)
        self.cs_val2 = [None] * CS_DEPTH
        make_sure_not_resized(self.cs_val2)
        self.cs_list = [None] * CS_DEPTH
        make_sure_not_resized(self.cs_list)
        self.cs_snap = [None] * CS_DEPTH
        make_sure_not_resized(self.cs_snap)
        self.cs_out = [None] * CS_DEPTH
        make_sure_not_resized(self.cs_out)
        self.cs_ptr = 0

    def load_definitions(self, defs):
        for name, body in defs.items():
            self.dict[name] = body
            if name in self.words:
                self.words[name].redefine(body)
            else:
                self.words[name] = Word(body)

    def lookup(self, name):
        if name in self.dict:
            return self.dict[name]
        if is_primitive(name) or is_combinator(name) or name == "body":
            return None
        raise FactorError("unknown word: %s" % name)

    def _word_cell(self, name):
        if name in self.words:
            return self.words[name]
        if name in self.dict:
            cell = Word(self.dict[name])
            self.words[name] = cell
            return cell
        return None

    def run(self, program):
        self.cs_ptr = 0
        self._execute(program)

    def run_source(self, text):
        from rpyfactor.lexer import parse_source
        defs, program = parse_source(text)
        self.load_definitions(defs)
        self.run(program)

    def run_quot(self, program):
        self.cs_ptr = 0
        self._execute_nested(program)

    def _push_frame(self, kind, ret_prog, ret_ip):
        ptr = self.cs_ptr
        if ptr >= CS_DEPTH:
            raise FactorError("call stack overflow")
        assert ptr >= 0
        self.cs_kinds[ptr] = kind
        self.cs_progs[ptr] = ret_prog
        self.cs_ips[ptr] = ret_ip
        self.cs_body[ptr] = None
        self.cs_body2[ptr] = None
        self.cs_body3[ptr] = None
        self.cs_body4[ptr] = None
        self.cs_n[ptr] = 0
        self.cs_i[ptr] = 0
        self.cs_val[ptr] = None
        self.cs_val2[ptr] = None
        self.cs_list[ptr] = None
        self.cs_snap[ptr] = None
        self.cs_out[ptr] = None
        self.cs_ptr = ptr + 1
        return ptr

    def _enter_ret(self, body, program, ip):
        if ip == _prog_len(program) and _prog_len(body) != 0:
            return body, 0
        return self._enter(body, program, ip, KIND_RET)

    def _execute_nested(self, body):
        """Run body as a top-level nested program (fragment + RET frame)."""
        self._push_frame(KIND_RET, None, 0)
        self.st().push_fragment()
        self._execute(body)

    def _enter(self, body, ret_prog, ret_ip, kind):
        # An empty body runs zero items: entering it would only push a frame
        # and a fragment that the very next dispatch pops again -- and every
        # such entry is a can_enter_jit loop header, so hot empty ifte/branch
        # arms (`[ ]`) become pathological trace heads. Skip the call.
        if kind == KIND_RET and _prog_len(body) == 0:
            # ret_prog can be a None+resume-ip marker (linrec/binrec/primrec
            # continuations); _resume_ret decodes those.
            return self._resume_ret(ret_prog, ret_ip)
        self._push_frame(kind, ret_prog, ret_ip)
        self.st().push_fragment()
        return body, 0

    def _execute(self, program):
        ip = 0
        while True:
            jitdriver.jit_merge_point(ip=ip, program=program, self=self)
            if ip >= _prog_len(program):
                program, ip = self._on_return()
                if program is None:
                    break
                if ip == 0:
                    jitdriver.can_enter_jit(ip=ip, program=program, self=self)
                continue
            item = _prog_item(program, ip)
            ip += 1
            st = self.st()
            if isinstance(item, LitInt):
                st.push_int(item.n)
            elif isinstance(item, LitBool):
                st.push_int(1 if item.b else 0)
            elif is_literal_item(item):
                st.push(item_to_value(item))
            elif isinstance(item, CallWord):
                program, ip = self._call_word(item, program, ip)
                # A call lands at ip 0 of the callee body: a loop header for
                # the JIT. Without this, recursive words never start tracing
                # (the return path only covers times/step-style iteration).
                if ip == 0:
                    jitdriver.can_enter_jit(ip=ip, program=program, self=self)
            else:
                raise FactorError("bad program item")

    def _resume_ret(self, prog, ip):
        if prog is not None:
            return prog, ip
        if ip <= IP_RESUME_BINREC:
            return self._binrec_after_child(-ip + IP_RESUME_BINREC)
        if ip <= IP_RESUME_LINREC:
            return self._linrec_after_child(-ip + IP_RESUME_LINREC)
        if ip <= IP_RESUME_PRIMREC:
            return self._primrec_after_child(-ip + IP_RESUME_PRIMREC)
        return None, 0

    def _on_return(self):
        ptr = self.cs_ptr - 1
        if ptr < 0:
            # Top-level program finished: no fragment was pushed for it.
            return None, 0
        assert ptr >= 0
        self.st().pop_fragment_commit()
        kind = self.cs_kinds[ptr]

        if kind == KIND_RET:
            prog = self.cs_progs[ptr]
            ip = self.cs_ips[ptr]
            self.cs_ptr = ptr
            return self._resume_ret(prog, ip)

        if kind == KIND_TIMES:
            i = self.cs_i[ptr] + 1
            if i < self.cs_n[ptr]:
                self.cs_i[ptr] = i
                self.st().push_fragment()
                return self.cs_body[ptr], 0
            prog = self.cs_progs[ptr]
            ip = self.cs_ips[ptr]
            self.cs_ptr = ptr
            return self._resume_ret(prog, ip)

        if kind == KIND_DIP:
            self.st().push(self.cs_val[ptr])
            prog = self.cs_progs[ptr]
            ip = self.cs_ips[ptr]
            self.cs_ptr = ptr
            return self._resume_ret(prog, ip)

        if kind == KIND_IFTE_COND:
            flag = self.st().pop_truthy()
            snap = self.cs_snap[ptr]
            then_q = self.cs_body[ptr]
            else_q = self.cs_body2[ptr]
            ret_prog = self.cs_progs[ptr]
            ret_ip = self.cs_ips[ptr]
            self.cs_ptr = ptr
            self._restore_stack(snap)
            if flag:
                return self._enter_ret(then_q, ret_prog, ret_ip)
            return self._enter_ret(else_q, ret_prog, ret_ip)

        if kind == KIND_WHILE:
            return self._while_continue(ptr)

        if kind == KIND_STEP:
            return self._step_continue(ptr)

        if kind == KIND_MAP:
            return self._map_continue(ptr)

        if kind == KIND_FOLD:
            return self._fold_continue(ptr)

        if kind == KIND_FILTER:
            return self._filter_continue(ptr)

        if kind == KIND_LINREC:
            return self._linrec_continue(ptr)

        if kind == KIND_BINREC:
            return self._binrec_continue(ptr)

        if kind == KIND_PRIMREC:
            # Primrec frame is only a marker; child returns via KIND_RET resume.
            raise FactorError("primrec frame should not end a program")

        raise FactorError("bad call-stack kind %d" % kind)

    def _call_word(self, item, program, ip):
        name = item.name
        if name == "body":
            self.prim_body()
            return program, ip
        if is_combinator(name):
            return self._dispatch_combinator(name, program, ip)
        if self._dispatch_prim(name):
            return program, ip
        cell = item.cell
        if cell is None:
            cell = self._word_cell(name)
            if cell is not None:
                item.cell = cell
        if cell is not None:
            return self._enter_ret(cell.body, program, ip)
        raise FactorError("unknown word: %s" % name)

    def _dispatch_prim(self, name):
        st = self.st()
        if name == "+":
            prim_add(st)
        elif name == "-":
            prim_sub(st)
        elif name == "*":
            prim_mul(st)
        elif name == "/":
            prim_div(st)
        elif name == "rem" or name == "mod":
            prim_rem(st)
        elif name == "<":
            prim_lt(st)
        elif name == "<=":
            prim_le(st)
        elif name == ">":
            prim_gt(st)
        elif name == ">=":
            prim_ge(st)
        elif name == "=":
            prim_eq(st)
        elif name == "!=":
            prim_ne(st)
        elif name == "and":
            prim_and(st)
        elif name == "or":
            prim_or(st)
        elif name == "not":
            prim_not(st)
        elif name == "dup":
            prim_dup(st)
        elif name == "2dup":
            prim_2dup(st)
        elif name == "2drop":
            prim_2drop(st)
        elif name == "2over":
            prim_2over(st)
        elif name == "swap":
            prim_swap(st)
        elif name == "nip":
            prim_nip(st)
        elif name == "pop" or name == "drop":
            prim_pop(st)
        elif name == "over":
            prim_over(st)
        elif name == "rot":
            prim_rot(st)
        elif name == "rolldown" or name == "-rot":
            prim_rolldown(st)
        elif name == "rollup":
            prim_rollup(st)
        elif name == "pick":
            prim_pick(st)
        elif name == "3dup":
            prim_3dup(st)
        elif name == "dupd":
            prim_dupd(st)
        elif name == "swapd":
            prim_swapd(st)
        elif name == "stack":
            prim_stack(st)
        elif name == "unstack":
            prim_unstack(st)
        elif name == "cons":
            prim_cons(st)
        elif name == "uncons":
            prim_uncons(st)
        elif name == "swons":
            prim_swons(st)
        elif name == "first":
            prim_first(st)
        elif name == "rest":
            prim_rest(st)
        elif name == "concat":
            prim_concat(st)
        elif name == "size" or name == "length":
            prim_size(st)
        elif name == "null" or name == "empty?":
            prim_null(st)
        elif name == "small":
            prim_small(st)
        elif name == "reverse":
            prim_reverse(st)
        elif name == "nil":
            prim_nil(st)
        elif name == "succ" or name == "1+":
            prim_succ(st)
        elif name == "pred" or name == "1-":
            prim_pred(st)
        elif name == ".":
            prim_dot(st)
        elif name == "put":
            prim_put(st)
        elif name == "putchars":
            prim_putchars(st)
        elif name == "intern":
            prim_intern(st)
        elif name == "name":
            prim_name(st)
        elif name == "clock":
            prim_clock(st)
        elif name == "<array>":
            prim_new_array(st)
        elif name == "nth":
            prim_nth(st)
        elif name == "set-nth":
            prim_set_nth(st)
        else:
            return False
        return True

    def _dispatch_combinator(self, name, program, ip):
        if name == "i" or name == "call":
            return self.comb_i(program, ip)
        if name == "x":
            return self.comb_x(program, ip)
        if name == "dip":
            return self.comb_dip(program, ip)
        if name == "ifte" or name == "if":
            return self.comb_ifte(program, ip)
        if name == "branch":
            return self.comb_branch(program, ip)
        if name == "times":
            return self.comb_times(program, ip)
        if name == "step" or name == "each":
            return self.comb_step(program, ip)
        if name == "map":
            return self.comb_map(program, ip)
        if name == "fold":
            return self.comb_fold(program, ip)
        if name == "reduce":
            return self.comb_reduce(program, ip)
        if name == "filter":
            return self.comb_filter(program, ip)
        if name == "while":
            return self.comb_while(program, ip)
        if name == "linrec":
            return self.comb_linrec(program, ip)
        if name == "binrec":
            return self.comb_binrec(program, ip)
        if name == "primrec":
            return self.comb_primrec(program, ip)
        raise FactorError("unknown combinator: %s" % name)

    def prim_body(self):
        st = self.st()
        sym = _pop_symbol(st)
        name = sym.name
        prog = self.lookup(name)
        st.push(W_Quotation(prog))

    def _pop_quot(self):
        v = self.st().pop()
        if not isinstance(v, W_Quotation):
            raise FactorError("expected quotation")
        return v.program

    def _snapshot_stack(self):
        return self.st().snapshot_cache()

    def _restore_stack(self, snap):
        self.st().restore_cache(snap)

    def comb_i(self, program, ip):
        return self._enter_ret(self._pop_quot(), program, ip)

    def comb_x(self, program, ip):
        st = self.st()
        v = st.pop()
        if not isinstance(v, W_Quotation):
            raise FactorError("expected quotation")
        st.push(v)
        return self._enter(v.program, program, ip, KIND_RET)

    def comb_dip(self, program, ip):
        st = self.st()
        body = self._pop_quot()
        saved = st.pop()
        ptr = self._push_frame(KIND_DIP, program, ip)
        self.cs_val[ptr] = saved
        st.push_fragment()
        return body, 0

    def comb_ifte(self, program, ip):
        st = self.st()
        else_q = self._pop_quot()
        then_q = self._pop_quot()
        # Avoid peek()-boxing ints: only inspect object tops for quotation form.
        if not st.top_is_int():
            top = st.peek(0)
            if isinstance(top, W_Quotation):
                if_q = self._pop_quot()
                snap = self._snapshot_stack()
                ptr = self._push_frame(KIND_IFTE_COND, program, ip)
                self.cs_body[ptr] = then_q
                self.cs_body2[ptr] = else_q
                self.cs_snap[ptr] = snap
                st.push_fragment()
                return if_q, 0
        flag = st.pop_truthy()
        if flag:
            return self._enter_ret(then_q, program, ip)
        return self._enter_ret(else_q, program, ip)

    def comb_branch(self, program, ip):
        st = self.st()
        flag = st.pop_truthy()
        false_q = self._pop_quot()
        true_q = self._pop_quot()
        if flag:
            return self._enter_ret(true_q, program, ip)
        return self._enter_ret(false_q, program, ip)

    def comb_times(self, program, ip):
        st = self.st()
        body = self._pop_quot()
        count = st.pop_int()
        if count < 0:
            raise FactorError("times expects non-negative integer")
        if count == 0:
            return program, ip
        ptr = self._push_frame(KIND_TIMES, program, ip)
        self.cs_body[ptr] = body
        self.cs_n[ptr] = count
        self.cs_i[ptr] = 0
        st.push_fragment()
        return body, 0

    def comb_step(self, program, ip):
        st = self.st()
        body = self._pop_quot()
        lst = st.pop()
        if not isinstance(lst, W_List):
            raise FactorError("step expects list")
        if not isinstance(lst, W_Cons):
            return program, ip
        ptr = self._push_frame(KIND_STEP, program, ip)
        self.cs_body[ptr] = body
        self.cs_list[ptr] = lst
        st.push(lst.head)
        st.push_fragment()
        return body, 0

    def _step_continue(self, ptr):
        cur = self.cs_list[ptr]
        assert isinstance(cur, W_Cons)
        nxt = cur.tail
        if isinstance(nxt, W_Cons):
            self.cs_list[ptr] = nxt
            st = self.st()
            st.push(nxt.head)
            st.push_fragment()
            return self.cs_body[ptr], 0
        prog = self.cs_progs[ptr]
        ip = self.cs_ips[ptr]
        self.cs_ptr = ptr
        return self._resume_ret(prog, ip)

    def comb_map(self, program, ip):
        st = self.st()
        body = self._pop_quot()
        lst = st.pop()
        if not isinstance(lst, W_List):
            raise FactorError("map expects list")
        if not isinstance(lst, W_Cons):
            st.push(nil_list())
            return program, ip
        ptr = self._push_frame(KIND_MAP, program, ip)
        self.cs_body[ptr] = body
        self.cs_list[ptr] = lst
        self.cs_out[ptr] = []
        st.push(lst.head)
        st.push_fragment()
        return body, 0

    def _map_continue(self, ptr):
        st = self.st()
        out = self.cs_out[ptr]
        out.append(st.pop())
        cur = self.cs_list[ptr]
        assert isinstance(cur, W_Cons)
        nxt = cur.tail
        if isinstance(nxt, W_Cons):
            self.cs_list[ptr] = nxt
            self.cs_out[ptr] = out
            st.push(nxt.head)
            st.push_fragment()
            return self.cs_body[ptr], 0
        st.push(w_list_from_items(out))
        prog = self.cs_progs[ptr]
        ip = self.cs_ips[ptr]
        self.cs_ptr = ptr
        return self._resume_ret(prog, ip)

    def comb_fold(self, program, ip):
        # fold argument order: acc [quot] list fold
        st = self.st()
        lst = st.pop()
        body = self._pop_quot()
        acc = st.pop()
        return self._fold_start(lst, body, acc, program, ip)

    def comb_reduce(self, program, ip):
        # Factor order: seq identity [quot] reduce
        st = self.st()
        body = self._pop_quot()
        acc = st.pop()
        lst = st.pop()
        return self._fold_start(lst, body, acc, program, ip)

    def _fold_start(self, lst, body, acc, program, ip):
        st = self.st()
        if not isinstance(lst, W_List):
            raise FactorError("reduce expects list")
        if not isinstance(lst, W_Cons):
            st.push(acc)
            return program, ip
        ptr = self._push_frame(KIND_FOLD, program, ip)
        self.cs_body[ptr] = body
        self.cs_list[ptr] = lst
        st.push(acc)
        st.push(lst.head)
        st.push_fragment()
        return body, 0

    def _fold_continue(self, ptr):
        st = self.st()
        acc = st.pop()
        cur = self.cs_list[ptr]
        assert isinstance(cur, W_Cons)
        nxt = cur.tail
        if isinstance(nxt, W_Cons):
            self.cs_list[ptr] = nxt
            st.push(acc)
            st.push(nxt.head)
            st.push_fragment()
            return self.cs_body[ptr], 0
        st.push(acc)
        prog = self.cs_progs[ptr]
        ip = self.cs_ips[ptr]
        self.cs_ptr = ptr
        return self._resume_ret(prog, ip)

    def comb_filter(self, program, ip):
        st = self.st()
        body = self._pop_quot()
        lst = st.pop()
        if not isinstance(lst, W_List):
            raise FactorError("filter expects list")
        if not isinstance(lst, W_Cons):
            st.push(nil_list())
            return program, ip
        ptr = self._push_frame(KIND_FILTER, program, ip)
        self.cs_body[ptr] = body
        self.cs_list[ptr] = lst
        self.cs_out[ptr] = []
        self.cs_val[ptr] = lst.head
        st.push(lst.head)
        st.push_fragment()
        return body, 0

    def _filter_continue(self, ptr):
        st = self.st()
        if st.pop_truthy():
            out = self.cs_out[ptr]
            out.append(self.cs_val[ptr])
            self.cs_out[ptr] = out
        cur = self.cs_list[ptr]
        assert isinstance(cur, W_Cons)
        nxt = cur.tail
        if isinstance(nxt, W_Cons):
            self.cs_list[ptr] = nxt
            elem = nxt.head
            self.cs_val[ptr] = elem
            st.push(elem)
            st.push_fragment()
            return self.cs_body[ptr], 0
        st.push(w_list_from_items(self.cs_out[ptr]))
        prog = self.cs_progs[ptr]
        ip = self.cs_ips[ptr]
        self.cs_ptr = ptr
        return self._resume_ret(prog, ip)

    def comb_while(self, program, ip):
        # Factor: ( ..a pred:( ..a -- ..a ? ) body:( ..a -- ..a ) -- ..a )
        # Pred leaves a boolean on top; we pop it and keep the rest.
        body = self._pop_quot()
        cond = self._pop_quot()
        ptr = self._push_frame(KIND_WHILE, program, ip)
        self.cs_body[ptr] = body
        self.cs_body2[ptr] = cond
        self.cs_n[ptr] = 0  # 0=cond done, 1=body done
        self.st().push_fragment()
        return cond, 0

    def _while_continue(self, ptr):
        phase = self.cs_n[ptr]
        body = self.cs_body[ptr]
        cond = self.cs_body2[ptr]
        if phase == 0:
            flag = self.st().pop_truthy()
            if not flag:
                prog = self.cs_progs[ptr]
                ip = self.cs_ips[ptr]
                self.cs_ptr = ptr
                return self._resume_ret(prog, ip)
            self.cs_n[ptr] = 1
            self.st().push_fragment()
            return body, 0
        self.cs_n[ptr] = 0
        self.st().push_fragment()
        return cond, 0

    def comb_linrec(self, program, ip):
        else2 = self._pop_quot()
        else1 = self._pop_quot()
        then_q = self._pop_quot()
        if_q = self._pop_quot()
        return self._linrec_begin(if_q, then_q, else1, else2, program, ip)

    def _linrec_begin(self, if_q, then_q, else1, else2, program, ip):
        snap = self._snapshot_stack()
        ptr = self._push_frame(KIND_LINREC, program, ip)
        self.cs_body[ptr] = if_q
        self.cs_body2[ptr] = then_q
        self.cs_body3[ptr] = else1
        self.cs_body4[ptr] = else2
        self.cs_snap[ptr] = snap
        self.cs_n[ptr] = 0
        self.st().push_fragment()
        return if_q, 0

    def _linrec_continue(self, ptr):
        phase = self.cs_n[ptr]
        if_q = self.cs_body[ptr]
        then_q = self.cs_body2[ptr]
        else1 = self.cs_body3[ptr]
        else2 = self.cs_body4[ptr]
        ret_prog = self.cs_progs[ptr]
        ret_ip = self.cs_ips[ptr]

        if phase == 0:
            flag = self.st().pop_truthy()
            self._restore_stack(self.cs_snap[ptr])
            if flag:
                self.cs_ptr = ptr
                return self._enter(then_q, ret_prog, ret_ip, KIND_RET)
            self.cs_n[ptr] = 1
            self.st().push_fragment()
            return else1, 0

        if phase == 1:
            # else1 done; recurse then else2
            self.cs_n[ptr] = 2
            resume = IP_RESUME_LINREC - ptr
            return self._linrec_begin(
                if_q, then_q, else1, else2, None, resume)

        if phase == 2:
            self.cs_ptr = ptr
            return self._enter(else2, ret_prog, ret_ip, KIND_RET)

        raise FactorError("bad linrec phase")

    def _linrec_after_child(self, ptr):
        self.cs_n[ptr] = 2
        return self._linrec_continue(ptr)

    def comb_binrec(self, program, ip):
        else2 = self._pop_quot()
        else1 = self._pop_quot()
        then_q = self._pop_quot()
        if_q = self._pop_quot()
        return self._binrec_begin(if_q, then_q, else1, else2, program, ip)

    def _binrec_begin(self, if_q, then_q, else1, else2, program, ip):
        snap = self._snapshot_stack()
        ptr = self._push_frame(KIND_BINREC, program, ip)
        self.cs_body[ptr] = if_q
        self.cs_body2[ptr] = then_q
        self.cs_body3[ptr] = else1
        self.cs_body4[ptr] = else2
        self.cs_snap[ptr] = snap
        self.cs_n[ptr] = 0
        self.st().push_fragment()
        return if_q, 0

    def _binrec_continue(self, ptr):
        phase = self.cs_n[ptr]
        if_q = self.cs_body[ptr]
        then_q = self.cs_body2[ptr]
        else1 = self.cs_body3[ptr]
        else2 = self.cs_body4[ptr]
        ret_prog = self.cs_progs[ptr]
        ret_ip = self.cs_ips[ptr]

        if phase == 0:
            flag = self.st().pop_truthy()
            self._restore_stack(self.cs_snap[ptr])
            if flag:
                self.cs_ptr = ptr
                return self._enter(then_q, ret_prog, ret_ip, KIND_RET)
            self.cs_n[ptr] = 1
            self.st().push_fragment()
            return else1, 0

        if phase == 1:
            st = self.st()
            left = st.pop()
            right = st.pop()
            self.cs_val[ptr] = left
            self.cs_val2[ptr] = right
            self.cs_n[ptr] = 2
            st.push(left)
            resume = IP_RESUME_BINREC - ptr
            return self._binrec_begin(
                if_q, then_q, else1, else2, None, resume)

        if phase == 2:
            st = self.st()
            res1 = st.pop()
            self.cs_val[ptr] = res1
            self.cs_n[ptr] = 3
            st.push(self.cs_val2[ptr])
            resume = IP_RESUME_BINREC - ptr
            return self._binrec_begin(
                if_q, then_q, else1, else2, None, resume)

        if phase == 3:
            st = self.st()
            res2 = st.pop()
            st.push(self.cs_val[ptr])
            st.push(res2)
            self.cs_ptr = ptr
            return self._enter(else2, ret_prog, ret_ip, KIND_RET)

        raise FactorError("bad binrec phase")

    def _binrec_after_child(self, ptr):
        return self._binrec_continue(ptr)

    def comb_primrec(self, program, ip):
        else2 = self._pop_quot()
        then_q = self._pop_quot()
        return self._primrec_begin(then_q, else2, program, ip)

    def _primrec_begin(self, then_q, else2, program, ip):
        st = self.st()
        n = st.pop_int()
        if n == 0:
            return self._enter(then_q, program, ip, KIND_RET)
        st.push_int(n)
        st.push_int(n - 1)
        ptr = self._push_frame(KIND_PRIMREC, program, ip)
        self.cs_body[ptr] = then_q
        self.cs_body2[ptr] = else2
        resume = IP_RESUME_PRIMREC - ptr
        return self._primrec_begin(then_q, else2, None, resume)

    def _primrec_after_child(self, ptr):
        else2 = self.cs_body2[ptr]
        ret_prog = self.cs_progs[ptr]
        ret_ip = self.cs_ips[ptr]
        self.cs_ptr = ptr
        return self._enter(else2, ret_prog, ret_ip, KIND_RET)

    def pop_int_result(self):
        return self.st().pop_int()

    def peek_int_result(self):
        return self.st().peek_int(0)
