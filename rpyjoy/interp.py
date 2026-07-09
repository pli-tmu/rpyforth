"""Main interpreter loop for rpyjoy."""

from rpython.rlib.jit import JitDriver, promote

from rpyjoy.stackrep import make_stack, USE_STACK_FRAGMENT
from rpyjoy.program import CallWord, item_to_value, is_literal_item
from rpyjoy.values import (
    W_Int, W_List, W_Quotation,
    JoyError, truthy,
)
from rpyjoy.primitives import (
    make_prim_table, is_combinator, _pop_symbol,
)


def _jit_location(ip, program):
    if ip < len(program):
        item = program[ip]
        if isinstance(item, CallWord):
            return "ip=%d %s" % (ip, item.name)
    return "ip=%d" % ip


if USE_STACK_FRAGMENT:
    jitdriver = JitDriver(
        greens=['ip', 'program'],
        reds=['self'],
        get_printable_location=_jit_location,
    )
else:
    jitdriver = JitDriver(
        greens=['ip', 'program'],
        reds=['self'],
        get_printable_location=_jit_location,
    )


class Interpreter(object):
    def __init__(self):
        self.stack = make_stack()
        self.dict = {}
        self.prims = make_prim_table(self)

    def load_definitions(self, defs):
        for name, body in defs.items():
            self.dict[name] = body

    def lookup(self, name):
        if name in self.dict:
            return self.dict[name]
        if name in self.prims:
            return None
        raise JoyError("unknown word: %s" % name)

    def run(self, program):
        self._execute(program)

    def run_source(self, text):
        from rpyjoy.lexer import parse_source
        defs, program = parse_source(text)
        self.load_definitions(defs)
        self.run(program)

    def run_quot(self, program):
        self._run_fragment(program)

    def _run_fragment(self, program):
        self.stack.push_fragment()
        self._execute(program)
        self.stack.pop_fragment_commit()

    def _execute(self, program):
        ip = 0
        while True:
            jitdriver.jit_merge_point(ip=ip, program=program, self=self)
            if ip >= len(program):
                break
            item = program[ip]
            ip += 1
            if is_literal_item(item):
                self.stack.push(item_to_value(item))
            elif isinstance(item, CallWord):
                self._call_word(item.name)
            else:
                raise JoyError("bad program item")

    def _call_word(self, name):
        if name == "body":
            self.prim_body(self.stack)
            return
        if is_combinator(name):
            self._dispatch_combinator(name)
            return
        if name in self.prims:
            self.prims[name](self.stack)
            return
        body = self.lookup(name)
        if body is not None:
            self._run_fragment(promote(body))

    def _dispatch_combinator(self, name):
        if name == "i":
            self.comb_i()
        elif name == "x":
            self.comb_x()
        elif name == "dip":
            self.comb_dip()
        elif name == "ifte":
            self.comb_ifte()
        elif name == "branch":
            self.comb_branch()
        elif name == "times":
            self.comb_times()
        elif name == "step":
            self.comb_step()
        elif name == "map":
            self.comb_map()
        elif name == "fold":
            self.comb_fold()
        elif name == "filter":
            self.comb_filter()
        elif name == "while":
            self.comb_while()
        elif name == "linrec":
            self.comb_linrec()
        elif name == "binrec":
            self.comb_binrec()
        elif name == "primrec":
            self.comb_primrec()
        else:
            raise JoyError("unknown combinator: %s" % name)

    def prim_body(self, st):
        sym = _pop_symbol(st)
        name = sym.name
        prog = self.lookup(name)
        st.push(W_Quotation(prog))

    def _pop_quot(self):
        v = self.stack.pop()
        if not isinstance(v, W_Quotation):
            raise JoyError("expected quotation")
        return v.program

    def _snapshot_stack(self):
        return self.stack.snapshot_flat()

    def _restore_stack(self, snap):
        self.stack.restore_flat(snap)

    def comb_i(self):
        self._run_fragment(self._pop_quot())

    def comb_x(self):
        v = self.stack.pop()
        if not isinstance(v, W_Quotation):
            raise JoyError("expected quotation")
        self.stack.push(v)
        self._run_fragment(v.program)

    def comb_dip(self):
        prog = self._pop_quot()
        saved = self.stack.pop()
        self._run_fragment(prog)
        self.stack.push(saved)

    def comb_ifte(self):
        else_q = self._pop_quot()
        then_q = self._pop_quot()
        top = self.stack.peek(0)
        if isinstance(top, W_Quotation):
            if_q = self._pop_quot()
            snap = self._snapshot_stack()
            self._run_fragment(if_q)
            flag = truthy(self.stack.pop())
            self._restore_stack(snap)
        else:
            flag = truthy(self.stack.pop())
        if flag:
            self._run_fragment(then_q)
        else:
            self._run_fragment(else_q)

    def comb_branch(self):
        flag = truthy(self.stack.pop())
        false_q = self._pop_quot()
        true_q = self._pop_quot()
        if flag:
            self._run_fragment(true_q)
        else:
            self._run_fragment(false_q)

    def comb_times(self):
        body = self._pop_quot()
        n = self.stack.pop()
        if not isinstance(n, W_Int):
            raise JoyError("times expects integer")
        count = n.val
        if count < 0:
            raise JoyError("times expects non-negative integer")
        i = 0
        while i < count:
            self._run_fragment(body)
            i += 1

    def comb_step(self):
        body = self._pop_quot()
        lst = self.stack.pop()
        if not isinstance(lst, W_List):
            raise JoyError("step expects list")
        items = lst.items
        i = 0
        while i < len(items):
            self.stack.push(items[i])
            self._run_fragment(body)
            i += 1

    def comb_map(self):
        body = self._pop_quot()
        lst = self.stack.pop()
        if not isinstance(lst, W_List):
            raise JoyError("map expects list")
        out = []
        items = lst.items
        i = 0
        while i < len(items):
            self.stack.push(items[i])
            self._run_fragment(body)
            out.append(self.stack.pop())
            i += 1
        self.stack.push(W_List(out))

    def comb_fold(self):
        lst = self.stack.pop()
        body = self._pop_quot()
        acc = self.stack.pop()
        if not isinstance(lst, W_List):
            raise JoyError("fold expects list")
        items = lst.items
        i = 0
        while i < len(items):
            self.stack.push(acc)
            self.stack.push(items[i])
            self._run_fragment(body)
            acc = self.stack.pop()
            i += 1
        self.stack.push(acc)

    def comb_filter(self):
        body = self._pop_quot()
        lst = self.stack.pop()
        if not isinstance(lst, W_List):
            raise JoyError("filter expects list")
        out = []
        items = lst.items
        i = 0
        while i < len(items):
            elem = items[i]
            self.stack.push(elem)
            self._run_fragment(body)
            if truthy(self.stack.pop()):
                out.append(elem)
            i += 1
        self.stack.push(W_List(out))

    def comb_while(self):
        body = self._pop_quot()
        cond = self._pop_quot()
        while True:
            snap = self._snapshot_stack()
            self._run_fragment(cond)
            if not truthy(self.stack.pop()):
                self._restore_stack(snap)
                break
            self._restore_stack(snap)
            self._run_fragment(body)

    def comb_linrec(self):
        else2 = self._pop_quot()
        else1 = self._pop_quot()
        then_q = self._pop_quot()
        if_q = self._pop_quot()
        self._linrec_loop(if_q, then_q, else1, else2)

    def _linrec_loop(self, if_q, then_q, else1, else2):
        snap = self._snapshot_stack()
        self._run_fragment(if_q)
        if truthy(self.stack.pop()):
            self._restore_stack(snap)
            self._run_fragment(then_q)
        else:
            self._restore_stack(snap)
            self._run_fragment(else1)
            self._linrec_loop(if_q, then_q, else1, else2)
            self._run_fragment(else2)

    def comb_binrec(self):
        else2 = self._pop_quot()
        else1 = self._pop_quot()
        then_q = self._pop_quot()
        if_q = self._pop_quot()
        self._binrec_loop(if_q, then_q, else1, else2)

    def _binrec_loop(self, if_q, then_q, else1, else2):
        snap = self._snapshot_stack()
        self._run_fragment(if_q)
        if truthy(self.stack.pop()):
            self._restore_stack(snap)
            self._run_fragment(then_q)
        else:
            self._restore_stack(snap)
            self._run_fragment(else1)
            left = self.stack.pop()
            right = self.stack.pop()
            self.stack.push(left)
            self._binrec_loop(if_q, then_q, else1, else2)
            res1 = self.stack.pop()
            self.stack.push(right)
            self._binrec_loop(if_q, then_q, else1, else2)
            res2 = self.stack.pop()
            self.stack.push(res1)
            self.stack.push(res2)
            self._run_fragment(else2)

    def comb_primrec(self):
        else2 = self._pop_quot()
        then_q = self._pop_quot()
        self._primrec_loop(then_q, else2)

    def _primrec_loop(self, then_q, else2):
        n = self.stack.pop()
        if not isinstance(n, W_Int):
            raise JoyError("primrec expects integer")
        if n.val == 0:
            self._run_fragment(then_q)
        else:
            self.stack.push(n)
            self.stack.push(W_Int(n.val - 1))
            self._primrec_loop(then_q, else2)
            self._run_fragment(else2)

    def pop_int_result(self):
        v = self.stack.pop()
        if not isinstance(v, W_Int):
            raise JoyError("expected integer result")
        return v.val

    def peek_int_result(self):
        v = self.stack.peek(0)
        if not isinstance(v, W_Int):
            raise JoyError("expected integer result")
        return v.val
