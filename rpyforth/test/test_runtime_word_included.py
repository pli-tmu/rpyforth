import os
from rpyforth.outer_interp import OuterInterpreter
from rpyforth.inner_interp import InnerInterpreter


def run_lines(lines):
    inner = InnerInterpreter()
    outer = OuterInterpreter(inner)
    for line in lines:
        outer.interpret_line(line)
    return inner


def _chars(inner, c_addr, u):
    out = []
    for k in range(u):
        out.append(chr(inner.char_fetch(c_addr + k)))
    return "".join(out)


def test_runtime_word_reads_next_token():
    # BL WORD executed from a colon body consumes the token after the call site
    # and returns a counted string; COUNT then yields ( c-addr u ).
    inner = run_lines([
        ": grab BL WORD COUNT ;",
        "grab hello",
    ])
    u = inner.pop_ds_int()
    c_addr = inner.pop_ds_int()
    assert u == 5
    assert _chars(inner, c_addr, u) == "hello"


def test_runtime_included_from_char_memory():
    # A filename assembled in char memory (as brainless does via PAD/MOVE) and
    # passed to INCLUDED at runtime must load the file.
    path = "/tmp/inc_target_rt.fs"
    f = open(path, "w")
    f.write(": got 111 ;\n")
    f.close()
    try:
        inner = run_lines([
            ': loadit S" ' + path + '" INCLUDED ;',
            "loadit",
            "got",
        ])
        assert inner.pop_ds_int() == 111
    finally:
        os.remove(path)


def test_runtime_included_via_word():
    # Mirror brainless load-part: BL WORD builds the counted string, COUNT turns
    # it into ( c-addr u ), INCLUDED loads it -- all at runtime in a colon body.
    path = "/tmp/inc_target_word.fs"
    f = open(path, "w")
    f.write(": viaword 222 ;\n")
    f.close()
    try:
        inner = run_lines([
            ": lp BL WORD COUNT INCLUDED ;",
            "lp " + path,
            "viaword",
        ])
        assert inner.pop_ds_int() == 222
    finally:
        os.remove(path)


def test_second_string_literal_after_dot_quote_compiles_content():
    # A ." and an s" on the same line must each track their own occurrence
    # index; the s" content used to compile empty.
    from rpyforth.outer_interp import OuterInterpreter
    from rpyforth.inner_interp import InnerInterpreter
    inner = InnerInterpreter()
    outer = OuterInterpreter(inner)
    outer.interpret_line(': t ." A" s" xyz" ;')
    outer.interpret_line('t')
    u = inner.pop_ds_int()
    c_addr = inner.pop_ds_int()
    assert u == 3
    s = "".join([chr(inner.char_fetch(c_addr + k)) for k in range(u)])
    assert s == "xyz"
