from rpyforth.outer_interp import OuterInterpreter
from rpyforth.inner_interp import InnerInterpreter
from rpyforth.prelude import load_prelude


def make():
    inner = InnerInterpreter()
    outer = OuterInterpreter(inner)
    load_prelude(outer)
    return inner, outer


def test_2constant_pushes_pair():
    inner, outer = make()
    outer.interpret_line("2 3 2CONSTANT PR  PR")
    assert inner.pop_ds_int() == 3
    assert inner.pop_ds_int() == 2


def test_struct_field_offsets():
    inner, outer = make()
    outer.interpret_line("struct  cell% field .a  cell% field .b  end-struct point%")
    # .a is at offset 0, .b at offset 1 cell
    outer.interpret_line("0 .a")
    assert inner.pop_ds_int() == 0
    outer.interpret_line("0 .b")
    assert inner.pop_ds_int() == 1  # CELL_SIZE_BYTES == 1


def test_struct_size():
    inner, outer = make()
    outer.interpret_line("struct  cell% field .x  cell% field .y  end-struct pt%")
    outer.interpret_line("pt% %size")
    assert inner.pop_ds_int() == 2  # two cells, 1 byte each


def test_percent_allot_reserves_size():
    inner, outer = make()
    outer.interpret_line("struct  cell% field .x  cell% field .y  end-struct pt%")
    outer.interpret_line("HERE  pt% %allot  HERE SWAP -")
    assert inner.pop_ds_int() == 2  # advanced HERE by %size bytes
