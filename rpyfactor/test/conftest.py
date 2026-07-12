"""Test helpers for rpyfactor."""

from rpyfactor.interp import Interpreter


def run(source):
    interp = Interpreter()
    interp.run_source(source)
    return interp


def run_result_int(source):
    return run(source).pop_int_result()
