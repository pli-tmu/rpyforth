import os

from rpyforth.heap import DICT_SIZE_BYTES, _alloc_region_bytes, _default_alloc_mb
from rpyforth.outer_interp import OuterInterpreter
from rpyforth.inner_interp import InnerInterpreter


def with_env(value, fn):
    old = os.environ.get("RPYFORTH_ALLOC_MB")
    try:
        if value is None:
            os.environ.pop("RPYFORTH_ALLOC_MB", None)
        else:
            os.environ["RPYFORTH_ALLOC_MB"] = value
        return fn()
    finally:
        if old is None:
            os.environ.pop("RPYFORTH_ALLOC_MB", None)
        else:
            os.environ["RPYFORTH_ALLOC_MB"] = old


def test_env_override_wins():
    assert with_env("64", _alloc_region_bytes) == 64 << 20


def test_untranslated_default_is_small():
    # Untranslated the raw buffer is a simulated array with O(size) creation
    # cost, so the suite default must stay at 1 MB.
    assert with_env(None, _alloc_region_bytes) == 1 << 20


def test_translated_default_is_generous():
    # brainless's ttable.fs does `#ttentries /ttentry * 2* ALLOCATE THROW` at
    # load (several MB). The translated binary must not need an env var for
    # stock appbench programs; calloc'd pages are lazily mapped so a generous
    # default costs nothing.
    assert _default_alloc_mb(translated=True) >= 64
    assert _default_alloc_mb(translated=False) == 1


def test_unused_reports_dictionary_space():
    inner = InnerInterpreter()
    outer = OuterInterpreter(inner)
    outer.interpret_line("UNUSED")
    unused = inner.pop_ds_int()
    assert 0 < unused <= DICT_SIZE_BYTES
