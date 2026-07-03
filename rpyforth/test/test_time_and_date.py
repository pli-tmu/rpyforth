import time as pytime

from rpyforth.outer_interp import OuterInterpreter
from rpyforth.inner_interp import InnerInterpreter


def test_time_and_date_matches_utc_clock():
    inner = InnerInterpreter()
    outer = OuterInterpreter(inner)
    before = pytime.gmtime()
    outer.interpret_line("TIME&DATE")
    after = pytime.gmtime()
    year = inner.pop_ds_int()
    month = inner.pop_ds_int()
    day = inner.pop_ds_int()
    hour = inner.pop_ds_int()
    minute = inner.pop_ds_int()
    sec = inner.pop_ds_int()
    assert 0 <= sec <= 60
    assert 0 <= minute <= 59
    assert 0 <= hour <= 23
    assert (year, month, day) in (
        (before.tm_year, before.tm_mon, before.tm_mday),
        (after.tm_year, after.tm_mon, after.tm_mday),
    )
    assert (hour, minute) in (
        (before.tm_hour, before.tm_min),
        (after.tm_hour, after.tm_min),
    )
