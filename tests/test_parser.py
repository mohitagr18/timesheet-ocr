"""Tests for the parser module — date, time, and hours parsing."""

from datetime import date, time

import pytest

from parser import clean_name, disambiguate_times, parse_date, parse_hours, parse_time


class TestParseDate:
    def test_mm_dd_yyyy(self):
        assert parse_date("03/15/2026") == date(2026, 3, 15)

    def test_mm_dd_yy(self):
        assert parse_date("03/15/26") == date(2026, 3, 15)

    def test_single_digit_month_day(self):
        assert parse_date("3/5/2026") == date(2026, 3, 5)

    def test_dashes(self):
        assert parse_date("03-15-2026") == date(2026, 3, 15)

    def test_dots(self):
        assert parse_date("03.15.2026") == date(2026, 3, 15)

    def test_invalid_date(self):
        assert parse_date("13/32/2026") is None

    def test_empty_string(self):
        assert parse_date("") is None

    def test_garbage(self):
        assert parse_date("hello") is None


class TestParseTime:
    def test_24h_format(self):
        assert parse_time("09:00") == time(9, 0)
        assert parse_time("17:30") == time(17, 30)

    def test_12h_format_am(self):
        assert parse_time("9:00 AM") == time(9, 0)

    def test_12h_format_pm(self):
        assert parse_time("5:30 PM") == time(17, 30)

    def test_12_pm(self):
        assert parse_time("12:00 PM") == time(12, 0)

    def test_12_am(self):
        assert parse_time("12:00 AM") == time(0, 0)

    def test_ocr_artifacts_O_for_0(self):
        # OCR might read "0" as "O"
        assert parse_time("9:OO") == time(9, 0)

    def test_no_separator(self):
        assert parse_time("0900") == time(9, 0)

    def test_empty(self):
        assert parse_time("") is None

    def test_invalid(self):
        assert parse_time("25:00") is None


class TestParseHours:
    def test_decimal(self):
        assert parse_hours("3.5") == 3.5

    def test_integer(self):
        assert parse_hours("8") == 8.0

    def test_half_fraction(self):
        assert parse_hours("3 1/2") == 3.5

    def test_quarter_fraction(self):
        assert parse_hours("7 1/4") == 7.25

    def test_three_quarter_fraction(self):
        assert parse_hours("7 3/4") == 7.75

    def test_empty(self):
        assert parse_hours("") is None


class TestCleanName:
    def test_basic(self):
        assert clean_name("  Jane Smith  ") == "Jane Smith"

    def test_remove_prefix(self):
        assert clean_name("Employee: Jane Smith") == "Jane Smith"
        assert clean_name("Name: Jane Smith") == "Jane Smith"

    def test_normalize_spaces(self):
        assert clean_name("Jane   Smith") == "Jane Smith"


class TestDisambiguateTimes:
    def test_bare_times_with_total_hours(self):
        t_in, t_out = disambiguate_times("8:00", "4:30", "8.5")
        assert t_in == time(8, 0)
        assert t_out == time(16, 30)

    def test_bare_times_integer_hours(self):
        t_in, t_out = disambiguate_times("7:00", "3:00", "8")
        assert t_in == time(7, 0)
        assert t_out == time(15, 0)

    def test_overnight_shift(self):
        # "11:00"/"7:00" with 8h: both AM→PM (11-19) and PM→AM (23-07) give 8h.
        # The algorithm picks AM→PM as it comes first in the loop.
        t_in, t_out = disambiguate_times("11:00", "7:00", "8")
        assert t_in == time(11, 0)
        assert t_out == time(19, 0)

    def test_already_has_am_pm(self):
        t_in, t_out = disambiguate_times("8:00 AM", "4:30 PM", "8.5")
        assert t_in == time(8, 0)
        assert t_out == time(16, 30)

    def test_missing_total_hours(self):
        t_in, t_out = disambiguate_times("8:00", "4:30", "")
        assert t_in == time(8, 0)
        assert t_out == time(4, 30)

    def test_empty_time_in(self):
        # Cannot disambiguate time_out without time_in
        t_in, t_out = disambiguate_times("", "4:30", "8")
        assert t_in is None
        assert t_out == time(4, 30)

    def test_empty_time_out(self):
        t_in, t_out = disambiguate_times("8:00", "", "8")
        assert t_in == time(8, 0)
        assert t_out is None

    def test_both_empty(self):
        t_in, t_out = disambiguate_times("", "", "")
        assert t_in is None
        assert t_out is None

    def test_half_hour_shift(self):
        t_in, t_out = disambiguate_times("9:00", "9:30", "0.5")
        assert t_in == time(9, 0)
        assert t_out == time(9, 30)

    def test_pm_am_disambiguation(self):
        # "1:00"/"9:00" with 8h: AM/AM (1-9) and PM/PM (13-21) both give 8h.
        # The algorithm picks AM/AM as it comes first.
        t_in, t_out = disambiguate_times("1:00", "9:00", "8")
        assert t_in == time(1, 0)
        assert t_out == time(9, 0)
