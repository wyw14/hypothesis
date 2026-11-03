# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from collections import OrderedDict
from itertools import pairwise

import pytest

from hypothesis import given, settings
from hypothesis.errors import Unsatisfiable
from hypothesis.strategies import (
    booleans,
    dictionaries,
    fixed_dictionaries,
    frozensets,
    integers,
    just,
    lists,
    none,
    nothing,
    sets,
    text,
    tuples,
)

from tests.common.debug import assert_all_examples, find_any, minimal
from tests.common.utils import flaky


def longest_equal_run(xs):
    # Length of the longest run of adjacent elements which compare equal.
    # Uses only ==, so it works for unhashable elements.
    longest = run = 0
    for i, x in enumerate(xs):
        run = run + 1 if i > 0 and xs[i - 1] == x else 1
        longest = max(longest, run)
    return longest


def has_no_adjacent_equals(xs):
    return all(a != b for a, b in pairwise(xs))


@pytest.mark.parametrize(
    ("col", "strat"),
    [
        ((), tuples()),
        ([], lists(none(), max_size=0)),
        (set(), sets(none(), max_size=0)),
        (frozenset(), frozensets(none(), max_size=0)),
        ({}, fixed_dictionaries({})),
        ({}, fixed_dictionaries({}, optional={})),
        (OrderedDict(), fixed_dictionaries(OrderedDict(), optional=OrderedDict())),
        ({}, fixed_dictionaries({}, optional={1: booleans()})),
        ({0: False}, fixed_dictionaries({0: booleans()}, optional={1: booleans()})),
        ({}, fixed_dictionaries({}, optional={(): booleans(), 0: booleans()})),
        ([], lists(nothing())),
        ([], lists(nothing(), unique=True)),
    ],
)
def test_find_empty_collection_gives_empty(col, strat):
    assert minimal(strat) == col


@pytest.mark.parametrize(
    ("coltype", "strat"), [(list, lists), (set, sets), (frozenset, frozensets)]
)
def test_find_non_empty_collection_gives_single_zero(coltype, strat):
    assert minimal(strat(integers()), bool) == coltype((0,))


@pytest.mark.parametrize(
    ("coltype", "strat"), [(list, lists), (set, sets), (frozenset, frozensets)]
)
def test_minimizes_to_empty(coltype, strat):
    assert minimal(strat(integers())) == coltype()


def test_minimizes_list_of_lists():
    xs = minimal(lists(lists(booleans())), lambda x: any(x) and not all(x))
    xs.sort()
    assert xs == [[], [False]]


@given(sets(integers(0, 100), min_size=2, max_size=10))
@settings(max_examples=100)
def test_sets_are_size_bounded(xs):
    assert 2 <= len(xs) <= 10


def test_fixed_dictionaries_vary_key_order():
    keys = list(range(10))
    find_any(
        fixed_dictionaries({k: booleans() for k in keys}),
        lambda x: list(x.keys()) != keys,
    )


@given(fixed_dictionaries({}, optional={0: booleans(), 1: nothing(), 2: booleans()}))
def test_fixed_dictionaries_with_optional_and_empty_keys(d):
    assert 1 not in d


@pytest.mark.parametrize("n", range(10))
def test_lists_of_fixed_length(n):
    assert minimal(lists(integers(), min_size=n, max_size=n)) == [0] * n


@pytest.mark.parametrize("n", range(10))
def test_sets_of_fixed_length(n):
    x = minimal(sets(integers(), min_size=n, max_size=n))
    assert len(x) == n

    if n == 0:
        assert x == set()
    else:
        assert x == set(range(min(x), min(x) + n))


@pytest.mark.parametrize("n", range(10))
def test_dictionaries_of_fixed_length(n):
    x = set(minimal(dictionaries(integers(), booleans(), min_size=n, max_size=n)))

    if not n:
        assert x == set()
    else:
        assert x == set(range(min(x), min(x) + n))


@pytest.mark.parametrize("n", range(10))
def test_lists_of_lower_bounded_length(n):
    l = minimal(lists(integers(), min_size=n), lambda x: sum(x) >= 2 * n)
    assert l == [] if n == 0 else [0] * (n - 1) + [n * 2]


@flaky(min_passes=1, max_runs=3)
def test_can_find_unique_lists_of_non_set_order():
    # This test checks that our strategy for unique lists doesn't accidentally
    # depend on the iteration order of sets.
    #
    # Unfortunately, that means that *this* test has to rely on set iteration
    # order. That makes it tricky to debug on CPython, because set iteration
    # order changes every time the process is launched.
    #
    # To get around this, define the PYTHONHASHSEED environment variable to
    # a consistent value. This could be 0, or it could be the PYTHONHASHSEED
    # value listed in a failure log from CI.

    ls = minimal(
        lists(text(), min_size=2, unique=True),
        lambda x: list(set(reversed(x))) != x,  # noqa: C414  # yes, reverse inside set
    )
    assert len(set(ls)) == len(ls)
    assert len(ls) == 2


def test_can_draw_empty_list_from_unsatisfiable_strategy():
    assert find_any(lists(integers().filter(lambda s: False))) == []


def test_can_draw_empty_set_from_unsatisfiable_strategy():
    assert find_any(sets(integers().filter(lambda s: False))) == set()


@given(lists(sets(none()), min_size=10))
def test_small_sized_sets(x):
    pass


def test_minimize_dicts_with_incompatible_keys():
    strat = fixed_dictionaries({1: booleans(), "hi": lists(booleans())})
    assert minimal(strat) == {1: False, "hi": []}


@given(
    lists(
        tuples(integers(), integers()),
        min_size=2,
        unique_by=(lambda x: x[0], lambda x: x[1]),
    )
)
def test_lists_unique_by_tuple_funcs(ls):
    firstitems, seconditems = zip(*ls, strict=True)
    assert len(set(firstitems)) == len(firstitems)
    assert len(set(seconditems)) == len(seconditems)


@pytest.mark.parametrize("max_run", [1, 2, 3, 5])
def test_lists_respect_max_run_and_sizes(max_run):
    strat = lists(integers(0, 2), min_size=2, max_size=12, max_run=max_run)
    assert_all_examples(
        strat,
        lambda xs: 2 <= len(xs) <= 12 and longest_equal_run(xs) <= max_run,
    )


def test_max_run_none_matches_default_behaviour():
    assert minimal(lists(integers(), max_run=None)) == []
    strat = lists(integers(0, 2), min_size=3, max_size=3, max_run=None)
    assert minimal(strat) == [0, 0, 0]


def test_max_run_with_max_size_zero():
    assert minimal(lists(just(0), max_size=0, max_run=1)) == []


@pytest.mark.parametrize("n", [1, 2, 3])
def test_max_run_feasible_exactly_at_boundary(n):
    # A list of n identical values is drawable precisely up to the boundary.
    assert minimal(lists(just(0), min_size=n, max_size=n, max_run=n)) == [0] * n


@pytest.mark.parametrize(("min_size", "max_run"), [(2, 1), (3, 2), (4, 1), (6, 2)])
def test_max_run_unsatisfiable_for_an_unvarying_element_strategy(min_size, max_run):
    # If the only drawable value can never differ from its neighbours,
    # Hypothesis reports the strategy as unsatisfiable rather than emitting
    # a list which violates the run limit.
    with pytest.raises(Unsatisfiable):
        find_any(lists(just(0), min_size=min_size, max_run=max_run))


def test_max_run_one_alternates_and_repeats_non_adjacently():
    strat = lists(integers(0, 1), min_size=6, max_size=6, max_run=1)
    assert_all_examples(
        strat,
        lambda xs: len(xs) == 6 and has_no_adjacent_equals(xs),
    )
    # The same values recur, in multiple runs - they are just never adjacent.
    xs = find_any(strat, lambda xs: xs.count(0) >= 2 and xs.count(1) >= 2)
    assert xs.count(0) == xs.count(1) == 3
    assert longest_equal_run(xs) == 1


def test_max_run_one_shrinks_to_an_alternating_list():
    xs = minimal(lists(integers(0, 1), min_size=6, max_size=6, max_run=1))
    assert len(xs) == 6
    assert has_no_adjacent_equals(xs)
    # Three of each value is the minimum possible for six alternating values.
    assert xs.count(0) == xs.count(1) == 3


def test_max_run_two_allows_separate_runs_and_shrinks():
    strat = lists(integers(0, 1), min_size=5, max_size=5, max_run=2)
    # Four zeroes can occur, but they must be split into separate runs.
    assert_all_examples(
        strat,
        lambda xs: longest_equal_run(xs) <= 2
        and (xs.count(0) < 4 or longest_equal_run(xs) == 2),
    )
    xs = minimal(strat)
    assert xs == [0, 0, 1, 0, 0]
    assert longest_equal_run(xs) == 2


def test_max_run_with_unhashable_elements():
    inner = lists(integers(0, 1), min_size=1, max_size=2)
    strat = lists(inner, min_size=6, max_size=10, max_run=1)
    assert_all_examples(
        strat,
        lambda xs: 6 <= len(xs) <= 10 and has_no_adjacent_equals(xs),
    )

    # The limit is also maintained while shrinking, and equal values may
    # recur non-adjacently even though they are unhashable.
    xs = minimal(lists(inner, min_size=4, max_size=4, max_run=1))
    assert len(xs) == 4
    assert has_no_adjacent_equals(xs)
    assert xs[0] == xs[2] == [0]
    assert xs[1] == xs[3]
    assert all(v == [0] or v == [0, 0] for v in xs)


def test_max_run_combined_with_unique():
    strat = lists(integers(0, 7), min_size=4, max_size=8, unique=True, max_run=1)

    def valid(xs):
        return (
            4 <= len(xs) <= 8 and len(xs) == len(set(xs)) and has_no_adjacent_equals(xs)
        )

    assert_all_examples(strat, valid)


def test_max_run_combined_with_unique_by():
    strat = lists(
        tuples(integers(0, 4), integers()),
        min_size=3,
        unique_by=lambda x: x[0],
        max_run=2,
    )

    def valid(xs):
        keys = [key for key, _ in xs]
        return len(keys) == len(set(keys)) and longest_equal_run(xs) <= 2

    assert_all_examples(strat, valid)


def test_max_run_is_preserved_when_shrinking_a_failing_example():
    # Mimics a test which fails on lists containing a "large" value, so that
    # the reported (shrunk) failure must still obey the run limit.
    xs = minimal(
        lists(integers(0, 4), max_run=1),
        lambda x: len(x) >= 5 and any(v >= 2 for v in x),
    )
    assert len(xs) == 5
    assert max(xs) == 2
    assert 2 in xs
    assert has_no_adjacent_equals(xs)
