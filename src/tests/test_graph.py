import pytest

import collections
import numpy as np

from ..graph import (make_symmetric, build_graph,
                     get_links, traverse, get_subgraphs)


def test_make_symmetric():
    g = collections.defaultdict(list)
    g.update({1: [2, 3], 4: [3], 5: []})
    make_symmetric(g)
    assert g == {1: [2, 3], 4: [3], 2: [1], 3: [1, 4], 5: []}


@pytest.mark.parametrize(
    'indices,method,neighbor_within_swath,'
    'positions,neighbor_across_swath,max_dist,expected',
    [
        pytest.param(
            [[(2, 3), (2, 4)],
             [(2, 9)],
             [(13, 1), (13, 2), (13, 3)]],
            'across',
            None,
            [np.array([[0, 0], [1, 0]]),
             np.array([[0, -5]]),
             np.array([[9.9, 0], [10.9, 0], [11.9, 0]])],
            1,
            10,
            {(2, 3): [(2, 9), (13, 1)],
             (2, 4): [(2, 9), (13, 1), (13, 2)],
             (2, 9): [(2, 3), (2, 4)],
             (13, 1): [(2, 4), (2, 3)],
             (13, 2): [(2, 4)]}
        ),
        pytest.param(
            [[(2, 3), (2, 4)],
             [(2, 9)],
             [(13, 1), (13, 2), (13, 3)]],
            'across',
            None,
            [np.array([[0, 0], [1, 0]]),
             np.array([[0, -5]]),
             np.array([[9.9, 0], [10.9, 0], [11.9, 0]])],
            2,
            10,
            {(2, 3): [(2, 9), (13, 1)],
             (2, 4): [(2, 9), (13, 1), (13, 2)],
             (2, 9): [(2, 3), (2, 4)],
             (13, 1): [(2, 4), (2, 3)],
             (13, 2): [(2, 4)]}
        ),
    ],
)
def test_build_graph(indices, method,
                     neighbor_within_swath,
                     positions, neighbor_across_swath, max_dist, expected):
    assert build_graph(
        indices=indices, method=method,
        neighbor_within_swath=neighbor_within_swath,
        positions=positions, neighbor_across_swath=neighbor_across_swath,
        max_dist=max_dist) == expected


@pytest.mark.parametrize(
    'graph,links,subset,symmetric,expected',
    [
        pytest.param(
            {(1, 2): [(1, 3), (1, 4)],
             (1, 3): [(1, 2)],
             (1, 4): [(1, 2)]},
            {((1, 2), (1, 3)): None,
             ((1, 3), (1, 2)): None,
             ((1, 2), (1, 4)): None,
             ((1, 4), (1, 2)): None,
             ((4, 5), (5, 7)): None},
            None,
            True,
            {((1, 2), (1, 3)): None,
             ((1, 3), (1, 2)): None,
             ((1, 2), (1, 4)): None,
             ((1, 4), (1, 2)): None}
        ),
        pytest.param(
            {(1, 2): [(1, 3), (1, 4)],
             (1, 3): [(1, 2)],
             (1, 4): [(1, 2)]},
            {((1, 2), (1, 3)): None,
             ((1, 3), (1, 2)): None,
             ((1, 2), (1, 4)): None,
             ((1, 4), (1, 2)): None,
             ((4, 5), (5, 7)): None},
            [(1, 2), (1, 3)],
            True,
            {((1, 2), (1, 3)): None,
             ((1, 3), (1, 2)): None}
        ),
        pytest.param(
            {(1, 2): [(1, 3), (1, 4)],
             (1, 3): [(1, 2)],
             (1, 4): [(1, 2)]},
            {((1, 2), (1, 3)): None,
             ((1, 3), (1, 2)): None,
             ((1, 2), (1, 4)): None,
             ((1, 4), (1, 2)): None,
             ((4, 5), (5, 7)): None},
            [(1, 2), (1, 3)],
            False,
            {((1, 2), (1, 3)): None}
        ),
    ],
)
def test_get_links(graph, links, subset, symmetric, expected):
    assert get_links(graph=graph, links=links,
                     subset=subset, symmetric=symmetric) == expected


@pytest.mark.parametrize(
    'graph,start_node,method,expected',
    [
        pytest.param(
            {'a': ['b', 'c'],
             'b': ['a', 'd', 'e'],
             'c': ['a'],
             'd': ['b'],
             'e': ['b'],
             'f': ['g'],
             'g': ['f']},
            'a',
            'bfs',
            collections.OrderedDict(
                {'a': None, 'b': 'a', 'c': 'a', 'd': 'b', 'e': 'b'}),
        ),
        pytest.param(
            {'a': ['b', 'c'],
             'b': ['a', 'd', 'e'],
             'c': ['a'],
             'd': ['b'],
             'e': ['b'],
             'f': ['g'],
             'g': ['f']},
            'a',
            'dfs',
            collections.OrderedDict(
                {'a': None, 'b': 'a', 'd': 'b', 'e': 'b', 'c': 'a'}),
        ),
    ]
)
def test_traverse(graph, start_node, method, expected):
    assert traverse(graph, start_node, method) == expected


@pytest.mark.parametrize(
    'graph,expected',
    [
        pytest.param(
            {'a': ['b', 'c'],
             'b': ['a', 'd', 'e'],
             'c': ['a'],
             'd': ['b'],
             'e': ['b'],
             'f': ['g'],
             'g': ['f'],
             'h': []},
            [{'a', 'b', 'c', 'd', 'e'}, {'f', 'g'}, {'h'}],
        ),
    ]
)
def test_get_subgraphs(graph, expected):
    assert (sorted(get_subgraphs(graph), key=lambda g: len(g)) ==
            sorted(expected, key=lambda g: len(g)))
