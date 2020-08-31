import collections
import numpy as np
import scipy
import scipy.spatial


def make_symmetric(graph):
    """Makes a graph (dict) symmetric inplace.

    Args:
        graph (collections.defaultdict(list)): {k: [v0, v1, ...]}
            with v0, v1, ... being neighbors of k
    """
    for k, vs in graph.copy().items():
        for v in vs:
            if k not in graph[v]:
                graph[v].append(k)


def build_graph(indices, method,
                neighbor_within_swath=None,
                positions=None, neighbor_across_swath=None, max_dist=None,
                verbose=False):
    """Builds an undirected graph that describes an image's neighbors.

    Args:
        indices (list of list of tuples): each tuple corresponds to an image
            identifier, each element in the outer list corresponds to a swath
        method (str): in ['within', 'across', 'all']
            within: make links only between neighbors in the swath
            across: make links only between nearest neighbors across swaths
            all: make links across all pairs of images
        neighbor_within_swath (int): number of links made between an image
            and images on its own swath, cannot be None if method = 'within'
        positions (list of numpy.ndarray [N, 2]): each element in the list
            corresponds to a swath, each row in the ndarray corresponds to the
            centroid of an image, cannot be None if method = 'across'
        neighbor_across_swath (int): number of links made between an image
            and images on each swath (different from its own),
            cannot be None if method = 'across'
        max_dist (float): max distance between images in order for
            a link to be made, cannot be None if method = 'across'
        verbose (bool)

    Returns:
        collections.defaultdict(list): symmetric graph
    """
    # initialize the graph
    graph = collections.defaultdict(list)
    if method == 'within':
        for swath_idx in indices:
            # iterate over every node/image
            for (idx0, idx1) in swath_idx:
                # for images on the same swath
                for i in range(neighbor_within_swath):
                    if (idx0, idx1 + i + 1) in swath_idx:
                        # add the next image in the image sequence to the list
                        graph[(idx0, idx1)] += [(idx0, idx1 + i + 1)]
    elif method == 'across':
        trees = {}
        for i, (swath_idx, swath_pos) in enumerate(zip(indices, positions)):
            # extract centroids, used to calculate distances
            # build kd trees for fast querying of nearest neighbors
            trees[i] = swath_idx, scipy.spatial.cKDTree(swath_pos)
        # iterate over every node/image
        for i, (swath_idx, swath_pos) in enumerate(zip(indices, positions)):
            for (idx0, idx1), img_pos in zip(swath_idx, swath_pos):
                neighbors = []
                for j, (tree_idx, tree) in trees.items():
                    # for images on different swaths
                    if i != j:
                        # get the nearest neighbors for each swath
                        tree_d, tree_i = tree.query(
                            img_pos[np.newaxis, :],
                            k=neighbor_across_swath)
                        # drop ones that are too far apart
                        neighbors += [tree_idx[k] for k in
                                      tree_i[tree_d < max_dist].flatten()]
                if len(neighbors) > 0:
                    graph[(idx0, idx1)] += neighbors
    elif method == 'all':
        # unnest the list to be a list of image indices
        all_indices = set([idx
                           for swath_idx in indices
                           for idx in swath_idx])
        # iterate over every node/image
        for idx in all_indices:
            graph[idx] = list(all_indices - set([idx]))
    else:
        raise NotImplementedError

    make_symmetric(graph)

    if verbose:
        print('Graph: ', graph)

    return graph


def get_links(graph, links, subset=None, symmetric=True):
    """Get all the links associated with a graph.

    Args:
        graph (collections.defaultdict(list)): {k: [v0, v1, ...]}
            the graph defining the universe of links to be considered
        links (dict): containing all links, this is a super set of the
            returned links
        subset (list of tuple of int): list of indices of images to be
            considered, if None, all the links on graph are returned
        symmetric (bool): whether a link should be included twice
            (i to j and j to i), if False, return links with i < j
    """
    if subset is None:
        sub_graph = graph
    else:
        sub_graph = collections.defaultdict(list)
        for i, js in graph.items():
            if i in subset:
                for j in js:
                    if j in subset:
                        sub_graph[i].append(j)
    sub_links = {}
    for i, js in sub_graph.items():
        for j in js:
            if i < j:
                sub_links[(i, j)] = links[(i, j)]
            else:
                if symmetric:
                    sub_links[(i, j)] = links[(i, j)]
    return sub_links


def traverse(graph, start_node, method):
    """Depth/breadth first search on a graph.

    Args:
        graph (dict of list): keys are node names, values are neighbor
            node names
        start_node (same as graph keys): name of the starting node, should
            be in graph keys
        method (str): in ['dfs', 'bfs'] depth/breadth first search

    Returns:
        collections.OrderedDict: order of visiting nodes {node: parent}
    """
    assert method in ['dfs', 'bfs']
    path = collections.OrderedDict()
    # visited is a list of tuples (node, parent), initialize with starting node
    visited = [(start_node, None)]
    while len(visited) > 0:
        if method == 'dfs':
            # pop from the right
            current, parent = visited.pop()
        elif method == 'bfs':
            # pop from the left
            current, parent = visited.pop(0)
        if current not in path.keys():
            path[current] = parent
            # reverse to maintain original priority for DFS
            order = -1 if method == 'dfs' else 1
            for neighbor in graph[current][::order]:
                if neighbor not in path.keys():
                    visited.append((neighbor, current))
            print(current)
    return path


def get_subgraphs(graph):
    """Gets all the connected subgraphs of the graphs.

    Args:
        graph (dict of list): keys are node names, values are neighbor
            node names

    Returns:
        list of set: each subgraph is a set of keys
    """
    subgraphs = []
    not_visited = set(graph.keys())
    while len(not_visited) > 0:
        subgraph = traverse(
            graph=graph, start_node=not_visited.pop(), method='bfs')
        subgraph = set(subgraph.keys())
        not_visited = not_visited - subgraph
        subgraphs.append(subgraph)
    return subgraphs
