efficiently_computable_properties = []


def is_ramanujan(g):
    """
    Returns whether a graph is a Ramanujan graph, that is, if it is k-regular and the absolute value of all non-k eigenvalues is no more than 2*sqrt(k-1)

    Definition from: Lubotzky, Alexander, Ralph Phillips, and Peter Sarnak. "Ramanujan graphs." Combinatorica 8, no. 3 (1988): 261-277.

    EXAMPLES:

        sage: is_ramanujan(graphs.PetersenGraph())
        True
    """
    if not g.is_regular():
        return False
    d = g.degree()[0]
    A = g.adjacency_matrix()
    evals = A.eigenvalues()
    evals.sort(reverse=True)
    X = max(abs(evals[1]),abs(evals[-1]))

    return X <= numerical_approx(2*sqrt(d-1))
add_to_lists(is_ramanujan, efficiently_computable_properties,all_properties)


def has_twin(g):
    """
    Return whether there are vertices v,w with N[v]=N[w].

    INPUT:

    -``g``-- Sage Graph

    OUTPUT:

    -Boolean
    """
    for v in g.vertices(sort=true):
        if is_v_twin(g,v):
            return True
    return False
add_to_lists(has_twin,efficiently_computable_properties,all_properties)


def is_twin_free(g):
    """
    Return True if there are no vertices v,w with N[v]=N[w].

    INPUT:

    -``g``-- Sage Graph

    OUTPUT:

    -Boolean
    """
    return not has_twin(g)
add_to_lists(is_twin_free, efficiently_computable_properties, all_properties)


def has_star_center(g):
    """
    Evalutes whether graph ``g`` has a vertex adjacent to all others.

    EXAMPLES:

        sage: has_star_center(flower_with_3_petals)
        True

        sage: has_star_center(c4)
        False

    Edge cases ::

        sage: has_star_center(Graph(1))
        True

        sage: has_star_center(Graph(0))
        False
    """
    return (g.order() - 1) in g.degree()
add_to_lists(has_star_center, efficiently_computable_properties, all_properties)


#a graph has a pair of vertices who dominate the remaining vertices
#remove each vertex and see if the remaining graph has a star center
def has_double_star_center(g):
    V=g.vertices(sort=true)
    for v in V:
        Hv=copy(V)
        Hv.remove(v)
        H=g.subgraph(Hv)
        if has_star_center(H):
            return True
    return False
add_to_lists(has_double_star_center, efficiently_computable_properties, all_properties)


def is_complement_of_chordal(g):
    """
    Evaluates whether graph ``g`` is a complement of a chordal graph.

    A chordal graph is one in which all cycles of four or more vertices have a
    chord, which is an edge that is not part of the cycle but connects two
    vertices of the cycle.

    EXAMPLES:

        sage: is_complement_of_chordal(p4)
        True

        sage: is_complement_of_chordal(Graph(4))
        True

        sage: is_complement_of_chordal(p5)
        False

    Any graph without a 4-or-more cycle is vacuously chordal. ::

        sage: is_complement_of_chordal(graphs.CompleteGraph(4))
        True

        sage: is_complement_of_chordal(Graph(3))
        True

        sage: is_complement_of_chordal(Graph(0))
        True
    """
    return g.complement().is_chordal()
add_to_lists(is_complement_of_chordal, efficiently_computable_properties, all_properties)


def pairs_have_unique_common_neighbor(g):
    """
    Evalaute if each pair of vertices in ``g`` has exactly one common neighbor.

    Also known as the friendship property.
    By the Friendship Theorem, the only connected graphs with the friendship
    property are flowers.

    EXAMPLES:

        sage: pairs_have_unique_common_neighbor(flower(5))
        True

        sage: pairs_have_unique_common_neighbor(k3)
        True

        sage: pairs_have_unique_common_neighbor(k4)
        False

        sage: pairs_have_unique_common_neighbor(graphs.CompleteGraph(2))
        False

    Vacuous cases ::

        sage: pairs_have_unique_common_neighbor(Graph(1))
        True

        sage: pairs_have_unique_common_neighbor(Graph(0))
        True
    """
    from itertools import combinations
    for (u,v) in combinations(g.vertices(sort=true), 2):
        if len(common_neighbors(g, u, v)) != 1:
            return False
    return True
add_to_lists(pairs_have_unique_common_neighbor, efficiently_computable_properties, all_properties)


def is_dirac(g):
    """
    Evaluates if graph ``g`` has order at least 3 and min. degree at least n/2.

    See Dirac's Theorem: If graph is_dirac, then it is hamiltonian.

    EXAMPLES:

        sage: is_dirac(graphs.CompleteGraph(6))
        True

        sage: is_dirac(graphs.CompleteGraph(3))
        True

        sage: is_dirac(graphs.CompleteGraph(2))
        False

        sage: is_dirac(graphs.CycleGraph(5))
        False
    """
    n = g.order()
    return n > 2 and min(g.degree()) >= n/2
add_to_lists(is_dirac, efficiently_computable_properties, all_properties)


def is_ore(g):
    """
    Evaluate if deg(v)+deg(w)>=n for all non-adjacent pairs v,w in graph ``g``.

    See Ore's Theorem: If graph is_ore, then it is hamiltonian.

    EXAMPLES:

        sage: is_ore(graphs.CompleteGraph(5))
        True

        sage: is_ore(graphs.CompleteGraph(2))
        True

        sage: is_ore(dart)
        False

        sage: is_ore(Graph(2))
        False

        sage: is_ore(graphs.CompleteGraph(2).disjoint_union(Graph(1)))
        False

    Vacous cases ::

        sage: is_ore(Graph(0))
        True

        sage: is_ore(Graph(1))
        True
    """
    A = g.adjacency_matrix()
    n = g.order()
    D = g.degree()
    for i in range(n):
        for j in range(i):
            if A[i][j]==0:
                if D[i] + D[j] < n:
                    return False
    return True
add_to_lists(is_ore, efficiently_computable_properties, all_properties)


def is_haggkvist_nicoghossian(g):
    r"""
    Evaluates if g is 2-connected and min degree >= (n + vertex_connectivity)/3.

    INPUT:

    - ``g`` -- graph

    EXAMPLES:

        sage: is_haggkvist_nicoghossian(graphs.CompleteGraph(3))
        True

        sage: is_haggkvist_nicoghossian(graphs.CompleteGraph(5))
        True

        sage: is_haggkvist_nicoghossian(graphs.CycleGraph(5))
        False

        sage: is_haggkvist_nicoghossian(graphs.CompleteBipartiteGraph(4,3))
        False

        sage: is_haggkvist_nicoghossian(Graph(1))
        False

        sage: is_haggkvist_nicoghossian(graphs.CompleteGraph(2))
        False

    REFERENCES:

    Theorem: If a graph ``is_haggkvist_nicoghossian``, then it is Hamiltonian.

    .. [HN1981]     \\R. Häggkvist and G. Nicoghossian, "A remark on Hamiltonian
                    cycles". Journal of Combinatorial Theory, Series B, 30(1):
                    118--120, 1981.
    """
    k = g.vertex_connectivity()
    return k >= 2 and min(g.degree()) >= (1.0/3) * (g.order() + k)
add_to_lists(is_haggkvist_nicoghossian , efficiently_computable_properties, all_properties)


def is_genghua_fan(g):
    r"""
    Evaluates if graph ``g`` satisfies a condition for Hamiltonicity by G. Fan.

    Returns ``True`` if ``g`` is 2-connected and satisfies that
    `dist(u,v)=2` implies `\max(deg(u), deg(v)) \geq n/2` for all
    vertices `u,v`.
    Returns ``False`` otherwise.

    INPUT:

    -``g``-- Sage Graph

    OUTPUT:

    -Boolean

    EXAMPLES:

        sage: is_genghua_fan(graphs.DiamondGraph())
        True

        sage: is_genghua_fan(graphs.CycleGraph(4))
        True

        sage: is_genghua_fan(graphs.ButterflyGraph())
        False

        sage: is_genghua_fan(Graph(1))
        False

    REFERENCES:

    Theorem: If a graph ``is_genghua_fan``, then it is Hamiltonian.

    .. [Fan1984]    Geng-Hua Fan, "New sufficient conditions for cycles in
                    graphs". Journal of Combinatorial Theory, Series B, 37(3):
                    221--227, 1984.
    """
    if not is_two_connected(g):
        return False
    D = g.degree()
    Dist = g.distance_all_pairs()
    V = g.vertices(sort=true)
    n = g.order()
    for i in range(n):
        for j in range(i):
            if Dist[V[i]][V[j]] == 2 and max(D[i], D[j]) < n / 2.0:
                return False
    return True
add_to_lists(is_genghua_fan, efficiently_computable_properties, all_properties)


def is_generalized_dirac(g):
    r"""
    Test if ``graph`` g meets condition in a generalization of Dirac's Theorem.

    Returns ``True`` if g is 2-connected and for all non-adjacent u,v,
    the cardinality of the union of neighborhood(u) and neighborhood(v)
    is `>= (2n-1)/3`.

    INPUT:

    -``g``-- Sage Graph

    OUTPUT:

    -Boolean

    EXAMPLES:

        sage: is_generalized_dirac(graphs.HouseGraph())
        True

        sage: is_generalized_dirac(graphs.PathGraph(5))
        False

        sage: is_generalized_dirac(graphs.DiamondGraph())
        False

        sage: is_generalized_dirac(Graph(1))
        False

    REFERENCES:

    Theorem: If graph g is_generalized_dirac, then it is Hamiltonian.

    .. [FGJS1989]   \R.J. Faudree, Ronald Gould, Michael Jacobson, and
                    R.H. Schelp, "Neighborhood unions and hamiltonian
                    properties in graphs". Journal of Combinatorial
                    Theory, Series B, 47(1): 1--9, 1989.
    """
    from itertools import combinations

    if not is_two_connected(g):
        return False
    for (u,v) in combinations(g.vertices(sort=true), 2):
        if not g.has_edge(u,v):
            if len(neighbors_set(g,[u,v])) < (2.0 * g.order() - 1) / 3:
                return False
    return True
add_to_lists(is_generalized_dirac, efficiently_computable_properties, all_properties)


def is_van_den_heuvel(g):
    r"""
    Evaluates if g meets an eigenvalue condition related to Hamiltonicity.

    INPUT:

    - ``g`` -- graph

    OUTPUT:

    - Boolean

    Let ``g`` be of order `n`.
    Let `A_H` denote the adjacency matrix of a graph `H`, and `D_H` denote
    the matrix with the degrees of the vertices of `H` on the diagonal.
    Define `Q_H = D_H + A_H` and `L_H = D_H - A_H` (i.e. the Laplacian).
    Finally, let `C` be the cycle graph on `n` vertices.

    Returns ``True`` if the `i`-th eigenvalue of `L_C` is at most the `i`-th
    eigenvalue of `L_g` and the `i`-th eigenvalue of `Q_C` is at most the
    `i`-th eigenvalue of `Q_g for all `i`.

    EXAMPLES:

        sage: is_van_den_heuvel(graphs.CycleGraph(5))
        True

        sage: is_van_den_heuvel(graphs.PetersenGraph())
        False

    REFERENCES:

    Theorem: If a graph is Hamiltonian, then it ``is_van_den_heuvel``.

    .. [Heu1995]    \\J.van den Heuvel, "Hamilton cycles and eigenvalues of
                    graphs". Linear Algebra and its Applications, 226--228:
                    723--730, 1995.

    TESTS::

        sage: is_van_den_heuvel(Graph(1))
        True
    """
    cycle_n = graphs.CycleGraph(g.order())

    cycle_laplac_eigen = sorted(cycle_n.laplacian_matrix().eigenvalues())
    g_laplac_eigen = sorted(g.laplacian_matrix().eigenvalues())
    for cycle_lambda_i, g_lambda_i in zip(cycle_laplac_eigen, g_laplac_eigen):
        if cycle_lambda_i > g_lambda_i:
            return False

    def Q(g):
        A = g.adjacency_matrix(sparse=False)
        D = matrix(g.order(), sparse=False)
        row_sums = [sum(r) for r in A.rows()]
        for i in range(A.nrows()):
            D[i,i] = row_sums[i]
        return D + A
    cycle_q_matrix = sorted(Q(cycle_n).eigenvalues())
    g_q_matrix = sorted(Q(g).eigenvalues())
    for cycle_q_lambda_i, g_q_lambda_i in zip(cycle_q_matrix, g_q_matrix):
        if cycle_q_lambda_i > g_q_lambda_i:
            return False

    return True
add_to_lists(is_van_den_heuvel, efficiently_computable_properties, all_properties)


def is_two_connected(g):
    """
    Evaluates whether graph ``g`` is 2-connected.

    A 2-connected graph is a connected graph on at least 3 vertices such that
    the removal of any single vertex still gives a connected graph.
    Follows convention that complete graph `K_n` is `n-1`-connected.

    Almost equivalent to ``Graph.is_biconnected()``. We prefer our name. AND,
    while that method defines that ``graphs.CompleteGraph(2)`` is biconnected,
    we follow the convention that `K_n` is `n-1`-connected, so `K_2` is
    only 1-connected.

    INPUT:
    - ``g`` -- graph

    OUTPUT:
    - Boolean

    EXAMPLES:

        sage: is_two_connected(graphs.CycleGraph(5))
        True

        sage: is_two_connected(graphs.CompleteGraph(3))
        True

        sage: is_two_connected(graphs.PathGraph(5))
        False

        sage: is_two_connected(graphs.CompleteGraph(2))
        False

        sage: is_two_connected(Graph(3))
        False

    Edge cases ::

        sage: is_two_connected(Graph(0))
        False

        sage: is_two_connected(Graph(1))
        False
    """
    if g.is_isomorphic(graphs.CompleteGraph(2)):
        return False
    return g.is_biconnected()
add_to_lists(is_two_connected, efficiently_computable_properties, all_properties)


def is_three_connected(g):
    """
    Evaluates whether graph ``g`` is 3-connected.

    A 3-connected graph is a connected graph on at least 4 vertices such that
    the removal of any two vertices still gives a connected graph.
    Follows convention that complete graph `K_n` is `n-1`-connected.

    INPUT:
    - ``g`` -- graph

    OUTPUT:
    - Boolean

    EXAMPLES:

        sage: is_three_connected(graphs.PetersenGraph())
        True

        sage: is_three_connected(graphs.CompleteGraph(4))
        True

        sage: is_three_connected(graphs.CycleGraph(5))
        False

        sage: is_three_connected(graphs.PathGraph(5))
        False

        sage: is_three_connected(graphs.CompleteGraph(3))
        False

        sage: is_three_connected(graphs.CompleteGraph(2))
        False

        sage: is_three_connected(Graph(4))
        False

    Edge cases ::

        sage: is_three_connected(Graph(0))
        False

        sage: is_three_connected(Graph(1))
        False

    .. WARNING::

        Implementation requires Sage 8.2+.
    """
    return g.vertex_connectivity(k = 3)
add_to_lists(is_three_connected, efficiently_computable_properties, all_properties)


def is_four_connected(g):
    """
    Evaluates whether ``g`` is 4-connected.

    A 4-connected graph is a connected graph on at least 5 vertices such that
    the removal of any three vertices still gives a connected graph.
    Follows convention that complete graph `K_n` is `n-1`-connected.

    EXAMPLES:


        sage: is_four_connected(graphs.CompleteGraph(5))
        True

        sage: is_four_connected(graphs.PathGraph(5))
        False

        sage: is_four_connected(Graph(5))
        False

        sage: is_four_connected(graphs.CompleteGraph(4))
        False

    Edge cases ::

        sage: is_four_connected(Graph(0))
        False

        sage: is_four_connected(Graph(1))
        False

    .. WARNING::

        Implementation requires Sage 8.2+.
    """
    return g.vertex_connectivity(k = 4)
add_to_lists(is_four_connected, efficiently_computable_properties, all_properties)


def is_lindquester(g):
    r"""
    Test if graph ``g`` meets a neighborhood union condition for Hamiltonicity.

    OUTPUT:

    Let ``g`` be of order `n`.

    Returns ``True`` if ``g`` is 2-connected and for all vertices `u,v`,
    `dist(u,v) = 2` implies that the cardinality of the union of
    neighborhood(`u`) and neighborhood(`v`) is `\geq (2n-1)/3`.
    Returns ``False`` otherwise.

    EXAMPLES:

        sage: is_lindquester(graphs.HouseGraph())
        True

        sage: is_lindquester(graphs.OctahedralGraph())
        True

        sage: is_lindquester(graphs.PathGraph(3))
        False

        sage: is_lindquester(graphs.DiamondGraph())
        False

    REFERENCES:

    Theorem: If a graph ``is_lindquester``, then it is Hamiltonian.

    .. [Lin1989]    \T.E. Lindquester, "The effects of distance and
                    neighborhood union conditions on hamiltonian properties
                    in graphs". Journal of Graph Theory, 13(3): 335-352,
                    1989.
    """
    if not is_two_connected(g):
        return False
    D = g.distance_all_pairs()
    n = g.order()
    V = g.vertices(sort=true)
    for i in range(n):
        for j in range(i):
            if D[V[i]][V[j]] == 2:
                if len(neighbors_set(g,[V[i],V[j]])) < (2*n-1)/3.0:
                    return False
    return True
add_to_lists(is_lindquester, efficiently_computable_properties, all_properties)


def is_complete(g):
    """
    Tests whether ``g`` is a complete graph.

    OUTPUT:

    Returns ``True`` if ``g`` is a complete graph; returns ``False`` otherwise.
    A complete graph is one where every vertex is connected to every others
    vertex.

    EXAMPLES:

        sage: is_complete(graphs.CompleteGraph(1))
        True

        sage: is_complete(graphs.CycleGraph(3))
        True

        sage: is_complete(graphs.CompleteGraph(6))
        True

        sage: is_complete(Graph(0))
        True

        sage: is_complete(graphs.PathGraph(5))
        False

        sage: is_complete(graphs.CycleGraph(4))
        False
    """
    n = g.order()
    e = g.size()
    if not g.has_multiple_edges():
        return e == n*(n-1)/2
    else:
        D = g.distance_all_pairs()
        for i in range(n):
            for j in range(i):
                if D[V[i]][V[j]] != 1:
                    return False
    return True
add_to_lists(is_complete, efficiently_computable_properties, all_properties)


def has_c4(g):
    """
    Tests whether graph ``g`` contains Cycle_4 as an *induced* subgraph.

    EXAMPLES:

        sage: has_c4(graphs.CycleGraph(4))
        True

        sage: has_c4(graphs.HouseGraph())
        True

        sage: has_c4(graphs.CycleGraph(5))
        False

        sage: has_c4(graphs.DiamondGraph())
        False
    """
    return g.subgraph_search(c4, induced=True) is not None
add_to_lists(has_c4, efficiently_computable_properties, all_properties)


def is_c4_free(g):
    """
    Tests whether graph ``g`` does not contain Cycle_4 as an *induced* subgraph.

    EXAMPLES:

        sage: is_c4_free(graphs.CycleGraph(4))
        False

        sage: is_c4_free(graphs.HouseGraph())
        False

        sage: is_c4_free(graphs.CycleGraph(5))
        True

        sage: is_c4_free(graphs.DiamondGraph())
        True
    """
    return not has_c4(g)
add_to_lists(is_c4_free, efficiently_computable_properties, all_properties)


# contains an induced 5-cycle
def has_c5(g):
    return g.subgraph_search(c5, induced=True) is not None
add_to_lists(has_c5, efficiently_computable_properties, all_properties)


# has no induced 5-cycle
def is_c5_free(g):
    return not has_c5(g)
add_to_lists(is_c5_free, efficiently_computable_properties, all_properties)


def has_paw(g):
    """
    Tests whether graph ``g`` contains a Paw as an *induced* subgraph.

    OUTPUT:

    Define a Paw to be a 4-vertex graph formed by a triangle and a pendant.
    Returns ``True`` if ``g`` contains a Paw as an induced subgraph.
    Returns ``False`` otherwise.

    EXAMPLES:

        sage: has_paw(paw)
        True

        sage: has_paw(graphs.BullGraph())
        True

        sage: has_paw(graphs.ClawGraph())
        False

        sage: has_paw(graphs.DiamondGraph())
        False
    """
    return g.subgraph_search(paw, induced=True) is not None
add_to_lists(has_paw, efficiently_computable_properties, all_properties)


def is_paw_free(g):
    """
    Tests whether graph ``g`` does not contain a Paw as an *induced* subgraph.

    OUTPUT:

    Define a Paw to be a 4-vertex graph formed by a triangle and a pendant.
    Returns ``False`` if ``g`` contains a Paw as an induced subgraph.
    Returns ``True`` otherwise.

    EXAMPLES:

        sage: is_paw_free(paw)
        False

        sage: is_paw_free(graphs.BullGraph())
        False

        sage: is_paw_free(graphs.ClawGraph())
        True

        sage: is_paw_free(graphs.DiamondGraph())
        True
    """
    return not has_paw(g)
add_to_lists(is_paw_free, efficiently_computable_properties, all_properties)


def has_dart(g):
    """
    Tests whether graph ``g`` contains a Dart as an *induced* subgraph.

    OUTPUT:

    Define a Dart to be a 5-vertex graph formed by ``graphs.DiamondGraph()``
    with and a pendant added to one of the degree-3 vertices.
    Returns ``True`` if ``g`` contains a Dart as an induced subgraph.
    Returns ``False`` otherwise.

    EXAMPLES:

        sage: has_dart(dart)
        True

        sage: has_dart(umbrella_4)
        True

        sage: has_dart(graphs.DiamondGraph())
        False

        sage: has_dart(bridge)
        False
    """
    return g.subgraph_search(dart, induced=True) is not None
add_to_lists(has_dart, efficiently_computable_properties, all_properties)


def is_dart_free(g):
    """
    Tests whether graph ``g`` does not contain a Dart as an *induced* subgraph.

    OUTPUT:

    Define a Dart to be a 5-vertex graph formed by ``graphs.DiamondGraph()``
    with and a pendant added to one of the degree-3 vertices.
    Returns ``False`` if ``g`` contains a Dart as an induced subgraph.
    Returns ``True`` otherwise.

    EXAMPLES:

        sage: is_dart_free(dart)
        False

        sage: is_dart_free(umbrella_4)
        False

        sage: is_dart_free(graphs.DiamondGraph())
        True

        sage: is_dart_free(bridge)
        True
    """
    return not has_dart(g)
add_to_lists(is_dart_free, efficiently_computable_properties, all_properties)


def has_p4(g):
    """
    Tests whether graph ``g`` contains a Path_4 as an *induced* subgraph.

    Might also be known as "is not a cograph".

    EXAMPLES:

        sage: has_p4(graphs.PathGraph(4))
        True

        sage: has_p4(graphs.CycleGraph(5))
        True

        sage: has_p4(graphs.CycleGraph(4))
        False

        sage: has_p4(graphs.CompleteGraph(5))
        False
    """
    return g.subgraph_search(p4, induced=True) is not None
add_to_lists(has_p4, efficiently_computable_properties, all_properties)


def is_p4_free(g):
    """
    Equivalent to is a cograph - https://en.wikipedia.org/wiki/Cograph
    """
    return not has_p4(g)
add_to_lists(is_p4_free, efficiently_computable_properties, all_properties)


def has_kite(g):
    """
    Tests whether graph ``g`` contains a Kite as an *induced* subgraph.

    A Kite is a 5-vertex graph formed by a ``graphs.DiamondGraph()`` with a
    pendant attached to one of the degree-2 vertices.

    EXAMPLES:

        sage: has_kite(kite_with_tail)
        True

        sage: has_kite(graphs.KrackhardtKiteGraph())
        True

        sage: has_kite(graphs.DiamondGraph())
        False

        sage: has_kite(bridge)
        False
    """
    return g.subgraph_search(kite_with_tail, induced=True) is not None
add_to_lists(has_kite, efficiently_computable_properties, all_properties)


def is_kite_free(g):
    """
    Tests whether graph ``g`` does not contain a Kite as an *induced* subgraph.

    A Kite is a 5-vertex graph formed by a ``graphs.DiamondGraph()`` with a
    pendant attached to one of the degree-2 vertices.

    EXAMPLES:

        sage: is_kite_free(kite_with_tail)
        False

        sage: is_kite_free(graphs.KrackhardtKiteGraph())
        False

        sage: is_kite_free(graphs.DiamondGraph())
        True

        sage: is_kite_free(bridge)
        True
    """
    return not has_kite(g)
add_to_lists(is_kite_free, efficiently_computable_properties, all_properties)


def has_claw(g):
    """
    Tests whether graph ``g`` contains a Claw as an *induced* subgraph.

    A Claw is a 4-vertex graph with one central vertex and 3 pendants.
    This is encoded as ``graphs.ClawGraph()``.

    EXAMPLES:

        sage: has_claw(graphs.ClawGraph())
        True

        sage: has_claw(graphs.PetersenGraph())
        True

        sage: has_claw(graphs.BullGraph())
        False

        sage: has_claw(graphs.HouseGraph())
        False
    """
    return g.subgraph_search(graphs.ClawGraph(), induced=True) is not None
add_to_lists(has_claw, efficiently_computable_properties, all_properties)


def is_claw_free(g):
    """
    Tests whether graph ``g`` does not contain a Claw as an *induced* subgraph.

    A Claw is a 4-vertex graph with one central vertex and 3 pendants.
    This is encoded as ``graphs.ClawGraph()``.

    EXAMPLES:

        sage: is_claw_free(graphs.ClawGraph())
        False

        sage: is_claw_free(graphs.PetersenGraph())
        False

        sage: is_claw_free(graphs.BullGraph())
        True

        sage: is_claw_free(graphs.HouseGraph())
        True
    """
    return not has_claw(g)
add_to_lists(is_claw_free, efficiently_computable_properties, all_properties)


def has_H(g):
    """
    Tests whether graph ``g`` contains an H graph as an *induced* subgraph.

    An H graph may also be known as a double fork. It is a 6-vertex graph
    formed by two Path_3s with their midpoints joined by a bridge.

    EXAMPLES:

        sage: has_H(double_fork)
        True

        sage: has_H(graphs.PetersenGraph())
        True

        sage: has_H(ce71) # double_fork with extra edge
        False

        sage: has_H(graphs.BullGraph())
        False
    """
    return g.subgraph_search(double_fork, induced=True) is not None
add_to_lists(has_H, efficiently_computable_properties, all_properties)


def is_H_free(g):
    """
    Tests if graph ``g`` does not contain a H graph as an *induced* subgraph.

    An H graph may also be known as a double fork. It is a 6-vertex graph
    formed by two Path_3s with their midpoints joined by a bridge.

    EXAMPLES:

        sage: is_H_free(double_fork)
        False

        sage: is_H_free(graphs.PetersenGraph())
        False

        sage: is_H_free(ce71) # double_fork with extra edge
        True

        sage: is_H_free(graphs.BullGraph())
        True
    """
    return not has_H(g)
add_to_lists(is_H_free, efficiently_computable_properties, all_properties)


def has_fork(g):
    """
    Tests if graph ``g`` contains a Fork graph as an *induced* subgraph.

    A Fork graph may also be known as a Star_1_1_3. It is a 6-vertex graph
    formed by a Path_4 with two pendants connected to one end.
    It is stored as `star_1_1_3`.

    EXAMPLES:

        sage: has_fork(star_1_1_3)
        True

        sage: has_fork(graphs.PetersenGraph())
        True

        sage: has_fork(graphs.LollipopGraph(3, 2))
        False

        sage: has_fork(graphs.HouseGraph())
        False

        sage: has_fork(graphs.ClawGraph())
        False
    """
    return g.subgraph_search(star_1_1_3, induced=True) is not None
add_to_lists(has_fork, efficiently_computable_properties, all_properties)


def is_fork_free(g):
    """
    Tests if graph ``g`` does not contain Fork graph as an *induced* subgraph.

    A Fork graph may also be known as a Star_1_1_3. It is a 6-vertex graph
    formed by a Path_4 with two pendants connected to one end.
    It is stored as `star_1_1_3`.

    EXAMPLES:

        sage: is_fork_free(star_1_1_3)
        False

        sage: is_fork_free(graphs.PetersenGraph())
        False

        sage: is_fork_free(graphs.LollipopGraph(3, 2))
        True

        sage: is_fork_free(graphs.HouseGraph())
        True

        sage: is_fork_free(graphs.ClawGraph())
        True
    """
    return not has_fork(g)
add_to_lists(is_fork_free, efficiently_computable_properties, all_properties)


def has_k4(g):
    """
    Tests if graph ``g`` contains a `K_4` as an *induced* subgraph.

    `K_4` is the complete graph on 4 vertices.

    EXAMPLES:

        sage: has_k4(graphs.CompleteGraph(4))
        True

        sage: has_k4(graphs.CompleteGraph(5))
        True

        sage: has_k4(graphs.CompleteGraph(3))
        False

        sage: has_k4(graphs.PetersenGraph())
        False
    """
    return g.subgraph_search(alpha_critical_easy[2], induced=True) is not None
add_to_lists(has_k4, efficiently_computable_properties, all_properties)


def is_k4_free(g):
    """
    Tests if graph ``g`` does not contain a `K_4` as an *induced* subgraph.

    `K_4` is the complete graph on 4 vertices.

    EXAMPLES:

        sage: is_k4_free(graphs.CompleteGraph(4))
        False

        sage: is_k4_free(graphs.CompleteGraph(5))
        False

        sage: is_k4_free(graphs.CompleteGraph(3))
        True

        sage: is_k4_free(graphs.PetersenGraph())
        True
    """
    return not has_k4(g)
add_to_lists(is_k4_free, efficiently_computable_properties, all_properties)


def is_double_clique(g):
    """
    Tests if graph ``g`` can be partitioned into 2 sets which induce cliques.

    EXAMPLE:

        sage: is_double_clique(p4)
        True

        sage: is_double_clique(graphs.ButterflyGraph())
        True

        sage: is_double_clique(graphs.CompleteBipartiteGraph(3,4))
        False

        sage: is_double_clique(graphs.ClawGraph())
        False

        sage: is_double_clique(Graph(3))
        False

    Edge cases ::

        sage: is_double_clique(Graph(0))
        True

        sage: is_double_clique(Graph(1))
        True

        sage: is_double_clique(Graph(2))
        True
    """
    gc = g.complement()
    return gc.is_bipartite()
add_to_lists(is_double_clique, efficiently_computable_properties, all_properties)


def has_radius_equal_diameter(g):
    """
    Evaluates whether the radius of graph ``g`` equals its diameter.

    Recall the radius of a graph is the minimum eccentricity over all vertices,
    or the minimum over all longest distances from a vertex to any other vertex.
    Diameter is the maximum eccentricity over all vertices.
    Both radius and diamter are defined to be `+Infinity` for disconnected
    graphs.

    Both radius and diameter are undefined for the empty graph.

    EXAMPLES:

        sage: has_radius_equal_diameter(Graph(4))
        True

        sage: has_radius_equal_diameter(graphs.HouseGraph())
        True

        sage: has_radius_equal_diameter(graphs.ClawGraph())
        False

        sage: has_radius_equal_diameter(graphs.BullGraph())
        False
    """
    return g.radius() == g.diameter()
add_to_lists(has_radius_equal_diameter, efficiently_computable_properties, all_properties)


def is_not_forest(g):
    """
    Evaluates if graph ``g`` is not a forest.

    A forest is a disjoint union of trees. Equivalently, a forest is any acylic
    graph, which may or may not be connected.

    EXAMPLES:
        sage: is_not_forest(graphs.BalancedTree(2,3))
        False

        sage: is_not_forest(graphs.BalancedTree(2,3).disjoint_union(graphs.BalancedTree(3,3)))
        False

        sage: is_not_forest(graphs.CycleGraph(5))
        True

        sage: is_not_forest(graphs.HouseGraph())
        True

    Edge cases ::

        sage: is_not_forest(Graph(1))
        False

        sage: is_not_forest(Graph(0))
        False
    """
    return not g.is_forest()
add_to_lists(is_not_forest, efficiently_computable_properties, all_properties)


def has_empty_KE_part(g):
    r"""
    Evaluates whether graph ``g`` has an empty Konig-Egervary subgraph.

    A Konig-Egervary graph satisfies
        independence number + matching number = order.
    By [Lar2011]_, every graph contains a unique induced subgraph which is a
    Konig-Egervary graph.

    EXAMPLES:

        sage: has_empty_KE_part(graphs.PetersenGraph())
        True

        sage: has_empty_KE_part(graphs.CycleGraph(5))
        True

        sage: has_empty_KE_part(graphs.CompleteBipartiteGraph(3,4))
        False

        sage: has_empty_KE_part(graphs.CycleGraph(6))
        False

    Edge cases ::

        sage: has_empty_KE_part(Graph(1))
        False

        sage: has_empty_KE_part(Graph(0))
        True

    ALGORITHM:

    This function is implemented using the Maximum Critical Independent
    Set (MCIS) algorithm of [DL2013]_ and applying a Theorem of [Lar2011]_.

    Define that an independent set `I` is a critical independent set if
    `|I|−|N(I)| \geq |J|−|N(J)|` for any independent set J. Define that a
    maximum critical independent set is a critical independent set of maximum
    cardinality.

    By a Theorem of [Lar2011]_, for every maximum critical independent set `J`
    of `G`, the unique Konig-Egervary inducing set `X` is `X = J \cup N(J)`,
    where `N(J)` is the neighborhood of `J`.
    Therefore, the ``g`` has an empty Konig-Egervary induced subgraph if and
    only if the MCIS `J = \emptyset`.

    Next, we describe the MCIS algorithm.
    Let `B(G) = K_2 \ times G`, i.e. `B(G)` is the bipartite double cover
    of `G`. Let `v' \in B(G)` denote the new "twin" of vertex `v \in G`.
    Let `a` be the independence number of `B(G)`.
    For each vertex `v` in `B(G)`, calculate
        `t := independence number(B(G) - \{v,v'\} - N(\{v,v'\})) + 2`.
    If `t = a`, then `v` is in the MCIS.
        Since we only care about whether the MCIS is empty, if `t = a`,
        we return ``False`` and terminate.

    Finally, use the Gallai identities to show matching

    Finally, we apply the Konig-Egervary Theorem (1931) that for all bipartite
    graphs, matching number = vertex cover number. We substitute this into
    one of the Gallai identities, that
        independence number + covering number = order,
    yielding,
        independence number = order - matching number.
    Since matching number is efficient to compute, our final algorithm is
    in fact efficient.

    REFERENCES:

    .. [DL2013]     \Ermelinda DeLaVina and Craig Larson, "A parallel ALGORITHM
                    for computing the critical independence number and related
                    sets". ARS Mathematica Contemporanea 6: 237--245, 2013.

    .. [Lar2011]    \C.E. Larson, "Critical Independent Sets and an
                    Independence Decomposition Theorem". European Journal of
                    Combinatorics 32: 294--300, 2011.
    """
    b = bipartite_double_cover(g)
    alpha = b.order() - b.matching(value_only=True)
    for v in g.vertices(sort=true):
        test = b.copy()
        test.delete_vertices(closed_neighborhood(b,[(v,0), (v,1)]))
        alpha_test = test.order() - test.matching(value_only=True) + 2
        if alpha_test == alpha:
            return False
    return True
add_to_lists(has_empty_KE_part, efficiently_computable_properties, all_properties)


def is_cubic(g):
    """
    Evalutes whether graph ``g`` is cubic, i.e. is 3-regular.

    EXAMPLES:

        sage: is_cubic(graphs.CompleteGraph(4))
        True

        sage: is_cubic(graphs.PetersenGraph())
        True

        sage: is_cubic(graphs.CompleteGraph(3))
        False

        sage: is_cubic(graphs.HouseGraph())
        False
    """
    D = g.degree()
    return min(D) == 3 and max(D) == 3
add_to_lists(is_cubic, efficiently_computable_properties, all_properties)


def diameter_equals_twice_radius(g):
    """
    Evaluates whether the diameter of graph ``g`` is equal to twice its radius.

    Diameter and radius are undefined for the empty graph.

    EXAMPLES:

        sage: diameter_equals_twice_radius(graphs.ClawGraph())
        True

        sage: diameter_equals_twice_radius(graphs.KrackhardtKiteGraph())
        True

        sage: diameter_equals_twice_radius(graphs.HouseGraph())
        False

        sage: diameter_equals_twice_radius(graphs.BullGraph())
        False

    Disconnected graphs have both diameter and radius equal infinity.

        sage: diameter_equals_twice_radius(Graph(4))
        True
    """
    return g.diameter() == 2*g.radius()
add_to_lists(diameter_equals_twice_radius, efficiently_computable_properties, all_properties)


def diameter_equals_two(g):
    """
    Evaluates whether the diameter of graph ``g`` equals 2.

    Diameter is undefined for the empty graph.

    EXAMPLES:

        sage: diameter_equals_two(graphs.ClawGraph())
        True

        sage: diameter_equals_two(graphs.HouseGraph())
        True

        sage: diameter_equals_two(graphs.KrackhardtKiteGraph())
        False

        sage: diameter_equals_two(graphs.BullGraph())
        False

    Disconnected graphs have diameter equals infinity.

        sage: diameter_equals_two(Graph(4))
        False
    """
    return g.diameter() == 2
add_to_lists(diameter_equals_two, efficiently_computable_properties, all_properties)


def matching_covered(g):
    """
    Skipping because broken. See Issue #585.
    """
    g = g.copy()
    nu = matching_number(g)
    E = g.edges(sort=true)
    for e in E:
        g.delete_edge(e)
        nu2 = matching_number(g)
        if nu != nu2:
            return False
        g.add_edge(e)
    return True
add_to_lists(matching_covered, efficiently_computable_properties, all_properties)


def radius_greater_than_center(g):
    """
    Test if connected graph ``g`` has radius greater than num. of center verts.

    If ``g`` is not connected, returns ``False``.
    Radius is undefined for the empty graph.

    EXAMPLES:

        sage: radius_greater_than_center(graphs.TutteGraph())
        True

        sage: radius_greater_than_center(graphs.KrackhardtKiteGraph())
        True

        sage: radius_greater_than_center(graphs.SousselierGraph())
        True

        sage: radius_greater_than_center(graphs.PetersenGraph())
        False

        sage: radius_greater_than_center(graphs.DiamondGraph())
        False
    """
    return g.is_connected() and g.radius() > card_center(g)
add_to_lists(radius_greater_than_center, efficiently_computable_properties, all_properties)


def avg_distance_greater_than_girth(g):
    """
    Tests if graph ``g`` is connected and avg. distance greater than the girth.

    Average distance is undefined for 1- and 0- vertex graphs.

    EXAMPLES:

        sage: avg_distance_greater_than_girth(graphs.TutteGraph())
        True

        sage: avg_distance_greater_than_girth(graphs.HarborthGraph())
        True

        sage: avg_distance_greater_than_girth(graphs.HortonGraph())
        True

        sage: avg_distance_greater_than_girth(graphs.BullGraph())
        False

        sage: avg_distance_greater_than_girth(Graph("NC`@A?_C?@_JA??___W"))
        False

        sage: avg_distance_greater_than_girth(Graph(2))
        False

    Acyclic graphs have girth equals infinity. ::

        sage: avg_distance_greater_than_girth(graphs.CompleteGraph(2))
        False
    """
    return g.is_connected() and g.average_distance() > g.girth()
add_to_lists(avg_distance_greater_than_girth, efficiently_computable_properties, all_properties)


def is_traceable(g):
    """
    Evaluates whether graph ``g`` is traceable.

    A graph ``g`` is traceable iff there exists a Hamiltonian path, i.e. a path
    which visits all vertices in ``g`` once.
    This is different from ``is_hamiltonian``, since that checks if there
    exists a Hamiltonian *cycle*, i.e. a path which then connects backs to
    its starting point.

    EXAMPLES:

        sage: is_traceable(graphs.CompleteGraph(5))
        True

        sage: is_traceable(graphs.PathGraph(5))
        True

        sage: is_traceable(graphs.PetersenGraph())
        True

        sage: is_traceable(graphs.CompleteGraph(2))
        True

        sage: is_traceable(Graph(3))
        False

        sage: is_traceable(graphs.ClawGraph())
        False

        sage: is_traceable(graphs.ButterflyGraph())
        True

    Edge cases ::

        sage: is_traceable(Graph(0))
        False

        sage: is_traceable(Graph(1))
        False

    ALGORITHM:

    A graph `G` is traceable iff the join `G'` of `G` with a single new vertex
    `v` is Hamiltonian, where join means to connect every vertex of `G` to the
    new vertex `v`.
    Why? Suppose there exists a Hamiltonian path between `u` and `w` in `G`.
    Then, in `G'`, make a cycle from `v` to `u` to `w` and back to `v`.
    For the reverse direction, just note that the additional vertex `v` cannot
    "help" since Hamiltonian paths can only visit any vertex once.
    """
    gadd = g.join(Graph(1),labels="integers")
    return gadd.is_hamiltonian()
add_to_lists(is_traceable, efficiently_computable_properties, all_properties)


def has_residue_equals_two(g):
    r"""
    Evaluates whether the residue of graph ``g`` equals 2.

    The residue of a graph ``g`` with degrees `d_1 \geq d_2 \geq ... \geq d_n`
    is found iteratively. First, remove `d_1` from consideration and subtract
    `d_1` from the following `d_1` number of elements. Sort. Repeat this
    process for `d_2,d_3, ...` until only 0s remain. The number of elements,
    i.e. the number of 0s, is the residue of ``g``.

    Residue is undefined on the empty graph.

    EXAMPLES:

        sage: has_residue_equals_two(graphs.ButterflyGraph())
        True

        sage: has_residue_equals_two(graphs.IcosahedralGraph())
        True

        sage: has_residue_equals_two(graphs.OctahedralGraph())
        True

        sage: has_residue_equals_two(Graph(1))
        False

        sage: has_residue_equals_two(graphs.BullGraph())
        False

        sage: has_residue_equals_two(graphs.BrinkmannGraph())
        False
    """
    return residue(g) == 2
add_to_lists(has_residue_equals_two, efficiently_computable_properties, all_properties)


def is_chordal_or_not_perfect(g):
    """
    Evaluates if graph ``g`` is either chordal or not perfect.

    There is a known theorem that every chordal graph is perfect.

    OUTPUT:

    Returns ``True`` iff ``g`` is chordal OR (inclusive or) ``g`` is not
    perfect.

    EXAMPLES:

        sage: is_chordal_or_not_perfect(graphs.DiamondGraph())
        True

        sage: is_chordal_or_not_perfect(graphs.CycleGraph(5))
        True

        sage: is_chordal_or_not_perfect(graphs.LollipopGraph(5,3))
        True

        sage: is_chordal_or_not_perfect(graphs.CycleGraph(4))
        False

        sage: is_chordal_or_not_perfect(graphs.HexahedralGraph())
        False

    Vacuously chordal cases ::

        sage: is_chordal_or_not_perfect(Graph(0))
        True

        sage: is_chordal_or_not_perfect(Graph(1))
        True

        sage: is_complement_of_chordal(Graph(4))
        True
    """
    if g.is_chordal():
        return true
    else:
        return not g.is_perfect()
add_to_lists(is_chordal_or_not_perfect, efficiently_computable_properties, all_properties)


def is_alpha_equals_two(g):
    """
    Return whether the independence number alpha is equal to 2.

    INPUT:

    -``g``-- Sage Graph

    OUTPUT:

    - Boolean Value
    """
    gc = g.complement()
    if gc.is_triangle_free():
        if not is_complete(g):
            return True
    return False
add_to_lists(is_alpha_equals_two, efficiently_computable_properties, all_properties)


def order_leq_twice_max_degree(g):
    """
    Tests if the order of graph ``g`` is at most twice the max of its degrees.

    Undefined for the empty graph.

    EXAMPLES:

        sage: order_leq_twice_max_degree(graphs.BullGraph())
        True

        sage: order_leq_twice_max_degree(graphs.ThomsenGraph())
        True

        sage: order_leq_twice_max_degree(graphs.CycleGraph(4))
        True

        sage: order_leq_twice_max_degree(graphs.BidiakisCube())
        False

        sage: order_leq_twice_max_degree(graphs.CycleGraph(5))
        False

        sage: order_leq_twice_max_degree(Graph(1))
        False
    """
    return (g.order() <= 2*max(g.degree()))
add_to_lists(order_leq_twice_max_degree, efficiently_computable_properties, all_properties)


# graph is KE if matching number + independence number = n, test does *not* compute alpha
def is_KE(g):
    """
    A graph is KE if matching number + independence number is equal to n
    Test does NOT compute alpha
    """
    return g.order() == len(find_KE_part(g))
add_to_lists(is_KE, efficiently_computable_properties, all_properties)

# graph is KE if matching number + independence number = n, test comoutes alpha
# def is_KE(g):
#    return (g.matching(value_only = True) + independence_number(g) == g.order())

# possibly faster version of is_KE (not currently in invariants)
# def is_KE2(g):
#    return (independence_number(g) == critical_independence_number(g))


def is_independence_irreducible(g):
    """
    Return whether or not a graph "g" is independence irreducible.

    A graph is independence_irreducible if every non-empty independent set I has more than |I| neighbors

    INPUT:

    -``g``-- Sage graph

    OUTPUT:

    -Boolean value

    REFERENCE:
    G. Abay-Asmerom, R. Hammack, C. Larson, D. Taylor, "Notes on the independence number in the Cartesian product of graphs".
    Discussiones Mathematicae Graph Theory, 2011
"""
    return g.order() == card_independence_irreducible_part(g)
add_to_lists(is_independence_irreducible, efficiently_computable_properties, all_properties)


def is_factor_critical(g):
    """
    Return whether or not a graph "g" is factor-critical.

    A graph is factor-critical if order is odd and removal of any vertex gives graph with perfect matching

    INPUT:

    -``g``-- Sage graph

    OUTPUT:

    -Boolean value

    EXAMPLES:

        sage: is_factor_critical(graphs.PathGraph(3))
        True

        sage: is_factor_critical(graphs.CycleGraph(5))
        True

    REFERENCES:

    https://en.wikipedia.org/wiki/Factor-critical_graph
    """
    if g.order() % 2 == 0:
        return False
    for v in g.vertices(sort=true):
        gc = copy(g)
        gc.delete_vertex(v)
        if not gc.has_perfect_matching:
            return False
    return True
add_to_lists(is_factor_critical, efficiently_computable_properties, all_properties)


def has_twin(g):
    """
    Return whether or not any vertices within "g" are twin.

    Two vertices are twins if they are non-adjacent and have the same neighbors.

    INPUT:

    -``g``-- Sage graph

    OUTPUT:

    -Boolean value

    """
    t = find_twin(g)
    if t == None:
        return False
    else:
        return True
add_to_lists(has_twin, efficiently_computable_properties, all_properties)


def is_twin_free(g):
    """
    Return whether or not any vertices within "g" are twin.  Returns True if there are none.

    Two vertices are twins if they are non-adjacent and have the same neighbors.

    INPUT:

    -``g``-- Sage graph

    OUTPUT:

    -Boolean value

    """
    return not has_twin(g)
add_to_lists(is_twin_free, efficiently_computable_properties, all_properties)


def girth_greater_than_2log(g):
    """
    Determines if the girth of the graph g is greater than twice the log of the order of g
    Returns true if greater, false if not.

    INPUT:

    -``g``-- Sage graph

    OUTPUT:

    -Boolean

    """
    return bool(g.girth() > 2*log(g.order(),2))
add_to_lists(girth_greater_than_2log, efficiently_computable_properties, all_properties)


def is_subcubic(g):
    """
    Return whether or not a Graph g is subcubic.

    A Graph is subcubic is each vertex is at most degree 3.

    INPUT:

    -``g``- Sage Graph

    OUTPUT:

    - Boolean, True if a graph is subcubic, False otherwise.
    """
    return max_degree(g) <= 3
add_to_lists(is_subcubic, efficiently_computable_properties, all_properties)


# Max and min degree varies by at most 1
def is_quasi_regular(g):
    """
    Return whether or not a graph is quasi-regular.

    A Graph is quasi-regular is its Max and min degree varies by at most 1.

    INPUT:

    -``g``-- Sage Graph

    OUTPUT:

    -Boolean, True if the graph is quasi-regular, False if otherwise.
    """
    if max_degree(g) - min_degree(g) < 2:
        return True
    return False
add_to_lists(is_quasi_regular, efficiently_computable_properties, all_properties)


# g is bad if a block is isomorphic to k3, c5, k4*, c5*
#the complexity will be no more than n/5 checks of small-graph isomorhism
#where does this function/definition originate?
def is_bad(g):
    blocks = g.blocks_and_cut_vertices()[0]
    # To make a subgraph of g from the ith block
    for i in blocks:
        h = g.subgraph(i)
        boolean = h.is_isomorphic(alpha_critical_easy[1]) or h.is_isomorphic(alpha_critical_easy[4]) or h.is_isomorphic(alpha_critical_easy[5]) or h.is_isomorphic(alpha_critical_easy[21])
        if boolean == True:
            return True
    return False
add_to_lists(is_bad, efficiently_computable_properties, all_properties)


# A graph is unicyclic if it is connected and has order == size
# Equivalently, graph is connected and has exactly one cycle
def is_unicyclic(g):
    """
    Return whether the graph g is unicyclic.

    A graph is unicyclic if it is connected and contains only one cycle.

    INPUT:

    -``g``-- Sage Graph

    OUTPUT:

    -Boolean

    Tests:
        sage: is_unicyclic(graphs.BullGraph())
        True
        sage: is_unicyclic(graphs.ButterflyGraph())
        False
    """
    return g.is_connected() and g.order() == g.size()
add_to_lists(is_unicyclic, efficiently_computable_properties, all_properties)


def has_simplicial_vertex(g):
    """
    v is a simplicial vertex if induced neighborhood is a clique.
    """
    for v in g.vertices(sort=true):
        if is_simplicial_vertex(g, v):
            return True
    return False
add_to_lists(has_simplicial_vertex, efficiently_computable_properties, all_properties)


def has_exactly_two_simplicial_vertices(g):
    """
    Returns true if there are precisely two simplicial vertices within g, a graph.
    A vertex is simplicial if the induced neighborhood is a clique.

    INPUT:

    -``g``-- Sage Graph

    OUTPUT:

    -Boolean

    """
    return simplicial_vertices(g) == 2
add_to_lists(has_exactly_two_simplicial_vertices, efficiently_computable_properties, all_properties)


def is_two_tree(g):
    """
    Define k-tree recursively:
        - Complete Graph on (k+1)-vertices is a k-tree
        - A k-tree on n+1 vertices is constructed by selecting some k-tree on n vertices and
            adding a degree k vertex such that its open neighborhood is a clique.
    """
    if(g.is_isomorphic(graphs.CompleteGraph(3))):
        return True

    # We can just recurse from any degree-2 vertex; no need to test is_two_tree(g-w) if is_two_tree(g-v) returns False.
    # Intuition why: if neighborhood of a degree-2 v is not a triangle, it won't become one if we remove w (so clique check OK),
    # and, removing a degree-2 vertex of one triangle cannot destroy another triangle (so recursion OK).
    degree_two_vertices = (v for v in g.vertices(sort=true) if g.degree(v) == 2)
    try:
        v = next(degree_two_vertices)
    except StopIteration: # Empty list. No degree 2 vertices.
        return False

    if not g.has_edge(g.neighbors(v)): # Clique
        return false
    g2 = g.copy()
    g2.delete_vertex(v)
    return is_two_tree(g2)
add_to_lists(is_two_tree, efficiently_computable_properties, all_properties)


def is_two_path(g):
    """
    Graph g is a two_path if it is a two_tree and has exactly 2 simplicial vertices
    """
    return has_exactly_two_simplicial_vertices(g) and is_two_tree(g)
add_to_lists(is_two_path, efficiently_computable_properties, all_properties)


def is_claw_free_paw_free(g):
    return is_claw_free(g) and is_paw_free(g)
add_to_lists(is_claw_free_paw_free, efficiently_computable_properties, all_properties)


def has_bull(g):
    """
    Returns true if a graph has an induced subgraph isomorphic to graphs.BullGraph()

    INPUT:

    -``g``-- Sage Graph

    OUTPUT:

    -Boolean
    """
    return g.subgraph_search(graphs.BullGraph(), induced = True) != None
add_to_lists(has_bull, efficiently_computable_properties, all_properties)


def is_bull_free(g):
    """
    Returns true if g does not have an induced subgraph isomoprhic to graphs.BullGraph()

    INPUT:

    -``g``-- Sage Graph

    OUTPUT:

    -Boolean

    """
    return not has_bull(g)
add_to_lists(is_bull_free, efficiently_computable_properties, all_properties)


def is_claw_free_bull_free(g):
    """
    Returns true if g does not have an induced subgraph isomoprhic to graphs.BullGraph()
    and is claw free.

    INPUT:

    -``g``-- Sage Graph

    OUTPUT:

    -Boolean

    """
    return is_claw_free(g) and is_bull_free(g)
add_to_lists(is_claw_free_bull_free, efficiently_computable_properties, all_properties)


def has_F(g):
    """
    Let F be a triangle with 3 pendants. True if g has an induced F.
    """
    F = graphs.CycleGraph(3)
    F.add_vertices([3,4,5])
    F.add_edges([(0,3), (1,4), (2,5)])
    return g.subgraph_search(F, induced = True) != None
add_to_lists(has_F, efficiently_computable_properties, all_properties)


def is_F_free(g):
    """
    Let F be a triangle with 3 pendants. True if g has no induced F.
    """
    return not has_F(g)
add_to_lists(is_F_free, efficiently_computable_properties, all_properties)


# Ronald Gould, Updating the Hamiltonian problem — a survey. Journal of Graph Theory 15.2: 121-157, 1991.
def is_oberly_sumner(g):
    """
    g is_oberly_sumner if order >= 3, is_two_connected, is_claw_free, AND is_F_free
    """
    return g.order() >= 3 and is_two_connected(g) and is_claw_free(g) and is_F_free(g)
add_to_lists(is_oberly_sumner, efficiently_computable_properties, all_properties)


def is_oberly_sumner_bull(g):
    """
    True if g is 2-connected, claw-free, and bull-free
    """
    return is_two_connected(g) and is_claw_free_bull_free(g)
add_to_lists(is_oberly_sumner_bull, efficiently_computable_properties, all_properties)


def is_oberly_sumner_p4(g):
    """
    True if g is 2-connected, claw-free, and p4-free
    """
    return is_two_connected(g) and is_claw_free(g) and is_p4_free(g)
add_to_lists(is_oberly_sumner_p4, efficiently_computable_properties, all_properties)


# Ronald Gould, Updating the Hamiltonian problem — a survey. Journal of Graph Theory 15.2: 121-157, 1991.
def is_matthews_sumner(g):
    """
    True if g is 2-connected, claw-free, and minimum-degree >= (order-1) / 3
    """
    return is_two_connected(g) and is_claw_free(g) and min_degree(g) >= (g.order() - 1) / 3
add_to_lists(is_matthews_sumner, efficiently_computable_properties, all_properties)


def is_broersma_veldman_gould(g):
    """
    True if g is 2-connected, claw-free, and diameter <= 2
    """
    return is_two_connected(g) and is_claw_free(g) and g.diameter() <= 2
add_to_lists(is_broersma_veldman_gould, efficiently_computable_properties, all_properties)


def chvatals_condition(g):
    """
    Returns true if g.order()>=3 and given increasing degrees d_1,..,d_n, for all i, i>=n/2 or d_i>i or d_{n-i}>=n-1

    INPUT:

    -``g``-- Sage Graph

    OUTPUT:

    -Boolean

    REFERENCES:

    This condition is based on Thm 1 of
    Chvátal, Václav. "On Hamilton's ideals." Journal of Combinatorial Theory, Series B 12.2 (1972): 163-168.

    [Chvatal, 72] also showed this condition is sufficient to imply g is Hamiltonian.
    """
    if g.order() < 3:
        return False
    degrees = g.degree()
    degrees.sort()
    n = g.order()
    return all(degrees[i] > i or i >= n/2 or degrees[n-i] >= n-i for i in range(0, len(degrees)))
add_to_lists(chvatals_condition, efficiently_computable_properties, all_properties)


def is_matching(g):
    """
    Returns True if the graph g is the disjoint union of complete graphs of order 2.

    INPUT:

    -``g``-- Sage Graph

    OUTPUT:

    -Boolean

    TESTS:
        sage: is_matching(graphs.CompleteGraph(2))
        True
        sage: is_matching(graphs.PathGraph(4))
        False
        sage: is_matching(graphs.CompleteGraph(2).disjoint_union(graphs.CompleteGraph(2)))
        True
    """
    return min(g.degree())==1 and max(g.degree())==1
add_to_lists(is_matching, efficiently_computable_properties, all_properties)


def has_odd_order(g):
    """
    True if the number of vertices in g is odd

    sage: has_odd_order(Graph(5))
    True
    sage: has_odd_order(Graph(2))
    False
    """
    return g.order() % 2 == 1
add_to_lists(has_odd_order, efficiently_computable_properties, all_properties)


def has_even_order(g):
    """
    True if the number of vertices in g is even

    sage: has_even_order(Graph(5))
    False
    sage: has_even_order(Graph(2))
    True
    """
    return g.order() % 2 == 0
add_to_lists(has_even_order, efficiently_computable_properties, all_properties)


def is_maximal_triangle_free(g):
    """
    Evaluates whether graph g is a maximal triangle-free graph.  A graph is maximal triangle-free if adding any edge to g
    will create a triangle.
    If g is not triangle-free, returns false.

    INPUT:

    -``g``-- Sage Graph

    OUTPUT:

    -Boolean

    EXAMPLES:

        sage: is_maximal_triangle_free(graphs.CompleteGraph(2))
        True

        sage: is_maximal_triangle_free(graphs.CycleGraph(5))
        True

        sage: is_maximal_triangle_free(Graph('Esa?'))
        True

        sage: is_maximal_triangle_free(Graph('KsaCCA?_C?O?'))
        True

        sage: is_maximal_triangle_free(graphs.PathGraph(5))
        False

        sage: is_maximal_triangle_free(Graph('LQY]?cYE_sBOE_'))
        False

        sage: is_maximal_triangle_free(graphs.HouseGraph())
        False

    Edge cases ::

        sage: is_maximal_triangle_free(Graph(1))
        True

        sage: is_maximal_triangle_free(Graph(3))
        False
    """
    if not g.is_triangle_free():
        return False
    g_comp = g.complement() #has edges which are non-edges in the original graph
    g_copy = g.copy()
    for e in g_comp.edges(sort=true):
        g_copy.add_edge(e)
        if g_copy.is_triangle_free():
            return False
        g_copy.delete_edge(e)
    return True
add_to_lists(is_maximal_triangle_free, efficiently_computable_properties, all_properties)


def is_locally_two_connected(g):
    for v in g.vertices(sort=true):
        S=g.neighbors(v)
        if len(S)>2: #not defined unless there are at least 3 neighbors
            h=g.subgraph(S)
            if not is_two_connected(h):
                return False
    return True #all neighborhoods are too small or are connected
add_to_lists(is_locally_two_connected, efficiently_computable_properties, all_properties)


def is_2_bootstrap_good(G):
    """
    Return whether or not G contains a subset of 2 vertices which 2-infect the whole graph.

    Assumes G has at least 2 vertices

    INPUT:

    -``G``--Sage Graph

    OUTPUT:

    -Boolean

    EXAMPLES:
    sage: is_2_bootstrap_good(k4)
    True
    sage: is_2_bootstrap_good(graphs.ThomsenGraph())
    True
    sage: is_2_bootstrap_good(graphs.CycleGraph(4))
    True
    sage: is_2_bootstrap_good(graphs.CycleGraph(5))
    False
    sage: is_2_bootstrap_good(ce83)
    True
    sage: is_2_bootstrap_good(graphs.PetersenGraph())
    False
    sage: is_2_bootstrap_good(p4)
    False
    """
    return is_k_bootstrap_good(G,2)
add_to_lists(is_2_bootstrap_good, efficiently_computable_properties,all_properties)


def is_3_bootstrap_good(G):
    """
    Return whether or not G contains a subset of 3 vertices which 3-infect the whole graph.

    Assumes G has at least 3 vertices.

    INPUT:

    -``G``--Sage Graph

    OUTPUT:

    -Boolean

    EXAMPLES:
    sage: is_3_bootstrap_good(graphs.BullGraph())
    False
    sage: is_3_bootstrap_good(graphs.ThomsenGraph())
    True
    sage: is_3_bootstrap_good(graphs.CycleGraph(4))
    False
    sage: is_3_bootstrap_good(graphs.BidiakisCube())
    False
    sage: is_3_bootstrap_good(graphs.CycleGraph(5))
    False
    sage: is_3_bootstrap_good(willis_page35_fig52)
    True
    sage: is_3_bootstrap_good(ce68)
    True
    """
    return is_k_bootstrap_good(G,3)
add_to_lists(is_3_bootstrap_good,efficiently_computable_properties,all_properties)


# any localized property for an efficient property is itself efficient
def is_locally_dirac(g):
    f=localise(is_dirac)
    return f(g)
add_to_lists(is_locally_dirac,efficiently_computable_properties,all_properties)

