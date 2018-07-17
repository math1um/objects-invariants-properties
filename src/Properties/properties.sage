# GRAPH PROPERTIES

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
    for (u,v) in combinations(g.vertices(), 2):
        if len(common_neighbors(g, u, v)) != 1:
            return False
    return True

def is_distance_transitive(g):
    """
    Evaluates if graph ``g`` is distance transitive.

    A graph is distance transitive if all a,b,u,v satisfy that
    dist(a,b) = dist(u,v) implies there's an automorphism with a->u and b->v.

    EXAMPLES:

        sage: is_distance_transitive(graphs.CompleteGraph(4))
        True

        sage: is_distance_transitive(graphs.PetersenGraph())
        True

        sage: is_distance_transitive(Graph(3))
        True

        sage: is_distance_transitive(graphs.ShrikhandeGraph())
        False

    This method accepts disconnected graphs. ::

        sage: is_distance_transitive(graphs.CompleteGraph(3).disjoint_union(graphs.CompleteGraph(3)))
        True

        sage: is_distance_transitive(graphs.CompleteGraph(2).disjoint_union(Graph(2)))
        False

    Vacuous cases ::

        sage: is_distance_transitive(Graph(0))
        True

        sage: is_distance_transitive(Graph(1))
        True

        sage: is_distance_transitive(Graph(2))
        True

    ... WARNING ::

        This method calls, via the automorphism group, the Gap package. This
        package behaves badly with most threading or multiprocessing tools.
    """
    from itertools import combinations
    dist_dict = g.distance_all_pairs()
    auto_group = g.automorphism_group()

    for d in g.distances_distribution():
        sameDistPairs = []
        for (u,v) in combinations(g.vertices(), 2):
            # By default, no entry if disconnected. We substitute +Infinity.
            if dist_dict[u].get(v, +Infinity) == d:
                sameDistPairs.append(Set([u,v]))
        if len(sameDistPairs) >= 2:
            if len(sameDistPairs) != len(auto_group.orbit(sameDistPairs[0], action = "OnSets")):
                return False
    return True

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
    for i in xrange(n):
        for j in xrange(i):
            if A[i][j]==0:
                if D[i] + D[j] < n:
                    return False
    return True

def is_haggkvist_nicoghossian(g):
    """
    Evaluates if g is 2-connected and min degree >= (n + vertex_connectivity)/3.

    INPUT:

    - ``g`` -- graph

    REFERENCES:

    Theorem: If a graph ``is_haggkvist_nicoghossian``, then it is Hamiltonian.

    .. [HN1981]     \R. Häggkvist and G. Nicoghossian, "A remark on Hamiltonian
                    cycles". Journal of Combinatorial Theory, Series B, 30(1):
                    118--120, 1981.

    EXAMPLES:

        sage: is_haggkvist_nicoghossian(graphs.CompleteGraph(3))
        True

        sage: is_haggkvist_nicoghossian(graphs.CompleteGraph(5))
        True

        sage: is_haggkvist_nicoghossian(graphs.CycleGraph(5))
        False

        sage: is_haggkvist_nicoghossian(graphs.CompleteBipartiteGraph(4,3)
        False

        sage: is_haggkvist_nicoghossian(Graph(1))
        False

        sage: is_haggkvist_nicoghossian(graphs.CompleteGraph(2))
        False
    """
    k = g.vertex_connectivity()
    return k >= 2 and min(g.degree()) >= (1.0/3) * (g.order() + k)

def is_genghua_fan(g):
    """
    Evaluates if graph ``g`` satisfies a condition for Hamiltonicity by G. Fan.

    OUTPUT:

    Returns ``True`` if ``g`` is 2-connected and satisfies that
    `dist(u,v)=2` implies `\max(deg(u), deg(v)) \geq n/2` for all
    vertices `u,v`.
    Returns ``False`` otherwise.

    REFERENCES:

    Theorem: If a graph ``is_genghua_fan``, then it is Hamiltonian.

    .. [Fan1984]    Geng-Hua Fan, "New sufficient conditions for cycles in
                    graphs". Journal of Combinatorial Theory, Series B, 37(3):
                    221--227, 1984.

    EXAMPLES:

        sage: is_genghua_fan(graphs.DiamondGraph())
        True

        sage: is_genghua_fan(graphs.CycleGraph(4))
        False

        sage: is_genghua_fan(graphs.ButterflyGraph())
        False

        sage: is_genghua_fan(Graph(1))
        False
    """
    if not is_two_connected(g):
        return False
    D = g.degree()
    Dist = g.distance_all_pairs()
    V = g.vertices()
    n = g.order()
    for i in xrange(n):
        for j in xrange(i):
            if Dist[V[i]][V[j]] == 2 and max(D[i], D[j]) < n / 2.0:
                return False
    return True

def is_planar_transitive(g):
    """
    Evaluates whether graph ``g`` is planar and is vertex-transitive.

    EXAMPLES:

        sage: is_planar_transitive(graphs.HexahedralGraph())
        True

        sage: is_planar_transitive(graphs.CompleteGraph(2))
        True

        sage: is_planar_transitive(graphs.FranklinGraph())
        False

        sage: is_planar_transitive(graphs.BullGraph())
        False

    Vacuous cases ::

        sage: is_planar_transitive(Graph(1))
        True

    Sage defines `Graph(0).is_vertex_transitive() == False``. ::

        sage: is_planar_transitive(Graph(0))
        False
    """
    return g.is_planar() and g.is_vertex_transitive()

def is_generalized_dirac(g):
    """
    Test if ``graph`` g meets condition in a generalization of Dirac's Theorem.

    OUTPUT:

    Returns ``True`` if g is 2-connected and for all non-adjacent u,v,
    the cardinality of the union of neighborhood(u) and neighborhood(v)
    is `>= (2n-1)/3`.

    REFERENCES:

    Theorem: If graph g is_generalized_dirac, then it is Hamiltonian.

    .. [FGJS1989]   \R.J. Faudree, Ronald Gould, Michael Jacobson, and
                    R.H. Schelp, "Neighborhood unions and hamiltonian
                    properties in graphs". Journal of Combinatorial
                    Theory, Series B, 47(1): 1--9, 1989.

    EXAMPLES:

        sage: is_generalized_dirac(graphs.HouseGraph())
        True

        sage: is_generalized_dirac(graphs.PathGraph(5))
        False

        sage: is_generalized_dirac(graphs.DiamondGraph())
        False

        sage: is_generalized_dirac(Graph(1))
        False
    """
    from itertools import combinations

    if not is_two_connected(g):
        return False
    for (u,v) in combinations(g.vertices(), 2):
        if not g.has_edge(u,v):
            if len(neighbors_set(u, v)) < (2.0 * g.order() - 1) / 3:
                return False
    return True

def is_van_den_heuvel(g):
    """
    Evaluates if g meets an eigenvalue condition related to Hamiltonicity.

    INPUT:

    - ``g`` -- graph

    OUTPUT:

    Let ``g`` be of order `n`.
    Let `A_H` denote the adjacency matrix of a graph `H`, and `D_H` denote
    the matrix with the degrees of the vertices of `H` on the diagonal.
    Define `Q_H = D_H + A_H` and `L_H = D_H - A_H` (i.e. the Laplacian).
    Finally, let `C` be the cycle graph on `n` vertices.

    Returns ``True`` if the `i`-th eigenvalue of `L_C` is at most the `i`-th
    eigenvalue of `L_g` and the `i`-th eigenvalue of `Q_C` is at most the
    `i`-th eigenvalue of `Q_g for all `i`.

    REFERENCES:

    Theorem: If a graph is Hamiltonian, then it ``is_van_den_heuvel``.

    .. [Heu1995]    \J.van den Heuvel, "Hamilton cycles and eigenvalues of
                    graphs". Linear Algebra and its Applications, 226--228:
                    723--730, 1995.

    EXAMPLES:

        sage: is_van_den_heuvel(graphs.CycleGraph(5))
        True

        sage: is_van_den_heuvel(graphs.PetersenGraph())
        False

    TESTS::

        sage: is_van_den_heuvel(Graph(0))
        False

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
        for i in xrange(A.nrows()):
            D[i,i] = row_sums[i]
        return D + A
    cycle_q_matrix = sorted(Q(cycle_n).eigenvalues())
    g_q_matrix = sorted(Q(g).eigenvalues())
    for cycle_q_lambda_i, g_q_lambda_i in zip(cycle_q_matrix, g_q_matrix):
        if cycle_q_lambda_i > g_q_lambda_i:
            return False

    return True

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

def is_three_connected(g):
    """
    Evaluates whether graph ``g`` is 3-connected.

    A 3-connected graph is a connected graph on at least 4 vertices such that
    the removal of any two vertices still gives a connected graph.
    Follows convention that complete graph `K_n` is `n-1`-connected.

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

def is_lindquester(g):
    """
    Test if graph ``g`` meets a neighborhood union condition for Hamiltonicity.

    OUTPUT:

    Let ``g`` be of order `n`.

    Returns ``True`` if ``g`` is 2-connected and for all vertices `u,v`,
    `dist(u,v) = 2` implies that the cardinality of the union of
    neighborhood(`u`) and neighborhood(`v`) is `\geq (2n-1)/3`.
    Returns ``False`` otherwise.

    REFERENCES:

    Theorem: If a graph ``is_lindquester``, then it is Hamiltonian.

    .. [Lin1989]    \T.E. Lindquester, "The effects of distance and
                    neighborhood union conditions on hamiltonian properties
                    in graphs". Journal of Graph Theory, 13(3): 335-352,
                    1989.

    EXAMPLES:

        sage: is_lindquester(graphs.HouseGraph())
        True

        sage: is_lindquester(graphs.OctahedralGraph())
        True

        sage: is_lindquester(graphs.PathGraph(3))
        False

        sage: is_lindquester(graphs.DiamondGraph())
        False
    """
    if not is_two_connected(g):
        return False
    D = g.distance_all_pairs()
    n = g.order()
    V = g.vertices()
    for i in range(n):
        for j in range(i):
            if D[V[i]][V[j]] == 2:
                if len(neighbors_set(g,[V[i],V[j]])) < (2*n-1)/3.0:
                    return False
    return True

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

def is_p4_free(g):
    """
    Equivalent to is a cograph - https://en.wikipedia.org/wiki/Cograph
    """
    return not has_p4(g)

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

        sage: has_radius_equal_diameter(Graph(1))
        True

        sage: has_radius_equal_diameter(graphs.ClawGraph())
        False

        sage: has_radius_equal_diameter(graphs.BullGraph())
        False
    """
    return g.radius() == g.diameter()

def has_residue_equals_alpha(g):
    """
    Evaluate whether the residue of graph ``g`` equals its independence number.

    The independence number is the cardinality of the largest independent set
    of vertices in ``g``.
    The residue of a graph ``g`` with degrees `d_1 \geq d_2 \geq ... \geq d_n`
    is found iteratively. First, remove `d_1` from consideration and subtract
    `d_1` from the following `d_1` number of elements. Sort. Repeat this
    process for `d_2,d_3, ...` until only 0s remain. The number of elements,
    i.e. the number of 0s, is the residue of ``g``.

    Residue is undefined on the empty graph.

    EXAMPLES:

        sage: has_residue_equals_alpha(graphs.HouseGraph())
        True

        sage: has_residue_equals_alpha(graphs.ClawGraph())
        True

        sage: has_residue_equals_alpha(graphs.CompleteGraph(4))
        True

        sage: has_residue_equals_alpha(Graph(1))
        True

        sage: has_residue_equals_alpha(graphs.PetersenGraph())
        False

        sage: has_residue_equals_alpha(graphs.PathGraph(5))
        False
    """
    return residue(g) == independence_number(g)

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

def has_empty_KE_part(g):
    """
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
    for v in g.vertices():
        test = b.copy()
        test.delete_vertices(closed_neighborhood(b,[(v,0), (v,1)]))
        alpha_test = test.order() - test.matching(value_only=True) + 2
        if alpha_test == alpha:
            return False
    return True

def is_class1(g):
    """
    Evaluates whether the chomatic index of graph ``g`` equals its max degree.

    Let `D` be the maximum degree. By Vizing's Thoerem, all graphs can be
    edge-colored in either `D` or `D+1` colors. The case of `D` colors is
    called "class 1".

    Max degree is undefined for the empty graph.

    EXAMPLES:

        sage: is_class1(graphs.CompleteGraph(4))
        True

        sage: is_class1(graphs.WindmillGraph(4,3))
        True

        sage: is_class1(Graph(1))
        True

        sage: is_class1(graphs.CompleteGraph(3))
        False

        sage: is_class1(graphs.PetersenGraph())
        False
    """
    return g.chromatic_index() == max(g.degree())

def is_class2(g):
    """
    Evaluates whether the chomatic index of graph ``g`` equals max degree + 1.

    Let `D` be the maximum degree. By Vizing's Thoerem, all graphs can be
    edge-colored in either `D` or `D+1` colors. The case of `D+1` colors is
    called "class 2".

    Max degree is undefined for the empty graph.

    EXAMPLES:

        sage: is_class2(graphs.CompleteGraph(4))
        False

        sage: is_class2(graphs.WindmillGraph(4,3))
        False

        sage: is_class2(Graph(1))
        False

        sage: is_class2(graphs.CompleteGraph(3))
        True

        sage: is_class2(graphs.PetersenGraph())
        True
    """
    return not(g.chromatic_index() == max(g.degree()))

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

def is_anti_tutte(g):
    """
    Evalutes if graph ``g`` is connected and indep. number <= diameter + girth.

    This property is satisfied by many Hamiltonian graphs, but notably not by
    the Tutte graph ``graphs.TutteGraph()``.

    Diameter is undefined for the empty graph.

    EXAMPLES:

        sage: is_anti_tutte(graphs.CompleteBipartiteGraph(4, 5))
        True

        sage: is_anti_tutte(graphs.PetersenGraph())
        True

        sage: is_anti_tutte(Graph(1))

        sage: is_anti_tutte(graphs.TutteGraph())
        False

        sage: is_anti_tutte(graphs.TutteCoxeterGraph())
        False
    """
    if not g.is_connected():
        return False
    return independence_number(g) <= g.diameter() + g.girth()

def is_anti_tutte2(g):
    """
    Tests if graph ``g`` has indep. number <= domination number + radius - 1.

    ``g`` must also be connected.
    This property is satisfied by many Hamiltonian graphs, but notably not by
    the Tutte graph ``graphs.TutteGraph()``.

    Radius is undefined for the empty graph.

    EXAMPLES:

        sage: is_anti_tutte2(graphs.CompleteGraph(5))
        True

        sage: is_anti_tutte2(graphs.PetersenGraph())
        True

        sage: is_anti_tutte2(graphs.TutteGraph())
        False

        sage: is_anti_tutte2(graphs.TutteCoxeterGraph())
        False

        sage: is_anti_tutte2(Graph(1))
        False
    """
    if not g.is_connected():
        return False
    return independence_number(g) <=  domination_number(g) + g.radius()- 1

def diameter_equals_twice_radius(g):
    """
    Evaluates whether the diameter of graph ``g`` is equal to twice its radius.

    Diameter and radius are undefined for the empty graph.

    EXAMPLES:

        sage: has_radius_equal_diameter(graphs.ClawGraph())
        True

        sage: has_radius_equal_diameter(graphs.KrackhardtKiteGraph())
        True

        sage: diameter_equals_twice_radius(graphs.HouseGraph())
        False

        sage: has_radius_equal_diameter(graphs.BullGraph())
        False

    The radius and diameter of ``Graph(1)`` are both 1. ::

        sage: diameter_equals_twice_radius(Graph(1))
        True

    Disconnected graphs have both diameter and radius equal infinity.

        sage: diameter_equals_twice_radius(Graph(4))
        True
    """
    return g.diameter() == 2*g.radius()

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

def has_lovasz_theta_equals_alpha(g):
    """
    Tests if the Lovasz number of graph ``g`` equals its independence number.

    Examples:

        sage: has_lovasz_theta_equals_alpha(graphs.CompleteGraph(12))
        True

        sage: has_lovasz_theta_equals_alpha(double_fork)
        True

        sage: has_lovasz_theta_equals_alpha(graphs.PetersenGraph())
        True

        sage: has_lovasz_theta_equals_alpha(graphs.ClebschGraph())
        False

        sage: has_lovasz_theta_equals_alpha(graphs.CycleGraph(24))
        False

    True for all graphs with no edges ::

        sage: has_lovasz_theta_equals_alpha(graph(12))
        True

    Edge cases ::

        sage: has_lovasz_theta_equals_alpha(Graph(0))
        True

        sage: has_lovasz_theta_equals_alpha(Graph(1))
        True
    """
    return g.lovasz_theta() == independence_number(g):

def has_lovasz_theta_equals_cc(g):
    if g.lovasz_theta() == clique_covering_number(g):
        return True
    else:
        return False

#sufficient condition for hamiltonicity
def is_chvatal_erdos(g):
    return independence_number(g) <= g.vertex_connectivity()


#matching_covered if every edge is in a maximum matching (generalization of factor-covered which requires perfect matching)
def matching_covered(g):
    g = g.copy()
    nu = matching_number(g)
    E = g.edges()
    for e in E:
        g.delete_edge(e)
        nu2 = matching_number(g)
        if nu != nu2:
            return False
        g.add_edge(e)
    return True

#a property that separates tutte from known hamiltonian examples, must be connected
#radius(tutte) > center(tutte)
def radius_greater_than_center(g):
    if not g.is_connected() or g.radius() <= card_center(g):
        return False
    else:
        return True

#a property that separates tutte from known hamiltonian examples, must be connected
#avg_dist(tutte) > girth(tutte)
def avg_distance_greater_than_girth(g):
    if not g.is_connected() or g.average_distance() <= g.girth():
        return False
    else:
        return True

#chromatic number equals min of known chi upper bounds
def chi_equals_min_theory(g):
    chromatic_upper_theory = [brooks, wilf, welsh_powell, szekeres_wilf]
    min_theory = min([f(g) for f in chromatic_upper_theory])
    chi = g.chromatic_number()
    if min_theory == chi:
        return True
    else:
        return False

def is_heliotropic_plant(g):
    return (independence_number(g) == card_positive_eigenvalues(g))

def is_geotropic_plant(g):
    return (independence_number(g) == card_negative_eigenvalues(g))

#means has hamiltonian path, true iff g join a single vertex has hamiltonian cycle
def is_traceable(g):
     gadd = g.join(Graph(1),labels="integers")
     return gadd.is_hamiltonian()

def has_residue_equals_two(g):
    return residue(g) == 2

#a necessary condition for being even hole free
def is_chordal_or_not_perfect(g):
    if g.is_chordal():
        return true
    else:
        return not g.is_perfect()

def has_alpha_residue_equal_two(g):
    if residue(g) != 2:
        return false
    else:
        return independence_number(g) == 2

    # for vizing's independence number conjecture
def alpha_leq_order_over_two(g):
    return (2*independence_number(g) <= g.order())


#in a luo-zhao sufficient condition for alpha <= n/2 (vizing's ind number conj)
def order_leq_twice_max_degree(g):
    return (g.order() <= 2*max(g.degree()))

#not in properties list until meredith graph is computed
#critical if connected, class 2, and removal of any edge decreases chromatic number
def is_chromatic_index_critical(g):
    if not g.is_connected():
        return False
    Delta = max(g.degree())
    chi_e = g.chromatic_index()
    if chi_e != Delta + 1:
        return False

    lg=g.line_graph()
    equiv_lines = lg.automorphism_group(return_group=False,orbits=true)
    equiv_lines_representatives = [orb[0] for orb in equiv_lines]

    for e in equiv_lines_representatives:
        gc = copy(g)
        gc.delete_edge(e)
        chi_e_prime = gc.chromatic_index()
        if not chi_e_prime < chi_e:
            return False
    return True

#not in properties list
#alpha(g-e) > alpha(g) for *every* edge g
def is_alpha_critical(g):
    #if not g.is_connected():
        #return False
    alpha = independence_number(g)
    for e in g.edges():
        gc = copy(g)
        gc.delete_edge(e)
        alpha_prime = independence_number(gc)
        if alpha_prime <= alpha:
            return False
    return True

#graph is KE if matching number + independence number = n, test does *not* compute alpha
def is_KE(g):
    return g.order() == len(find_KE_part(g))

#graph is KE if matching number + independence number = n, test comoutes alpha
#def is_KE(g):
#    return (g.matching(value_only = True) + independence_number(g) == g.order())

#possibly faster version of is_KE (not currently in invariants)
#def is_KE2(g):
#    return (independence_number(g) == critical_independence_number(g))

def is_independence_irreducible(g):
    return g.order() == card_independence_irreducible_part(g)


def is_factor_critical(g):
    """
    a graph is factor-critical if order is odd and removal of any vertex gives graph with perfect matching
        is_factor_critical(graphs.PathGraph(3))
        False
        sage: is_factor_critical(graphs.CycleGraph(5))
        True
    """
    if g.order() % 2 == 0:
        return False
    for v in g.vertices():
        gc = copy(g)
        gc.delete_vertex(v)
        if not gc.has_perfect_matching:
            return False
    return True

#returns a list of (necessarily non-adjacent) vertices that have the same neighbors as v if a pair exists or None
def find_twins_of_vertex(g,v):
    L = []
    V = g.vertices()
    D = g.distance_all_pairs()
    for i in range(g.order()):
        w = V[i]
        if  D[v][w] == 2 and g.neighbors(v) == g.neighbors(w):
                L.append(w)
    return L

def has_twin(g):
    t = find_twin(g)
    if t == None:
        return False
    else:
        return True

def is_twin_free(g):
    return not has_twin(g)

#returns twin pair (v,w) if one exists, else returns None
#can't be adjacent
def find_twin(g):

    V = g.vertices()
    for v in V:
        Nv = set(g.neighbors(v))
        for w in V:
            Nw = set(g.neighbors(w))
            if v not in Nw and Nv == Nw:
                return (v,w)
    return None

def find_neighbor_twins(g):
    V = g.vertices()
    for v in V:
        Nv = g.neighbors(v)
        for w in Nv:
            if set(closed_neighborhood(g,v)) == set(closed_neighborhood(g,w)):
                return (v,w)
    return None

#given graph g and subset S, looks for any neighbor twin of any vertex in T
#if result = T, then no twins, else the result is maximal, but not necessarily unique
def find_neighbor_twin(g, T):
    gT = g.subgraph(T)
    for v in T:
        condition = False
        Nv = set(g.neighbors(v))
        #print "v = {}, Nv = {}".format(v,Nv)
        NvT = set(gT.neighbors(v))
        for w in Nv:
            NwT = set(g.neighbors(w)).intersection(set(T))
            if w not in T and NvT.issubset(NwT):
                T.append(w)
                condition = True
                #print "TWINS: v = {}, w = {}, sp3 = {}".format(v,w,sp3)
                break
        if condition == True:
            break

#if result = T, then no twins, else the result is maximal, but not necessarily unique
def iterative_neighbor_twins(g, T):
    T2 = copy(T)
    find_neighbor_twin(g, T)
    while T2 != T:
        T2 = copy(T)
        find_neighbor_twin(g, T)
    return T


#can't compute membership in this class directly. instead testing isomorhism for 400 known class0 graphs
def is_pebbling_class0(g):
    for hkey in class0graphs_dict:
        h = Graph(class0graphs_dict[hkey])
        if g.is_isomorphic(h):
            return True
    return False

def girth_greater_than_2log(g):
    return bool(g.girth() > 2*log(g.order(),2))

def szekeres_wilf_equals_chromatic_number(g):
    return szekeres_wilf(g) == g.chromatic_number()

def has_Havel_Hakimi_property(g, v):
    """
    This function returns whether the vertex v in the graph g has the Havel-Hakimi
    property as defined in [1]. A vertex has the Havel-Hakimi property if it has
    maximum degree and the minimum degree of its neighbours is at least the maximum
    degree of its non-neigbours.

    [1] Graphs with the strong Havel-Hakimi property, M. Barrus, G. Molnar, Graphs
        and Combinatorics, 2016, http://dx.doi.org/10.1007/s00373-015-1674-7

    Every vertex in a regular graph has the Havel-Hakimi property::

        sage: P = graphs.PetersenGraph()
        sage: for v in range(10):
        ....:     has_Havel_Hakimi_property(P,v)
        True
        True
        True
        True
        True
        True
        True
        True
        True
        True
        sage: has_Havel_Hakimi_property(Graph([[0,1,2,3],lambda x,y: False]),0)
        True
        sage: has_Havel_Hakimi_property(graphs.CompleteGraph(5),0)
        True
    """
    if max_degree(g) > g.degree(v): return False

    #handle the case where the graph is an independent set
    if len(g.neighbors(v)) == 0: return True

    #handle the case where v is adjacent with all vertices
    if len(g.neighbors(v)) == len(g.vertices()) - 1: return True

    return (min(g.degree(nv) for nv in g.neighbors(v)) >=
        max(g.degree(nnv) for nnv in g.vertices() if nnv != v and nnv not in g.neighbors(v)))

def has_strong_Havel_Hakimi_property(g):
    """
    This function returns whether the graph g has the strong Havel-Hakimi property
    as defined in [1]. A graph has the strong Havel-Hakimi property if in every
    induced subgraph H of G, every vertex of maximum degree has the Havel-Hakimi
    property.

    [1] Graphs with the strong Havel-Hakimi property, M. Barrus, G. Molnar, Graphs
        and Combinatorics, 2016, http://dx.doi.org/10.1007/s00373-015-1674-7

    The graph obtained by connecting two cycles of length 3 by a single edge has
    the strong Havel-Hakimi property::

        sage: has_strong_Havel_Hakimi_property(Graph('E{CW'))
        True
    """
    for S in Subsets(g.vertices()):
        if len(S)>2:
            H = g.subgraph(S)
            Delta = max_degree(H)
            if any(not has_Havel_Hakimi_property(H, v) for v in S if H.degree(v) == Delta):
                return False
    return True

# Graph is subcubic is each vertex is at most degree 3
def is_subcubic(g):
    return max_degree(g) <= 3

# Max and min degree varies by at most 1
def is_quasi_regular(g):
    if max_degree(g) - min_degree(g) < 2:
        return true
    return false

# g is bad if a block is isomorphic to k3, c5, k4*, c5*
def is_bad(g):
    blocks = g.blocks_and_cut_vertices()[0]
    # To make a subgraph of g from the ith block
    for i in blocks:
        h = g.subgraph(i)
        boolean = h.is_isomorphic(alpha_critical_easy[1]) or h.is_isomorphic(alpha_critical_easy[4]) or h.is_isomorphic(alpha_critical_easy[5]) or h.is_isomorphic(alpha_critical_easy[21])
        if boolean == True:
            return True
    return False

# Graph g is complement_hamiltonian if the complement of the graph is hamiltonian.
def is_complement_hamiltonian(g):
    return g.complement().is_hamiltonian()

# A graph is unicyclic if it is connected and has order == size
# Equivalently, graph is connected and has exactly one cycle
def is_unicyclic(g):
    """
    Tests:
        sage: is_unicyclic(graphs.BullGraph())
        True
        sage: is_unicyclic(graphs.ButterflyGraph())
        False
    """
    return g.is_connected() and g.order() == g.size()

def is_k_tough(g,k):
    return toughness(g) >= k # In invariants
def is_1_tough(g):
    return is_k_tough(g, 1)
def is_2_tough(g):
    return is_k_tough(g, 2)

# True if graph has at least two hamiltonian cycles. The cycles may share some edges.
def has_two_ham_cycles(gIn):
    g = gIn.copy()
    g.relabel()
    try:
        ham1 = g.hamiltonian_cycle()
    except EmptySetError:
        return False

    for e in ham1.edges():
        h = copy(g)
        h.delete_edge(e)
        if h.is_hamiltonian():
            return true
    return false

def has_simplical_vertex(g):
    """
    v is a simplical vertex if induced neighborhood is a clique.
    """
    for v in g.vertices():
        if is_simplical_vertex(g, v):
            return true
    return false

def has_exactly_two_simplical_vertices(g):
    """
    v is a simplical vertex if induced neighborhood is a clique.
    """
    return simplical_vertices(g) == 2

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
    degree_two_vertices = (v for v in g.vertices() if g.degree(v) == 2)
    try:
        v = next(degree_two_vertices)
    except StopIteration: # Empty list. No degree 2 vertices.
        return False

    if not g.has_edge(g.neighbors(v)): # Clique
        return false
    g2 = g.copy()
    g2.delete_vertex(v)
    return is_two_tree(g2)

def is_two_path(g):
    """
    Graph g is a two_path if it is a two_tree and has exactly 2 simplical vertices
    """
    return has_exactly_two_simplical_vertices(g) and is_two_tree(g)

def is_prism_hamiltonian(g):
    """
    A graph G is prism hamiltonian if G x K2 (cartesian product) is hamiltonian
    """
    return g.cartesian_product(graphs.CompleteGraph(2)).is_hamiltonian()

# Bauer, Douglas, et al. "Long cycles in graphs with large degree sums." Discrete Mathematics 79.1 (1990): 59-70.
def is_bauer(g):
    """
    True if g is 2_tough and sigma_3 >= order
    """
    return is_2_tough(g) and sigma_k(g, 3) >= g.order()

# Jung, H. A. "On maximal circuits in finite graphs." Annals of Discrete Mathematics. Vol. 3. Elsevier, 1978. 129-144.
def is_jung(g):
    """
    True if graph has n >= 11, if graph is 1-tough, and sigma_2 >= n - 4.
    See functions toughness(g) and sigma_2(g) for more details.
    """
    return g.order() >= 11 and is_1_tough(g) and sigma_2(g) >= g.order() - 4

# Bela Bollobas and Andrew Thomason, Weakly Pancyclic Graphs. Journal of Combinatorial Theory 77: 121--137, 1999.
def is_weakly_pancyclic(g):
    """
    True if g contains cycles of every length k from k = girth to k = circumfrence

    Returns False if g is acyclic (in which case girth = circumfrence = +Infinity).

    sage: is_weakly_pancyclic(graphs.CompleteGraph(6))
    True
    sage: is_weakly_pancyclic(graphs.PetersenGraph())
    False
    """
    lengths = cycle_lengths(g)
    if not lengths: # acyclic
        raise ValueError("Graph is acyclic. Property undefined.")
    else:
        return lengths == set(range(min(lengths),max(lengths)+1))

def is_pancyclic(g):
    """
    True if g contains cycles of every length from 3 to g.order()

    sage: is_pancyclic(graphs.OctahedralGraph())
    True
    sage: is_pancyclic(graphs.CycleGraph(10))
    False
    """
    lengths = cycle_lengths(g)
    return lengths == set(range(3, g.order()+1))

def has_two_walk(g):
    """
    A two-walk is a closed walk that visits every vertex and visits no vertex more than twice.

    Two-walk is a generalization of Hamiltonian cycles. If a graph is Hamiltonian, then it has a two-walk.

    sage: has_two_walk(c4c4)
    True
    sage: has_two_walk(graphs.WindmillGraph(3,3))
    False
    """
    for init_vertex in g.vertices():
        path_stack = [[init_vertex]]
        while path_stack:
            path = path_stack.pop()
            for neighbor in g.neighbors(path[-1]):
                if neighbor == path[0] and all(v in path for v in g.vertices()):
                    return True
                elif path.count(neighbor) < 2:
                    path_stack.append(path + [neighbor])
    return False

def is_claw_free_paw_free(g):
    return is_claw_free(g) and is_paw_free(g)

def has_bull(g):
    """
    True if g has an induced subgraph isomorphic to graphs.BullGraph()
    """
    return g.subgraph_search(graphs.BullGraph(), induced = True) != None

def is_bull_free(g):
    """
    True if g does not have an induced subgraph isomoprhic to graphs.BullGraph()
    """
    return not has_bull(g)

def is_claw_free_bull_free(g):
    return is_claw_free(g) and is_bull_free(g)

def has_F(g):
    """
    Let F be a triangle with 3 pendants. True if g has an induced F.
    """
    F = graphs.CycleGraph(3)
    F.add_vertices([3,4,5])
    F.add_edges([(0,3), [1,4], [2,5]])
    return g.subgraph_search(F, induced = True) != None

def is_F_free(g):
    """
    Let F be a triangle with 3 pendants. True if g has no induced F.
    """
    return not has_F(g)

# Ronald Gould, Updating the Hamiltonian problem — a survey. Journal of Graph Theory 15.2: 121-157, 1991.
def is_oberly_sumner(g):
    """
    g is_oberly_sumner if order >= 3, is_two_connected, is_claw_free, AND is_F_free
    """
    return g.order() >= 3 and is_two_connected(g) and is_claw_free(g) and is_F_free(g)
def is_oberly_sumner_bull(g):
    """
    True if g is 2-connected, claw-free, and bull-free
    """
    return is_two_connected(g) and is_claw_free_bull_free(g)
def is_oberly_sumner_p4(g):
    """
    True if g is 2-connected, claw-free, and p4-free
    """
    return is_two_connected(g) and is_claw_free(g) and is_p4_free(g)

# Ronald Gould, Updating the Hamiltonian problem — a survey. Journal of Graph Theory 15.2: 121-157, 1991.
def is_matthews_sumner(g):
    """
    True if g is 2-connected, claw-free, and minimum-degree >= (order-1) / 3
    """
    return is_two_connected(g) and is_claw_free(g) and min_degree(g) >= (g.order() - 1) / 3
def is_broersma_veldman_gould(g):
    """
    True if g is 2-connected, claw-free, and diameter <= 2
    """
    return is_two_connected(g) and is_claw_free(g) and g.diameter() <= 2

def chvatals_condition(g):
    """
    True if g.order()>=3 and given increasing degrees d_1,..,d_n, for all i, i>=n/2 or d_i>i or d_{n-i}>=n-1

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

def is_matching(g):
    """
    Returns True if this graph is the disjoint union of complete graphs of order 2.

    Tests:
        sage: is_matching(graphs.CompleteGraph(2))
        True
        sage: is_matching(graphs.PathGraph(4))
        False
        sage: is_matching(graphs.CompleteGraph(2).disjoint_union(graphs.CompleteGraph(2)))
        True
    """
    return min(g.degree())==1 and max(g.degree())==1

def has_odd_order(g):
    """
    True if the number of vertices in g is odd

    sage: has_odd_order(Graph(5))
    True
    sage: has_odd_order(Graph(2))
    False
    """
    return g.order() % 2 == 1

def has_even_order(g):
    """
    True if the number of vertices in g is even

    sage: has_even_order(Graph(5))
    False
    sage: has_even_order(Graph(2))
    True
    """
    return g.order() % 2 == 0

def is_locally_two_connected(g):
    """

    ALGORITHM:

    We modify the algorithm from our ``localise`` factory method to stop at
    subgraphs of 2 vertices, since ``is_two_connected`` is undefined on smaller
    subgraphs.
    """
    return all((f(g.subgraph(g.neighbors(v))) if len(g.neighbors(v)) >= 2
                                              else True) for v in g.vertices())

######################################################################################################################
#Below are some factory methods which create properties based on invariants or other properties

def has_equal_invariants(invar1, invar2, name=None):
    """
    This function takes two invariants as an argument and returns the property that these invariants are equal.
    Optionally a name for the new function can be provided as a third argument.
    """
    def equality_checker(g):
        return invar1(g) == invar2(g)

    if name is not None:
        equality_checker.__name__ = name
    elif hasattr(invar1, '__name__') and hasattr(invar2, '__name__'):
        equality_checker.__name__ = 'has_{}_equals_{}'.format(invar1.__name__, invar2.__name__)
    else:
        raise ValueError('Please provide a name for the new function')

    return equality_checker

"""
    sage: has_alpha_equals_clique_covering(graphs.CycleGraph(5))
    False
"""
has_alpha_equals_clique_covering = has_equal_invariants(independence_number, clique_covering_number, name="has_alpha_equals_clique_covering")


def has_invariant_equal_to(invar, value, name=None, documentation=None):
    """
    This function takes an invariant and a value as arguments and returns the property
    that the invariant value for a graph is equal to the provided value.

    Optionally a name and documentation for the new function can be provided.

    sage: order_is_five = has_invariant_equal_to(Graph.order, 5)
    sage: order_is_five(graphs.CycleGraph(5))
    True
    sage: order_is_five(graphs.CycleGraph(6))
    False
    """
    def equality_checker(g):
        return invar(g) == value

    if name is not None:
        equality_checker.__name__ = name
    elif hasattr(invar, '__name__'):
        equality_checker.__name__ = 'has_{}_equal_to_{}'.format(invar.__name__, value)
    else:
        raise ValueError('Please provide a name for the new function')

    equality_checker.__doc__ = documentation

    return equality_checker

def has_leq_invariants(invar1, invar2, name=None):
    """
    This function takes two invariants as an argument and returns the property that the first invariant is
    less than or equal to the second invariant.
    Optionally a name for the new function can be provided as a third argument.
    """
    def comparator(g):
        return invar1(g) <= invar2(g)

    if name is not None:
        comparator.__name__ = name
    elif hasattr(invar1, '__name__') and hasattr(invar2, '__name__'):
        comparator.__name__ = 'has_{}_leq_{}'.format(invar1.__name__, invar2.__name__)
    else:
        raise ValueError('Please provide a name for the new function')

    return comparator

#add all properties derived from pairs of invariants
invariant_relation_properties = [has_leq_invariants(f,g) for f in all_invariants for g in all_invariants if f != g]


def localise(f, name=None, documentation=None):
    """
    This function takes a property (i.e., a function taking only a graph as an argument) and
    returns the local variant of that property. The local variant is True if the property is
    True for the neighbourhood of each vertex and False otherwise.
    """
    #create a local version of f
    def localised_function(g):
        return all((f(g.subgraph(g.neighbors(v))) if g.neighbors(v) else True) for v in g.vertices())

    #we set a nice name for the new function
    if name is not None:
        localised_function.__name__ = name
    elif hasattr(f, '__name__'):
        if f.__name__.startswith('is_'):
            localised_function.__name__ = 'is_locally' + f.__name__[2:]
        elif f.__name__.startswith('has_'):
            localised_function.__name__ = 'has_locally' + f.__name__[2:]
        else:
            localised_function.__name__ = 'localised_' + f.__name__
    else:
        raise ValueError('Please provide a name for the new function')

    localised_function.__doc__ = documentation

    return localised_function

is_locally_dirac = localise(is_dirac)
is_locally_bipartite = localise(Graph.is_bipartite)
is_locally_planar = localise(Graph.is_planar, documentation="True if the open neighborhood of each vertex v is planar")
"""
Tests:
    sage: is_locally_unicyclic(graphs.OctahedralGraph())
    True
    sage: is_locally_unicyclic(graphs.BullGraph())
    False
"""
is_locally_unicyclic = localise(is_unicyclic, documentation="""A graph is locally unicyclic if all its local subgraphs are unicyclic.

Tests:
    sage: is_locally_unicyclic(graphs.OctahedralGraph())
    True
    sage: is_locally_unicyclic(graphs.BullGraph())
    False
""")
is_locally_connected = localise(Graph.is_connected, documentation="True if the neighborhood of every vertex is connected (stronger than claw-free)")
"""
sage: is_local_matching(graphs.CompleteGraph(3))
True
sage: is_local_matching(graphs.CompleteGraph(4))
False
sage: is_local_matching(graphs.FriendshipGraph(5))
True
"""
is_local_matching = localise(is_matching, name="is_local_matching", documentation="""True if the neighborhood of each vertex consists of independent edges.

Tests:
    sage: is_local_matching(graphs.CompleteGraph(3))
    True
    sage: is_local_matching(graphs.CompleteGraph(4))
    False
    sage: is_local_matching(graphs.FriendshipGraph(5))
    True
""")

######################################################################################################################

efficiently_computable_properties = [Graph.is_regular, Graph.is_planar,
Graph.is_forest, Graph.is_eulerian, Graph.is_connected, Graph.is_clique,
Graph.is_circular_planar, Graph.is_chordal, Graph.is_bipartite,
Graph.is_cartesian_product,Graph.is_distance_regular,  Graph.is_even_hole_free,
Graph.is_gallai_tree, Graph.is_line_graph, Graph.is_overfull, Graph.is_perfect,
Graph.is_split, Graph.is_strongly_regular, Graph.is_triangle_free,
Graph.is_weakly_chordal, is_dirac, is_ore,
is_generalized_dirac, is_van_den_heuvel, is_two_connected, is_three_connected,
is_lindquester, is_claw_free, Graph.has_perfect_matching, has_radius_equal_diameter,
is_not_forest, is_genghua_fan, is_cubic, diameter_equals_twice_radius,
is_locally_connected, matching_covered, is_locally_dirac,
is_locally_bipartite, is_locally_two_connected, Graph.is_interval, has_paw,
is_paw_free, has_p4, is_p4_free, has_dart, is_dart_free, has_kite, is_kite_free,
has_H, is_H_free, has_residue_equals_two, order_leq_twice_max_degree,
alpha_leq_order_over_two, is_factor_critical, is_independence_irreducible,
has_twin, is_twin_free, diameter_equals_two, girth_greater_than_2log, Graph.is_cycle,
pairs_have_unique_common_neighbor, has_star_center, is_complement_of_chordal,
has_c4, is_c4_free, is_subcubic, is_quasi_regular, is_bad, has_k4, is_k4_free,
is_distance_transitive, is_unicyclic, is_locally_unicyclic, has_simplical_vertex,
has_exactly_two_simplical_vertices, is_two_tree, is_locally_planar,
is_four_connected, is_claw_free_paw_free, has_bull, is_bull_free,
is_claw_free_bull_free, has_F, is_F_free, is_oberly_sumner, is_oberly_sumner_bull,
is_oberly_sumner_p4, is_matthews_sumner, chvatals_condition, is_matching, is_local_matching,
has_odd_order, has_even_order, Graph.is_circulant, Graph.has_loops,
Graph.is_asteroidal_triple_free, Graph.is_block_graph, Graph.is_cactus,
Graph.is_cograph, Graph.is_long_antihole_free, Graph.is_long_hole_free, Graph.is_partial_cube,
Graph.is_polyhedral, Graph.is_prime, Graph.is_tree, Graph.is_apex, Graph.is_arc_transitive,
Graph.is_self_complementary, is_double_clique, has_fork, is_fork_free,
has_empty_KE_part]

intractable_properties = [Graph.is_hamiltonian, Graph.is_vertex_transitive,
Graph.is_edge_transitive, has_residue_equals_alpha, Graph.is_odd_hole_free,
Graph.is_semi_symmetric, is_planar_transitive, is_class1,
is_class2, is_anti_tutte, is_anti_tutte2, has_lovasz_theta_equals_cc,
has_lovasz_theta_equals_alpha, is_chvatal_erdos, is_heliotropic_plant,
is_geotropic_plant, is_traceable, is_chordal_or_not_perfect,
has_alpha_residue_equal_two, is_complement_hamiltonian, is_1_tough, is_2_tough,
has_two_ham_cycles, is_two_path, is_prism_hamiltonian, is_bauer, is_jung,
is_weakly_pancyclic, is_pancyclic, has_two_walk, has_alpha_equals_clique_covering,
Graph.is_transitively_reduced, Graph.is_half_transitive, Graph.is_line_graph,
is_haggkvist_nicoghossian]

removed_properties = [is_pebbling_class0]

"""
    Last version of graphs packaged checked: Sage 8.2
    This means checked for new functions, and for any errors/changes in old functions!
    sage: sage.misc.banner.version_dict()['major'] < 8 or (sage.misc.banner.version_dict()['major'] == 8 and sage.misc.banner.version_dict()['minor'] <= 2)
    True

    Skip Graph.is_circumscribable() and Graph.is_inscribable() because they
        throw errors for the vast majority of our graphs.
    Skip Graph.is_biconnected() in favor of our is_two_connected(), because we
        prefer our name, and because we disagree with their definition on K2.
        We define that K2 is NOT 2-connected, it is n-1 = 1 connected.
    Implementation of Graph.is_line_graph() is intractable, despite a theoretically efficient algorithm existing.
"""
sage_properties = [Graph.is_hamiltonian, Graph.is_eulerian, Graph.is_planar,
Graph.is_circular_planar, Graph.is_regular, Graph.is_chordal, Graph.is_circulant,
Graph.is_interval, Graph.is_gallai_tree, Graph.is_clique, Graph.is_cycle,
Graph.is_transitively_reduced, Graph.is_self_complementary, Graph.is_connected,
Graph.has_loops, Graph.is_asteroidal_triple_free, Graph.is_bipartite,
Graph.is_block_graph, Graph.is_cactus, Graph.is_cartesian_product,
Graph.is_cograph, Graph.is_distance_regular, Graph.is_edge_transitive, Graph.is_even_hole_free,
Graph.is_forest, Graph.is_half_transitive, Graph.is_line_graph,
Graph.is_long_antihole_free, Graph.is_long_hole_free, Graph.is_odd_hole_free,
Graph.is_overfull, Graph.is_partial_cube, Graph.is_polyhedral, Graph.is_prime,
Graph.is_semi_symmetric, Graph.is_split, Graph.is_strongly_regular, Graph.is_tree,
Graph.is_triangle_free, Graph.is_weakly_chordal, Graph.has_perfect_matching, Graph.is_apex,
Graph.is_arc_transitive]

#speed notes
#FAST ENOUGH (tested for graphs on 140921): is_hamiltonian, is_vertex_transitive,
#    is_edge_transitive, has_residue_equals_alpha, is_odd_hole_free, is_semi_symmetric,
#    is_line_graph, is_line_graph, is_anti_tutte, is_planar_transitive
#SLOW but FIXED for SpecialGraphs: is_class1, is_class2

properties = efficiently_computable_properties + intractable_properties
properties_plus = efficiently_computable_properties + intractable_properties + invariant_relation_properties


invariants_from_properties = [make_invariant_from_property(property) for property in properties]
invariants_plus = all_invariants + invariants_from_properties

# weakly_chordal = weakly chordal, i.e., the graph and its complement have no induced cycle of length at least 5
