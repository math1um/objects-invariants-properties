intractable_properties = []
all_properties = []

# Syntax: after each defined property, we need:
# add_to_lists(just_defined_property_name, list1, list2, etc)

# speed notes
# FAST ENOUGH (tested for graphs on 140921): is_hamiltonian, is_vertex_transitive,
#    is_edge_transitive, has_residue_equals_alpha, is_odd_hole_free, is_semi_symmetric,
#    is_line_graph, is_line_graph, is_anti_tutte, is_planar_transitive
# SLOW but FIXED for SpecialGraphs: is_class1, is_class2


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
        for (u,v) in combinations(g.vertices(sort=true), 2):
            # By default, no entry if disconnected. We substitute +Infinity.
            if dist_dict[u].get(v, +Infinity) == d:
                sameDistPairs.append(Set([u,v]))
        if len(sameDistPairs) >= 2:
            if len(sameDistPairs) != len(auto_group.orbit(sameDistPairs[0], action = "OnSets")):
                return False
    return True
add_to_lists(is_distance_transitive, intractable_properties, all_properties)


def is_planar_transitive(g):
    """
    Evaluates whether graph ``g`` is planar and is vertex-transitive.

    INPUT:

    -``g``-- Sage Graph

    OUTPUT:

    -Boolean

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
add_to_lists(is_planar_transitive, intractable_properties, all_properties)


def has_residue_equals_alpha(g):
    r"""
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
add_to_lists(has_residue_equals_alpha, intractable_properties, all_properties)


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
add_to_lists(is_class1, intractable_properties, all_properties)


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
add_to_lists(is_class2, intractable_properties, all_properties)


def is_anti_tutte(g):
    """
    Evaluates if graph ``g`` is connected and independence number <= diameter + girth.

    This property is satisfied by many Hamiltonian graphs, but notably not by
    the Tutte graph ``graphs.TutteGraph()``.

    Diameter is undefined for the empty graph.

    EXAMPLES:

        sage: is_anti_tutte(graphs.CompleteBipartiteGraph(4, 5))
        True

        sage: is_anti_tutte(graphs.PetersenGraph())
        True

        sage: is_anti_tutte(Graph(1))
        True

        sage: is_anti_tutte(graphs.TutteGraph())
        False

        sage: is_anti_tutte(graphs.TutteCoxeterGraph())
        False
    """
    if not g.is_connected():
        return False
    return independence_number(g) <= g.diameter() + g.girth()
add_to_lists(is_anti_tutte,  intractable_properties, all_properties)


def is_anti_tutte2(g):
    """
    Tests if graph ``g`` has independence number <= domination number + radius - 1.

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
    """
    if not g.is_connected():
        return False
    return independence_number(g) <=  domination_number(g) + g.radius()- 1
add_to_lists(is_anti_tutte2, intractable_properties, all_properties)


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
        True

    True for all graphs with no edges ::

        sage: has_lovasz_theta_equals_alpha(Graph(12))
        True

    Edge cases ::

        sage: has_lovasz_theta_equals_alpha(Graph(0))
        True

        # Broken. Issue #584
        sage: has_lovasz_theta_equals_alpha(Graph(1)) # doctest: +SKIP
        True
    """
    return lovasz_theta(g) == independence_number(g)
add_to_lists(has_lovasz_theta_equals_alpha, intractable_properties, all_properties)


def has_lovasz_theta_equals_cc(g):
    """
    Test if the Lovasz number of graph ``g`` equals its clique covering number.

    Examples:

        sage: has_lovasz_theta_equals_cc(graphs.CompleteGraph(12))
        True

        sage: has_lovasz_theta_equals_cc(double_fork)
        True

        sage: has_lovasz_theta_equals_cc(graphs.PetersenGraph())
        False

        sage: has_lovasz_theta_equals_cc(Graph(12))
        True

        sage: has_lovasz_theta_equals_cc(graphs.ClebschGraph())
        False

        has_lovasz_theta_equals_alpha(graphs.BuckyBall())
        False

    Edge cases ::

        sage: has_lovasz_theta_equals_cc(Graph(0))
        True

        # Broken. Issue #584
        sage: has_lovasz_theta_equals_cc(Graph(1)) # doctest: +SKIP
        True
    """
    return lovasz_theta(g) == clique_covering_number(g)
add_to_lists(has_lovasz_theta_equals_cc, intractable_properties, all_properties)


def is_chvatal_erdos(g):
    r"""
    Evaluates whether graph ``g`` meets a Hamiltonicity condition of [CV1972]_.

    OUTPUT:

    Returns ``True`` if the independence number of ``g`` is less than or equal
    to the vertex connectivity of ``g``.
    Returns ``False`` otherwise.

    EXAMPLES:

        sage: is_chvatal_erdos(graphs.CompleteGraph(5))
        True

        sage: is_chvatal_erdos(graphs.CycleGraph(5))
        True

        sage: is_chvatal_erdos(graphs.CompleteGraph(2))
        True

        sage: is_chvatal_erdos(graphs.PetersenGraph())
        False

        sage: is_chvatal_erdos(graphs.ClawGraph())
        False

        sage: is_chvatal_erdos(graphs.DodecahedralGraph())
        False

    Edge cases ::

        sage: is_chvatal_erdos(Graph(1))
        False

        sage: is_chvatal_erdos(Graph(0))
        True

    REFERENCES:

    Theorem: If a graph ``is_chvatal_erdos``, then it is Hamiltonian.

    .. [CV1972]     \V. Chvatal and P. Erdos, "A note on hamiltonian circuits".
                    Discrete Mathematics, 2(2): 111--113, 1972.
    """
    return independence_number(g) <= g.vertex_connectivity()
add_to_lists(is_chvatal_erdos, intractable_properties, all_properties)


def chi_equals_min_theory(g):
    r"""
    Evaluate if chromatic num. of graph ``g`` equals min. of some upper bounds.

    Some known upper bounds on the chromatic number Chi (`\chi`) include
    our invariants `[brooks, wilf, welsh_powell, szekeres_wilf]`.
    Returns ``True`` if the actual chromatic number of ``g`` equals the minimum
    of / "the best of" these known upper bounds.

    Some of these invariants are undefined on the empty graph.

    EXAMPLES:

        sage: chi_equals_min_theory(Graph(1))
        True

        sage: chi_equals_min_theory(graphs.PetersenGraph())
        True

        sage: chi_equals_min_theory(double_fork)
        True

        sage: chi_equals_min_theory(Graph(3))
        False

        chi_equals_min_theory(graphs.CompleteBipartiteGraph(3,5))
        False

        chi_equals_min_theory(graphs.IcosahedralGraph())
        False
    """
    chromatic_upper_theory = [brooks, wilf, welsh_powell, szekeres_wilf]
    min_theory = min([f(g) for f in chromatic_upper_theory])
    return min_theory == g.chromatic_number()
add_to_lists(chi_equals_min_theory, intractable_properties, all_properties)


def is_heliotropic_plant(g):
    """
    Evaluates whether graph ``g`` is a heliotropic plant. BROKEN

    BROKEN: code should be nonnegative eigen, not just positive eigen.
    See Issue #586

    A graph is heliotropic iff the independence number equals the number of
    nonnegative eigenvalues.

    See [BDF1995]_ for a definition and some related conjectures, where
    [BDF1995]_ builds on the conjecturing work of Siemion Fajtlowicz.

    EXAMPLES:

    REFERENCES:

    .. [BDF1995]    Tony Brewster, Michael J.Dinneen, and Vance Faber, "A
                    computational attack on the conjectures of Graffiti: New
                    counterexamples and proofs". Discrete Mathematics,
                    147(1--3): 35--55, 1995.
    """
    return (independence_number(g) == card_positive_eigenvalues(g))
add_to_lists(is_heliotropic_plant, intractable_properties, all_properties)


def is_geotropic_plant(g):
    """
    Evaluates whether graph ``g`` is a heliotropic plant. BROKEN

    BROKEN: code should be nonpositive eigen, not just negative eigen.
    See Issue #586

    A graph is geotropic iff the independence number equals the number of
    nonnegative eigenvalues.

    See [BDF1995]_ for a definition and some related conjectures, where
    [BDF1995]_ builds on the conjecturing work of Siemion Fajtlowicz.

    EXAMPLES:

    REFERENCES:

    .. [BDF1995]    Tony Brewster, Michael J.Dinneen, and Vance Faber, "A
                    computational attack on the conjectures of Graffiti: New
                    counterexamples and proofs". Discrete Mathematics,
                    147(1--3): 35--55, 1995.
    """
    return (independence_number(g) == card_negative_eigenvalues(g))
add_to_lists(is_geotropic_plant, intractable_properties, all_properties)


def has_alpha_residue_equal_two(g):
    r"""
    Tests if both the residue and independence number of graphs ``g`` equal 2.

    The residue of a graph ``g`` with degrees `d_1 \geq d_2 \geq ... \geq d_n`
    is found iteratively. First, remove `d_1` from consideration and subtract
    `d_1` from the following `d_1` number of elements. Sort. Repeat this
    process for `d_2,d_3, ...` until only 0s remain. The number of elements,
    i.e. the number of 0s, is the residue of ``g``.

    Residue is undefined on the empty graph.

    EXAMPLES:

        sage: has_alpha_residue_equal_two(graphs.DiamondGraph())
        True

        sage: has_alpha_residue_equal_two(Graph(2))
        True

        sage: has_alpha_residue_equal_two(graphs.OctahedralGraph())
        True

        sage: has_alpha_residue_equal_two(graphs.BullGraph())
        False

        sage: has_alpha_residue_equal_two(graphs.BidiakisCube())
        False

        sage: has_alpha_residue_equal_two(Graph(3))
        False

        sage: has_alpha_residue_equal_two(Graph(1))
        False
    """
    if residue(g) != 2:
        return false
    else:
        return independence_number(g) == 2
add_to_lists(has_alpha_residue_equal_two, intractable_properties, all_properties)


def alpha_leq_order_over_two(g):
    """
    Tests if the independence number of graph ``g`` is at most half its order.

    EXAMPLES:

        sage: alpha_leq_order_over_two(graphs.ButterflyGraph())
        True

        sage: alpha_leq_order_over_two(graphs.DiamondGraph())
        True

        sage: alpha_leq_order_over_two(graphs.CoxeterGraph())
        True

        sage: alpha_leq_order_over_two(Graph(4))
        False

        sage: alpha_leq_order_over_two(graphs.BullGraph())
        False

    Edge cases ::

        sage: alpha_leq_order_over_two(Graph(0))
        True

        sage: alpha_leq_order_over_two(Graph(1))
        False
    """
    return (2*independence_number(g) <= g.order())
add_to_lists(alpha_leq_order_over_two, intractable_properties, all_properties)


def is_chromatic_index_critical(g):
    r"""
    Evaluates whether graph ``g`` is chromatic index critical.

    Let `\chi(G)` denote the chromatic index of a graph `G`.
    Then `G` is chromatic index critical if `\chi(G-e) < \chi(G)` (strictly
    less than) for all `e \in G` AND if (by definition) `G` is class 2.

    See [FW1977]_ for a more extended definition and discussion.

    We initially found it surprising that `G` is required to be class 2; for
    example, the Star Graph is a class 1 graph which satisfies the rest of
    the definition. We have found articles which equivalently define critical
    graphs as class 2 graphs which become class 1 when any edge is removed.
    Perhaps this latter definition inspired the one we state above?

    Max degree is undefined on the empty graph, so ``is_class`` is also
    undefined. Therefore this property is undefined on the empty graph.

    EXAMPLES:

        sage: is_chromatic_index_critical(Graph('Djk'))
        True

        sage: is_chromatic_index_critical(graphs.CompleteGraph(3))
        True

        sage: is_chromatic_index_critical(graphs.CycleGraph(5))
        True

        sage: is_chromatic_index_critical(graphs.CompleteGraph(5))
        False

        sage: is_chromatic_index_critical(graphs.PetersenGraph())
        False

        sage: is_chromatic_index_critical(graphs.FlowerSnark())
        False

    Non-trivially disconnected graphs ::

        sage: is_chromatic_index_critical(graphs.CycleGraph(4).disjoint_union(graphs.CompleteGraph(4)))
        False

    Class 1 graphs ::

        sage: is_chromatic_index_critical(Graph(1))
        False

        sage: is_chromatic_index_critical(graphs.CompleteGraph(4))
        False

        sage: is_chromatic_index_critical(graphs.CompleteGraph(2))
        False

        sage: is_chromatic_index_critical(graphs.StarGraph(4))
        False

    ALGORITHM:

    This function uses a series of tricks to reduce the number of cases that
    need to be considered, before finally checking in the obvious way.

    First, if a graph has more than 1 non-trivial connected component, then
    return ``False``. This is because in a graph with multiple such components,
    removing any edges from the smaller component cannot affect the chromatic
    index.

    Second, check if the graph is class 2. If not, stop and return ``False``.

    Finally, identify isomorphic edges using the line graph and its orbits.
    We then need only check the non-equivalent edges to see that they reduce
    the chromatic index when deleted.

    REFERENCES:

    .. [FW1977]     \S. Fiorini and R.J. Wilson, "Edge-colourings of graphs".
                    Pitman Publishing, London, UK, 1977.
    """
    component_sizes = g.connected_components_sizes()
    chi=g.chromatic_index()

    if len(component_sizes) > 1:
        if component_sizes[1] > 1:
            return False

    if chi == max_degree(g):
        return False

    lg = g.line_graph()
    equiv_lines = lg.automorphism_group(return_group=False, orbits=true)
    equiv_lines_representatives = [orb[0] for orb in equiv_lines]

    gc = g.copy()
    for e in equiv_lines_representatives:
        gc.delete_edge(e)
        chi_prime = gc.chromatic_index()
        if chi_prime == chi:
            return False
        gc.add_edge(e)
    return True
add_to_lists(is_chromatic_index_critical, intractable_properties, all_properties)


#alpha(g-e) > alpha(g) for *every* edge g
def is_alpha_critical(g):
    #if not g.is_connected():
        #return False
    alpha = independence_number(g)
    for e in g.edges(sort=true):
        gc = copy(g)
        gc.delete_edge(e)
        alpha_prime = independence_number(gc)
        if alpha_prime <= alpha:
            return False
    return True
add_to_lists(is_chromatic_index_critical, intractable_properties, all_properties)


#can't compute membership in this class directly. instead testing isomorhism for 400 known class0 graphs
def is_pebbling_class0(g):
    for hkey in class0graphs_dict:
        h = Graph(class0graphs_dict[hkey])
        if g.is_isomorphic(h):
            return True
    return False
add_to_lists(is_pebbling_class0,  intractable_properties, all_properties)


def szekeres_wilf_equals_chromatic_number(g):
    return szekeres_wilf(g) == g.chromatic_number()
add_to_lists(szekeres_wilf_equals_chromatic_number,  intractable_properties, all_properties)


#NOTE: the relevant theorem is a forbidden subgraph characterization
#its not clear if its theoretically efficient. the written code look at all subsets and is thus intractible
#we'll add to inractable properties. is there an efficient algorithm???
def has_strong_Havel_Hakimi_property(g):
    """
    Return whether the graph g has the strong Havel-Hakimi property.

    A graph has the strong Havel-Hakimi property if in every induced subgraph H of G, every vertex of maximum degree has the Havel-Hakimi property. Graphs with the strong Havel-Hakimi property, M. Barrus, G. Molnar, Graphs and Combinatorics, 2016, http://dx.doi.org/10.1007/s00373-015-1674-7

    INPUT:

    -``g``-- Sage Graph

    OUTPUT:

    -Boolean, True if the graph g has the strong Havel-Hakimi property, False if otherwise.

    EXAMPLE:

    The graph obtained by connecting two cycles of length 3 by a single edge has
    the strong Havel-Hakimi property::

        sage: has_strong_Havel_Hakimi_property(Graph('E{CW'))
        True
    """
    for S in Subsets(g.vertices(sort=true)):
        if len(S)>2:
            H = g.subgraph(S)
            Delta = max_degree(H)
            if any(not has_Havel_Hakimi_property(H, v) for v in S if H.degree(v) == Delta):
                return False
    return True
add_to_lists(has_strong_Havel_Hakimi_property, intractable_properties, all_properties)


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

# Graph g is complement_hamiltonian if the complement of the graph is hamiltonian.
def is_complement_hamiltonian(g):
    return g.complement().is_hamiltonian()
add_to_lists(is_complement_hamiltonian, intractable_properties, all_properties)


def is_1_tough(g):
    """
    See: https://en.wikipedia.org/wiki/Graph_toughness
    """
    return is_k_tough(g, 1)
add_to_lists(is_1_tough, intractable_properties, all_properties)


def is_2_tough(g):
    return is_k_tough(g, 2)
add_to_lists(is_2_tough, intractable_properties, all_properties)


# True if graph has at least two hamiltonian cycles. The cycles may share some edges.
def has_two_ham_cycles(gIn):
    g = gIn.copy()
    g.relabel()
    try:
        ham1 = g.hamiltonian_cycle()
    except EmptySetError:
        return False

    for e in ham1.edges(sort=true):
        h = copy(g)
        h.delete_edge(e)
        if h.is_hamiltonian():
            return True
    return False
add_to_lists(has_two_ham_cycles, intractable_properties, all_properties)


def is_prism_hamiltonian(g):
    """
    A graph G is prism hamiltonian if G x K2 (cartesian product) is hamiltonian
    """
    return g.cartesian_product(graphs.CompleteGraph(2)).is_hamiltonian()
add_to_lists(is_prism_hamiltonian, intractable_properties, all_properties)

# Bauer, Douglas, et al. "Long cycles in graphs with large degree sums." Discrete Mathematics 79.1 (1990): 59-70.
def is_bauer(g):
    """
    True if g is 2_tough and sigma_3 >= order
    """
    return is_2_tough(g) and sigma_k(g, 3) >= g.order()
add_to_lists(is_bauer, intractable_properties, all_properties)

# Jung, H. A. "On maximal circuits in finite graphs." Annals of Discrete Mathematics. Vol. 3. Elsevier, 1978. 129-144.
def is_jung(g):
    """
    True if graph has n >= 11, if graph is 1-tough, and sigma_2 >= n - 4.
    See functions toughness(g) and sigma_2(g) for more details.
    """
    return g.order() >= 11 and is_1_tough(g) and sigma_2(g) >= g.order() - 4
add_to_lists(is_jung, intractable_properties, all_properties)


# Bela Bollobas and Andrew Thomason, Weakly Pancyclic Graphs. Journal of Combinatorial Theory 77: 121--137, 1999.
def is_weakly_pancyclic(g):
    """
    True if g contains cycles of every length k, from k = girth to k = circumference

    Returns False if g is acyclic (in which case girth = circumference = +Infinity).

    INPUT:

    -``g``-- Sage Graph

    OUTPUT:

    -Boolean

    EXAMPLES:

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
add_to_lists(is_weakly_pancyclic, intractable_properties, all_properties)


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
add_to_lists(is_pancyclic, intractable_properties, all_properties)


def has_two_walk(g):
    """
    If the input graph g contains a two walk, returns true; otherwise false.

    A two-walk is a closed walk that visits every vertex and visits no vertex more than twice.
    Two-walk is a generalization of Hamiltonian cycles. If a graph is Hamiltonian, then it has a two-walk.

    INPUT:

    -``g``-- Sage Graph

    OUTPUT:

    -Boolean

    EXAMPLES:

    sage: has_two_walk(c4c4)
    True
    sage: has_two_walk(graphs.WindmillGraph(3,3))
    False
    """
    for init_vertex in g.vertices(sort=true):
        path_stack = [[init_vertex]]
        while path_stack:
            path = path_stack.pop()
            for neighbor in g.neighbors(path[-1]):
                if neighbor == path[0] and all(v in path for v in g.vertices(sort=true)):
                    return True
                elif path.count(neighbor) < 2:
                    path_stack.append(path + [neighbor])
    return False
add_to_lists(has_two_walk, intractable_properties, all_properties)


def has_alpha_equals_clique_covering(g):
    """
    Return true if the independence number of graph g equals its clique covering

    sage: has_alpha_equals_clique_covering(graphs.CycleGraph(5))
        False
    """
    temp = has_equal_invariants(independence_number, clique_covering_number)
    return temp(g)
add_to_lists(has_alpha_equals_clique_covering, intractable_properties, all_properties)


