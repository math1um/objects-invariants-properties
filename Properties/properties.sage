
#GRAPH PROPERTIES

def has_star_center(g):
    """
    tests if graph has vertex adjacent to all others
        sage: has_star_center(flower_with_3_petals)
        True
        sage: has_star_center(c4)
        False
    """
    n = g.order()
    return max_degree(g) == (n-1)


#split graphs have the property that their complements are chordal
def is_complement_of_chordal(g):
    """
    tests is a graph is a complement of a chordal graph
        sage: is_complement_of_chordal(p4)
        True
        sage: is_complement_of_chordal(p5)
        False
    """
    h = g.complement()
    return h.is_chordal()

#a consequence of the Friendship Theorem:
#the only connected graphs where every pair of vertices has a unique neghbor are flowers
def pairs_have_unique_common_neighbor(g):
    """
    tests if a graph is a collection of disjoint triangles with a single identified vertex
        sage: pairs_have_unique_common_neighbor(flower(5))
        True
        sage: pairs_have_unique_common_neighbor(k3)
        True
        sage: pairs_have_unique_common_neighbor(k4)
        False
    """
    if max_common_neighbors(g) != 1:
        return False
    elif min_common_neighbors(g) != 1:
        return False
    else:
        return True

#sufficient condition for hamiltonicity
def is_dirac(g):
    n = g.order()
    deg = g.degree()
    delta=min(deg)
    if n/2 <= delta and n > 2:
        return True
    else:
        return False

#sufficient condition for hamiltonicity
def is_ore(g):
    A = g.adjacency_matrix()
    n = g.order()
    D = g.degree()
    for i in range(n):
        for j in range(i):
            if A[i][j]==0:
                if D[i] + D[j] < n:
                    return False
    return True

#sufficient condition for hamiltonicity
def is_haggkvist_nicoghossian(g):
    k = vertex_con(g)
    n = g.order()
    delta = min(g.degree())
    if k >= 2 and delta >= (1.0/3)*(n+k):
        return True
    else:
        return False

#sufficient condition for hamiltonicity
def is_fan(g):
    k = vertex_con(g)
    if k < 2:
        return False
    D = g.degree()
    Dist = g.distance_all_pairs()
    V = g.vertices()
    n = g.order()
    for i in range(n):
        for j in range (i):
            if Dist[V[i]][V[j]]==2 and max(D[i],D[j]) < n/2.0:
                return False
    return True

#sufficient condition for hamiltonicity
def is_planar_transitive(g):
    if g.order() > 2 and g.is_planar() and g.is_vertex_transitive():
        return True
    else:
        return False

def neighbors_set(g,S):
    N = set()
    for v in S:
        for n in g.neighbors(v):
            N.add(n)
    return list(N)

#sufficient condition for hamiltonicity
def is_generalized_dirac(g):
    n = g.order()
    k = vertex_con(g)
    if k < 2:
        return False
    for p in Subsets(g.vertices(),2):
        if g.is_independent_set(p):
            if len(neighbors_set(g,p)) < (2.0*n-1)/3:
                return False
    return True

#necessary condition for hamiltonicity
def is_van_den_heuvel(g):
    n = g.order()
    lc = sorted(graphs.CycleGraph(n).laplacian_matrix().eigenvalues())
    lg = sorted(g.laplacian_matrix().eigenvalues())

    for lci, lgi in zip(lc, lg):
        if lci > lgi:
            return False

    def Q(g):
        A = g.adjacency_matrix()

        D = A.parent(0)

        if A.is_sparse():
            row_sums = {}
            for (i,j), entry in A.dict().iteritems():
                row_sums[i] = row_sums.get(i, 0) + entry
            for i in range(A.nrows()):
                D[i,i] += row_sums.get(i, 0)
        else:
            row_sums=[sum(v) for v in A.rows()]
            for i in range(A.nrows()):
                D[i,i] += row_sums[i]

        return D + A

    qc = sorted(Q(graphs.CycleGraph(n)).eigenvalues())
    qg = sorted(Q(g).eigenvalues())

    for qci, qgi in zip(qc, qg):
        if qci > qgi:
            return False

    return True

#necessary condition for hamiltonicity
def is_two_connected(g):
    """
    Returns True if the graph is 2-connected and False otherwise. A graph is
    2-connected if the removal of any single vertex gives a connected graph.
    By definition a graph on 2 or less vertices is not 2-connected.

        sage: is_two_connected(graphs.CycleGraph(5))
        True
        sage: is_two_connected(graphs.PathGraph(5))
        False
        sage: is_two_connected(graphs.CompleteGraph(2))
        False
        sage: is_two_connected(graphs.CompleteGraph(1))
        False
    """
    if g.order() <= 2:
        return False
    from itertools import combinations
    for s in combinations(g.vertices(), g.order() - 1):
        if not g.subgraph(s).is_connected():
            return False
    return True

#part of pebbling class0 sufficient condition
def is_three_connected(g):
    """
    Returns True if the graph is 3-connected and False otherwise. A graph is
    3-connected if the removal of any single vertex or any pair of vertices
    gives a connected graph. By definition a graph on 3 or less vertices is
    not 3-connected.

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
        sage: is_three_connected(graphs.CompleteGraph(1))
        False
    """
    if g.order() <= 3:
        return False
    from itertools import combinations
    for s in combinations(g.vertices(), g.order() - 2):
        if not g.subgraph(s).is_connected():
            return False
    return True

#sufficient condition for hamiltonicity
def is_lindquester(g):
    k = vertex_con(g)
    if k < 2:
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
    n = g.order()
    e = g.size()
    if not g.has_multiple_edges():
        return e == n*(n-1)/2

    D = g.distance_all_pairs()
    for i in range(n):
        for j in range(i):
            if D[V[i]][V[j]] != 1:
                return False
    return True

def has_c4(g):
    return g.subgraph_search(c4, induced=True) is not None

def is_c4_free(g):
    return not has_c4(g)

def has_paw(g):
    return g.subgraph_search(paw, induced=True) is not None

def is_paw_free(g):
    return not has_paw(g)

def has_dart(g):
    return g.subgraph_search(dart, induced=True) is not None

def is_dart_free(g):
    return not has_dart(g)

def is_p4_free(g):
    return not has_p4(g)

def has_p4(g):
    return g.subgraph_search(p4, induced=True) is not None

def has_kite(g):
    return g.subgraph_search(kite, induced=True) is not None

def is_kite_free(g):
    return not has_kite(g)

def has_claw(g):
    return g.subgraph_search(graphs.ClawGraph(), induced=True) is not None

def is_claw_free(g):
    return not has_claw(g)

def has_H(g):
    return g.subgraph_search(killer, induced=True) is not None

def is_H_free(g):
    return not has_H(g)

def has_fork(g):
    return g.subgraph_search(fork, induced=True) is not None

def is_fork_free(g):
    return not has_fork(g)

def has_k4(g):
    return g.subgraph_search(alpha_critical_easy[2], induced=True) is not None

def is_k4_free(g):
    return not has_k4(g)

def is_biclique(g):
    """
    a graph is a biclique if the vertices can be partitioned into 2 sets that induce cliques
    sage: is_biclique(p4)
    True
    sage: is_biclique(bow_tie)
    True
    """
    gc = g.complement()
    return gc.is_bipartite()

def has_perfect_matching(g):
    n = g.order()
    if n%2 == 1:
        return False
    nu = g.matching(value_only=True)
    if 2*nu == n:
        return True
    else:
        return False

#true if radius equals diameter
def has_radius_equal_diameter(g):
    return g.radius() == g.diameter()

#true if residue equals independence number
def has_residue_equals_alpha(g):
    return residue(g) == independence_number(g)

def is_not_forest(g):
    return not g.is_forest()


def bipartite_double_cover(g):
    return g.tensor_product(graphs.CompleteGraph(2))

def closed_neighborhood(g, verts):
    if isinstance(verts, list):
        neighborhood = []
        for v in verts:
            neighborhood += [v] + g.neighbors(v)
        return list(set(neighborhood))
    else:
        return [verts] + g.neighbors(verts)

#replaced with faster LP-solver is_independence_irreducible
#has no non-empty critical independent set (<=> the only solution to the LP independence number relaxation is all 1/2's)
def has_empty_KE_part(g):
    b = bipartite_double_cover(g)
    alpha = b.order() - b.matching(value_only=True)
    for v in g.vertices():
        test = b.copy()
        test.delete_vertices(closed_neighborhood(b,[(v,0), (v,1)]))
        alpha_test = test.order() - test.matching(value_only=True) + 2
        if alpha_test == alpha:
            return False
    return True

def is_bicritical(g):
    return has_empty_KE_part(g)


# Vizing's Theorem: chromatic index of any graph is either Delta or Delta+1
def is_class1(g):
    return chromatic_index(g) == max(g.degree())

def is_class2(g):
    return not(chromatic_index(g) == max(g.degree()))

def is_cubic(g):
    D = g.degree()
    return min(D) == 3 and max(D) == 3

#a property that applied to all entered hamiltonian graphs (before c60) but not the tutte graph, false for tutte graph
def is_anti_tutte(g):
    if not g.is_connected():
        return False
    return independence_number(g) <= g.diameter() + g.girth()

#a property that applied to all entered hamiltonian graphs upto c6cc6 but not the tutte graph, false for tutte graph
def is_anti_tutte2(g):
    if not g.is_connected():
        return False
    return independence_number(g) <=  domination_number(g) + g.radius()- 1

#for any graph diam <= 2*radius. this property is true in the extremal case
def diameter_equals_twice_radius(g):
    if not g.is_connected():
        return False
    return g.diameter() == 2*g.radius()

#for any graph diam >= radius. this property is true in the extremal case
def diameter_equals_radius(g):
    if not g.is_connected():
        return False
    return g.diameter() == g.radius()

#almost all graphs have diameter equals 2
def diameter_equals_two(g):
    if not g.is_connected():
        return False
    return g.diameter() == 2

def has_lovasz_theta_equals_alpha(g):
    if g.lovasz_theta() == independence_number(g):
        return True
    else:
        return False

def has_lovasz_theta_equals_cc(g):
    if g.lovasz_theta() == clique_covering_number(g):
        return True
    else:
        return False

#sufficient condition for hamiltonicity
def is_chvatal_erdos(g):
    return independence_number(g) <= vertex_con(g)

#locally connected if the neighborhood of every vertex is connected (stronger than claw-free)
def is_locally_connected(g):
    V = g.vertices()
    for v in V:
        N = g.neighbors(v)
        if len(N) > 0:
            GN = g.subgraph(N)
            if not GN.is_connected():
                return False
    return True


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
    chi = chromatic_num(g)
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
    chi_e = chromatic_index(g)
    if chi_e != Delta + 1:
        return False

    lg=g.line_graph()
    equiv_lines = lg.automorphism_group(return_group=False,orbits=true)
    equiv_lines_representatives = [orb[0] for orb in equiv_lines]

    for e in equiv_lines_representatives:
        gc = copy(g)
        gc.delete_edge(e)
        chi_e_prime = chromatic_index(gc)
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
        is_factor_critical(p3)
        False
        sage: is_factor_critical(c5)
        True
    """
    if g.order() % 2 == 0:
        return False
    for v in g.vertices():
        gc = copy(g)
        gc.delete_vertex(v)
        if not has_perfect_matching(gc):
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
                twin = w
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

def is_cycle(g):
    return g.is_isomorphic(graphs.CycleGraph(g.order()))


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


def localise(f):
    """
    This function takes a property (i.e., a function taking only a graph as an argument) and
    returns the local variant of that property. The local variant is True if the property is
    True for the neighbourhood of each vertex and False otherwise.
    """
    #create a local version of f
    def localised_function(g):
        return all((f(g.subgraph(g.neighbors(v))) if g.neighbors(v) else True) for v in g.vertices())

    #we set a nice name for the new function
    if hasattr(f, '__name__'):
        if f.__name__.startswith('is_'):
            localised_function.__name__ = 'is_locally' + f.__name__[2:]
        elif f.__name__.startswith('has_'):
            localised_function.__name__ = 'has_locally' + f.__name__[2:]
        else:
            localised_function.__name__ = 'localised_' + f.__name__

    return localised_function

is_locally_dirac = localise(is_dirac)
is_locally_bipartite = localise(Graph.is_bipartite)

#old versioncted), can't seem to memoize that

def is_locally_two_connected(g):
    V = g.vertices()
    for v in V:
        N = g.neighbors(v)
        if len(N) > 0:
            GN = g.subgraph(N)
            if not is_two_connected(GN):
                return False
    return True

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
    return max_degree(g) == 3

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

#add all properties derived from pairs of invariants
invariant_relation_properties = [has_leq_invariants(f,g) for f in all_invariants for g in all_invariants if f != g]


efficiently_computable_properties = [Graph.is_regular, Graph.is_planar,
Graph.is_forest, Graph.is_eulerian, Graph.is_connected, Graph.is_clique,
Graph.is_circular_planar, Graph.is_chordal, Graph.is_bipartite,
Graph.is_cartesian_product,Graph.is_distance_regular,  Graph.is_even_hole_free,
Graph.is_gallai_tree, Graph.is_line_graph, Graph.is_overfull, Graph.is_perfect,
Graph.is_split, Graph.is_strongly_regular, Graph.is_triangle_free,
Graph.is_weakly_chordal, is_dirac, is_ore, is_haggkvist_nicoghossian,
is_generalized_dirac, is_van_den_heuvel, is_two_connected, is_three_connected,
is_lindquester, is_claw_free, has_perfect_matching, has_radius_equal_diameter,
is_not_forest, is_fan, is_cubic, diameter_equals_twice_radius,
diameter_equals_radius, is_locally_connected, matching_covered, is_locally_dirac,
is_locally_bipartite, is_locally_two_connected, Graph.is_interval, has_paw,
is_paw_free, has_p4, is_p4_free, has_dart, is_dart_free, has_kite, is_kite_free,
has_H, is_H_free, has_residue_equals_two, order_leq_twice_max_degree,
alpha_leq_order_over_two, is_factor_critical, is_independence_irreducible,
has_twin, is_twin_free, diameter_equals_two, girth_greater_than_2log, is_cycle,
pairs_have_unique_common_neighbor, has_star_center, is_complement_of_chordal, 
has_c4, is_c4_free, is_subcubic, is_quasi_regular, is_bad, has_k4, is_k4_free]

intractable_properties = [Graph.is_hamiltonian, Graph.is_vertex_transitive,
Graph.is_edge_transitive, has_residue_equals_alpha, Graph.is_odd_hole_free,
Graph.is_semi_symmetric, Graph.is_line_graph, is_planar_transitive, is_class1,
is_class2, is_anti_tutte, is_anti_tutte2, has_lovasz_theta_equals_cc,
has_lovasz_theta_equals_alpha, is_chvatal_erdos, is_heliotropic_plant,
is_geotropic_plant, is_traceable, is_chordal_or_not_perfect,
has_alpha_residue_equal_two]

removed_properties = [is_pebbling_class0]

#speed notes
#FAST ENOUGH (tested for graphs on 140921): is_hamiltonian, is_vertex_transitive,
#    is_edge_transitive, has_residue_equals_alpha, is_odd_hole_free, is_semi_symmetric,
#    is_line_graph, is_line_graph, is_anti_tutte, is_planar_transitive
#SLOW but FIXED for SpecialGraphs: is_class1, is_class2

properties = efficiently_computable_properties + intractable_properties
properties_plus = efficiently_computable_properties + intractable_properties + invariant_relation_properties


invariants_from_properties = [make_invariant_from_property(property) for property in properties]
invariants_plus = all_invariants + invariants_from_properties

# Graph.is_prime removed as faulty 9/2014
# built in Graph.is_transitively_reduced removed 9/2014
# is_line_graph is theoretically efficient - but Sage's implementation is not 9/2014

# weakly_chordal = weakly chordal, i.e., the graph and its complement have no induced cycle of length at least 5


