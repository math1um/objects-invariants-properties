# GRAPH PROPERTIES

def has_star_center(g):
    """
    True if graph has a vertex adjacent to all others, also known as "universal vertex"
    
    sage: has_star_center(flower_with_3_petals)
    True
    sage: has_star_center(c4)
    False
    sage: has_star_center(Graph(1))
    True
    """
    return (g.order() - 1) in g.degree()

def is_complement_of_chordal(g):
    """
    True if graph g is a complement of a chordal graph
    
    sage: is_complement_of_chordal(p4)
    True
    sage: is_complement_of_chordal(p5)
    False
    """
    return g.complement().is_chordal()

def pairs_have_unique_common_neighbor(g):
    """
    True if each pair of vertices in g has exactly one common neighbor

    Also known as the friendship property. 
    By the Friendship Theorem, the only connected graphs with the friendship property are flowers.

    sage: pairs_have_unique_common_neighbor(flower(5))
    True
    sage: pairs_have_unique_common_neighbor(k3)
    True
    sage: pairs_have_unique_common_neighbor(k4)
    False
    sage: pairs_have_unique_common_neighbor(graphs.CompleteGraph(2))
    False
    """
    from itertools import combinations
    for (u,v) in combinations(g.vertices(), 2):
        if len(common_neighbors(g, u, v)) != 1:
            return False
    return True

def is_distance_transitive(g):
    """
    True if all a,b,u,v satisfy dist(a,b) = dist(u,v) => there's an automorphism with a->u and b->v

    Note that this method calls, via the automorphism group, the Gap package. This package behaves
    badly with most threading or multiprocessing tools.
    
    sage: is_distance_transitive(graphs.CompleteGraph(4))
    True
    sage: is_distance_transitive(graphs.PetersenGraph())
    True
    sage: is_distance_transitive(graphs.ShrikhandeGraph())
    False
    
    This method accepts disconnected graphs:
    sage: is_distance_transitive(graphs.CompleteGraph(3).disjoint_union(graphs.CompleteGraph(3)))
    True
    """
    from itertools import combinations
    dist_dict = g.distance_all_pairs()
    auto_group = g.automorphism_group()

    for d in g.distances_distribution():
        sameDistPairs = []
        for (u,v) in combinations(g.vertices(), 2):
            if dist_dict[u].get(v, +Infinity) == d: # By default, no entry if disconnected. We substitute +Infinity.
                sameDistPairs.append(Set([u,v]))
        if len(sameDistPairs) >= 2:
            if not(len(sameDistPairs) == len(auto_group.orbit(sameDistPairs[0], action = "OnSets"))):
                return False
    return True

def is_dirac(g):
    """
    True if g has order at least 3 and min. degree at least n/2
    
    See Dirac's Theorem: If graph is_dirac, then it is hamiltonian.
    
    sage: is_dirac(graphs.CompleteGraph(6))
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
    True if deg(v) + deg(w) >= n for all non-adjacent pairs v,w in graph g

    See Ore's Theorem: If graph is_ore, then it is hamiltonian.

    sage: is_ore(graphs.CompleteGraph(5))
    True
    sage: is_ore(dart)
    False
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
    True if g is 2-connected and min. degree >= (n + vertex_connectivity)/3
    
    R. Häggkvist and G. Nicoghossian, A remark on Hamiltonian cycles. Journal of Combinatorial 
    Theory, Series B, 30(1): 118--120, 1981.
    If is_haggkvist_nicoghossian, then is hamiltonian.
    
    sage: is_haggkvist_nicoghossian(graphs.CompleteGraph(5))
    True
    sage: is_haggkvist_nicoghossian(graphs.CycleGraph(5))
    False
    sage: is_haggkvist_nicoghossian(graphs.CompleteBipartiteGraph(4,3)
    False
    """
    k = g.vertex_connectivity()
    return k >= 2 and min(g.degree()) >= (1.0/3) * (g.order() + k)

def is_genghua_fan(g):
    """
    True if g is 2-connected and s.t. dist(u,v)=2 implies max(deg(u), deg(v)) >= n/2 for all u,v
    
    Geng-Hua Fan, New sufficient conditions for cycles in graphs. Journal of Combinatorial Theory,
    Series B, 37(3): 221--227, 1984.
    If a graph is_genghua_fan, then it is Hamiltonian.
    
    sage: is_genghua_fan(graphs.DiamondGraph())
    True
    sage: is_genghua_fan(graphs.CycleGraph(4))
    False
    sage: is_genghua_fan(graphs.ButterflyGraph())
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
    True if g is planar and is vertex-transitive
    
    sage: is_planar_transitive(graphs.HexahedralGraph())
    True
    sage: is_planar_transitive(graphs.CompleteGraph(2))
    True
    sage: is_planar_transitive(graphs.FranklinGraph())
    False
    sage: is_planar_transitive(graphs.BullGraph()) 
    False
    """
    return g.is_planar() and g.is_vertex_transitive()

def is_generalized_dirac(g):
    """
    Tests whether a graph g satisfies a condition in a generalization of Dirac's Theorem.
    
    OUTPUT:
    
    Returns ``True`` if g is 2-connected and for all non-adjacent u,v, the cardinality of the 
    union of neighborhood(u) and neighborhood(v) is `>= (2n-1)/3`.
    
    REFERENCES:
    
    Theorem: If graph g is_generalized_dirac, then it is Hamiltonian.
    
    .. [FGJS1989]   \R.J. Faudree, Ronald Gould, Michael Jacobson, and R.H. Schelp, 
                    "Neighborhood unions and hamiltonian properties in graphs". 
                    Journal of Combinatorial Theory, Series B, 47(1): 1--9, 1989.

    EXAMPLES:
    
    This example illustrates ::
        
        sage: 
        True
        sage:
        False
        
    We now ::
    
        sage:
    """
    from itertools import combinations

    
            if dist_dict[u].get(v, +Infinity) == d: # By default, no entry if disconnected. We substitute +Infinity.
                sameDistPairs.append(Set([u,v]))
                
                
    if not is_two_connected(g):
        return False
    for (u,v) in combinations(g.vertices(), 2):
        if not g.has_edge(u,v):
            if len(neighbors_set((u, v)) < (2.0 * g.order() - 1) / 3:
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
            for (i,_), entry in A.dict().iteritems():
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
    Equivalent to Graph.is_biconnected(), but we prefer our name

    Returns True if the graph is 2-connected and False otherwise. A graph is
    2-connected if the removal of any single vertex gives a connected graph.

    If a graph has fewer than 3 vertices, a ValueError is raised.

        sage: is_two_connected(graphs.CycleGraph(5))
        True
        sage: is_two_connected(graphs.PathGraph(5))
        False
        sage: is_two_connected(graphs.CompleteGraph(2))
        Traceback (most recent call last):
            File "", line 20, in is_two_connected
        ValueError: is_two_connected is only defined on graphs with at least 3 vertices.
    """
    if g.order() < 3:
        raise ValueError("is_two_connected is only defined on graphs with at least 3 vertices.")
    return g.is_biconnected()

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

def is_four_connected(g):
    """
    True if g is at least four connected, i.e. must remove at least 4 vertices to disconnect graph

    Implementation requires Sage 8.2+.
    """
    return g.vertex_connectivity(k = 4)

#sufficient condition for hamiltonicity
def is_lindquester(g):
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
    """
    Equivalent to is a cograph - https://en.wikipedia.org/wiki/Cograph
    """
    return not has_p4(g)

def has_p4(g):
    return g.subgraph_search(p4, induced=True) is not None

def has_kite(g):
    return g.subgraph_search(kite_with_tail, induced=True) is not None

def is_kite_free(g):
    return not has_kite(g)

def has_claw(g):
    return g.subgraph_search(graphs.ClawGraph(), induced=True) is not None

def is_claw_free(g):
    return not has_claw(g)

def has_H(g):
    return g.subgraph_search(double_fork, induced=True) is not None

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
    sage: is_biclique(graphs.ButterflyGraph())
    True
    """
    gc = g.complement()
    return gc.is_bipartite()

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
    sage: has_odd_order(Graph(0))
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
    sage: has_even_order(Graph(0))
    True
    """
    return g.order() % 2 == 0

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
is_locally_two_connected = localise(is_two_connected)
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
diameter_equals_radius, is_locally_connected, matching_covered, is_locally_dirac,
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
Graph.is_self_complementary, is_biclique]

intractable_properties = [Graph.is_hamiltonian, Graph.is_vertex_transitive,
Graph.is_edge_transitive, has_residue_equals_alpha, Graph.is_odd_hole_free,
Graph.is_semi_symmetric, Graph.is_line_graph, is_planar_transitive, is_class1,
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
    This means checked for new functions, and for any errors in old functions!
    sage: sage.misc.banner.version_dict()['major'] < 8 or (sage.misc.banner.version_dict()['major'] == 8 and sage.misc.banner.version_dict()['minor'] <= 2)
    True

    Skip Graph.is_circumscribable() and Graph.is_inscribable() because they throw errors for the vast majority of our graphs.
    Skip Graph.is_biconnected() in favor of our name is_two_connected().
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
