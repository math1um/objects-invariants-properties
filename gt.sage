#GRAPH INVARIANTS

def barrus_bound(g):
    """
    returns n - barrus q
    defined in: Barrus, Michael D. "Havel–Hakimi residues of unigraphs." Information Processing Letters 112.1 (2012): 44-48.
    sage: barrus_bound(k4):
    1
    sage: barrus_bound(graphs.OctahedralGraph())
    2
    """
    return g.order() - barrus_q(g)

def distinct_degrees(g):
    """
    returns the number of distinct degrees of a graph
        sage: distinct_degrees(p4)
        2
        sage: distinct_degrees(k4)
        1
    """
    return len(set(g.degree()))

#inspired by the Friendship Theorem
def common_neighbors(g,v,w):
    """
    returns the Set of common neighbors of v and w in graph g
        sage: common_neighbors(p4,0,3)
        {}
        sage: common_neighbors(p4,0,2)
        {1}
    """
    Nv = Set(g.neighbors(v))
    Nw = Set(g.neighbors(w))
    return Nv.intersection(Nw)

def max_common_neighbors(g):
    """
    returns the maximum number of common neighbors of any pair of distinct vertices in g
        sage: max_common_neighbors(p4)
        1
        sage: max_common_neighbors(k4)
        2
    """
    max = 0
    V = g.vertices()
    n = g.order()
    for i in range(n):
        for j in range(n):
            if i < j:
                temp = len(common_neighbors(g, V[i], V[j]))
                if temp > max:
                    max = temp
    return max

def min_common_neighbors(g):
    """
    returns the minimum number of common neighbors of any pair of distinct vertices in g,
    which is necessarily 0 for disconnected graphs
        sage: min_common_neighbors(p4)
        0
        sage: min_common_neighbors(k4)
        2
    """
    n = g.order()
    min = n
    V = g.vertices()
    for i in range(n):
        for j in range(n):
            if i < j:
                temp = len(common_neighbors(g, V[i], V[j]))
                #if temp == 0:
                    #print "i={}, j={}".format(i,j)
                if temp < min:
                    min = temp
    return min

def mean_common_neighbors(g):
    """
    returns the average number of common neighbors of any pair of distinct vertices in g
        sage: mean_common_neighbors(p4)
        1/3
        sage: mean_common_neighbors(k4)
        2
    """
    V = g.vertices()
    n = g.order()
    sum = 0
    for i in range(n):
        for j in range(n):
            if i < j:
                sum += len(common_neighbors(g, V[i], V[j]))
    return 2*sum/(n*(n-1))

def domination_number(g):
    """
    Returns the domination number of the graph g, i.e., the size of a maximum
    dominating set.

    A complete graph is dominated by any of its vertices::

        sage: domination_number(graphs.CompleteGraph(5))
        1

    A star graph is dominated by its central vertex::

        sage: domination_number(graphs.StarGraph(5))
        1

    The domination number of a cycle of length n is the ceil of n/3.

        sage: domination_number(graphs.CycleGraph(5))
        2
    """
    return g.dominating_set(value_only=True)

def min_degree(g):
    """
    Returns the minimum of all degrees of the graph g.

        sage: min_degree(graphs.CompleteGraph(5))
        4
        sage: min_degree(graphs.CycleGraph(5))
        2
        sage: min_degree(graphs.StarGraph(5))
        1
        sage: min_degree(graphs.CompleteBipartiteGraph(3,5))
        3
    """
    return min(g.degree())

def max_degree(g):
    """
    Returns the maximum of all degrees of the graph g.

        sage: max_degree(graphs.CompleteGraph(5))
        4
        sage: max_degree(graphs.CycleGraph(5))
        2
        sage: max_degree(graphs.StarGraph(5))
        5
        sage: max_degree(graphs.CompleteBipartiteGraph(3,5))
        5
    """
    return max(g.degree())

def eulerian_faces(g):
    """
    Returns 2 - order + size, which is the number of faces if the graph is planar,
    a consequence of Euler's Formula

        sage: eulerian_faces(graphs.CycleGraph(5))
        2
        sage: eulerian_faces(graphs.DodecahedralGraph())
        12
    """
    n = g.order()
    m = g.size()
    return 2 - n + m

def barrus_q(g):
    """
    If the degrees sequence is in non-increasing order, with index starting at 1,
    barrus_q = max(k:d_k >= k)

    Defined by M. Barrus in "Havel-Hakimi Residues of Unigraphs", 2012

        sage: barrus_q(graphs.CompleteGraph(5))
        4
        sage: barrus_q(graphs.StarGraph(3))
        1

    """
    Degrees = g.degree()
    Degrees.sort()
    Degrees.reverse()
    return max(k for k in range(g.order()) if Degrees[k] >= (k+1)) + 1

def matching_number(g):
    """
    Returns the matching number of the graph g, i.e., the size of a maximum
    matching. A matching is a set of independent edges.

        sage: matching_number(graphs.CompleteGraph(5))
        2
        sage: matching_number(graphs.CycleGraph(5))
        2
        sage: matching_number(graphs.StarGraph(5))
        1
        sage: matching_number(graphs.CompleteBipartiteGraph(3,5))
        3
    """
    return int(g.matching(value_only=True, use_edge_labels=False))

def residue(g):
    seq = g.degree_sequence()

    while seq[0] > 0:
        d = seq.pop(0)
        seq[:d] = [k-1 for k in seq[:d]]
        seq.sort(reverse=True)

    return len(seq)

def annihilation_number(g):
    seq = sorted(g.degree())

    a = 0
    while sum(seq[:a+1]) <= sum(seq[a+1:]):
        a += 1

    return a

def fractional_alpha(g):
    if len(g.vertices()) == 0:
        return 0
    p = MixedIntegerLinearProgram(maximization=True)
    x = p.new_variable(nonnegative=True)
    p.set_objective(sum(x[v] for v in g.vertices()))

    for v in g.vertices():
        p.add_constraint(x[v], max=1)

    for (u,v) in g.edge_iterator(labels=False):
        p.add_constraint(x[u] + x[v], max=1)

    return p.solve()

def fractional_covering(g):
    if len(g.vertices()) == 0:
        return 0
    p = MixedIntegerLinearProgram(maximization=False)
    x = p.new_variable(nonnegative=True)
    p.set_objective(sum(x[v] for v in g.vertices()))

    for v in g.vertices():
        p.add_constraint(x[v], min=1)

    for (u,v) in g.edge_iterator(labels=False):
        p.add_constraint(x[u] + x[v], min=1)

    return p.solve()



def cvetkovic(g):
    eigenvalues = g.spectrum()
    positive = 0
    negative = 0
    zero = 0
    for e in eigenvalues:
        if e > 0:
            positive += 1
        elif e < 0:
            negative += 1
        else:
            zero += 1

    return zero + min([positive, negative])

def cycle_space_dimension(g):
    return g.size()-g.order()+g.connected_components_number()

def card_center(g):
    return len(g.center())

def cycle_space_dimension(g):
    return g.size()-g.order()+g.connected_components_number()

def card_periphery(g):
    return len(g.periphery())

def max_eigenvalue(g):
    return max(g.adjacency_matrix().change_ring(RDF).eigenvalues())

def resistance_distance_matrix(g):
    L = g.laplacian_matrix()
    n = g.order()
    J = ones_matrix(n,n)
    temp = L+(1.0/n)*J
    X = temp.inverse()
    R = (1.0)*ones_matrix(n,n)
    for i in range(n):
        for j in range(n):
            R[i,j] = X[i,i] + X[j,j] - 2*X[i,j]
    return R


def kirchhoff_index(g):
    R = resistance_distance_matrix(g)
    return .5*sum(sum(R))

def largest_singular_value(g):
    A = matrix(RDF,g.adjacency_matrix())
    svd = A.SVD()
    sigma = svd[1]
    return sigma[0,0]

def independence_number(g):
    return g.independent_set(value_only=True)


def chromatic_index(g):
    if g.size() == 0:
        return 0
    import sage.graphs.graph_coloring
    return sage.graphs.graph_coloring.edge_coloring(g, value_only=True)


def chromatic_num(g):
    return g.chromatic_number()

def card_max_cut(g):
    return g.max_cut(value_only=True)


def clique_covering_number(g):
    # Finding the chromatic number of the complement of a fullerene
    # is extremely slow, even when using MILP as the algorithm.
    # Therefore we check to see if the graph is triangle-free.
    # If it is, then the clique covering number is equal to the
    # number of vertices minus the size of a maximum matching.
    if g.is_triangle_free():
        return g.order() - matching_number(g)
    gc = g.complement()
    return gc.chromatic_number(algorithm="MILP")

def welsh_powell(g):
    n= g.order()
    D = g.degree()
    D.sort(reverse=True)
    mx = 0
    for i in range(n):
        temp = min({i,D[i]})
        if temp > mx:
            mx = temp
    return mx + 1

#outputs upper bound from Brooks Theorem: returns Delta + 1 for complete and odd cycles
def brooks(g):
    Delta = max(g.degree())
    delta = min(g.degree())
    n = g.order()
    if is_complete(g):
        return Delta + 1
    elif n%2 == 1 and g.is_connected() and Delta == 2 and delta == 2: #same as if all degrees are 2
        return Delta + 1
    else:
        return Delta

#wilf's upper bound for chromatic number
def wilf(g):
    return max_eigenvalue(g) + 1

def n_over_alpha(g):
    n = g.order() + 0.0
    return n/independence_number(g)

def median_degree(g):
    return median(g.degree())


#a measure of irregularity
def different_degrees(g):
    return len(set(g.degree()))

def szekeres_wilf(g):
    #removes a vertex, if possible, of degree <= i
    def remove_vertex_of_degree(gc,i):
        Dc = gc.degree()
        V = gc.vertices()
        #print "Dc is %s, V is %s" %(Dc,V)
        mind = min(Dc)
        #print "mind is %s" %mind
        if mind <= i:

            ind = Dc.index(mind)
            #print "ind is %s, vertex is %s" %(ind,V[ind])
            return gc.delete_vertex(V[ind])
        else:
            return gc
    D = g.degree()
    delta = min(D)
    Delta = max(D)
    for i in range(delta,Delta+1):
        gc = copy(g)
        value = g.order() + 1
        while gc.size() > 0 and gc.order() < value:
            #gc.show()
            value = gc.order()
            remove_vertex_of_degree(gc,i)
        if gc.size() == 0:
            return i + 1

def average_vertex_temperature(g):
     D = g.degree()
     n = g.order()
     return sum([D[i]/(n-D[i]+0.0) for i in range(n)])/n

def sum_temperatures(g):
     D = g.degree()
     n = g.order()
     return sum([D[i]/(n-D[i]+0.0) for i in range(n)])

def randic(g):
     D = g.degree()
     V = g.vertices()
     if min(D) == 0:
          return oo
     sum = 0
     for e in g.edges():
         v = e[0]
         i = V.index(v)
         w = e[1]
         j = V.index(w)
         sum += 1.0/(D[i]*D[j])**0.5
     return sum

#a solution of the invariant interpolation problem for upper bound of chromatic number for c8chords
#all upper bounds in theory have value at least 3 for c8chords
#returns 2 for bipartite graphs, order for non-bipartite
def bipartite_chromatic(g):
    if g.is_bipartite():
        return 2
    else:
        return g.order()

#a very good lower bound for alpha
def max_even_minus_even_horizontal(g):
    """
    finds def max_even_minus_even_horizontal for each component and adds them up.
    """
    mx_even=0
    Gcomps=g.connected_components_subgraphs()

    while Gcomps != []:
            H=Gcomps.pop()
            temp=max_even_minus_even_horizontal_component(H)
            mx_even+=temp
            #print "temp = {}, mx_even = {}".format(temp,mx_even)

    return mx_even

def max_even_minus_even_horizontal_component(g):
    """
    for each vertex v, find the number of vertices at even distance from v,
    and substract the number of edges induced by these vertices.
    this number is a lower bound for independence number.
    take the max. returns 0 if graph is not connected
    """
    if g.is_connected()==False:
        return 0

    distances = g.distance_all_pairs()
    mx=0
    n=g.order()
    for v in g.vertices():
        Even=[]
        for w in g.vertices():
            if distances[v][w]%2==0:
                Even.append(w)

        #print len(Even), len(g.subgraph(Even).edges())
        l=len(Even)-len(g.subgraph(Even).edges())
        if l>mx:
            mx=l
    return mx

def median_degree(g):
    return median(g.degree())

def laplacian_energy(g):
     L = g.laplacian_matrix().change_ring(RDF).eigenvalues()
     Ls = [1/lam**2 for lam in L if lam > 0]
     return 1 + sum(Ls)

#sum of the positive eigenvalues of a graph
def gutman_energy(g):
     L = g.adjacency_matrix().change_ring(RDF).eigenvalues()
     Ls = [lam for lam in L if lam > 0]
     return sum(Ls)

#the second smallest eigenvalue of the Laplacian matrix of a graph, also called the "algebraic connectivity" - the smallest should be 0
def fiedler(g):
     L = g.laplacian_matrix().change_ring(RDF).eigenvalues()
     L.sort()
     return L[1]

def average_degree(g):
     return mean(g.degree())

def degree_variance(g):
     mu = mean(g.degree())
     s = sum((x-mu)**2 for x in g.degree())
     return s/g.order()

def number_of_triangles(g):
     E = g.edges()
     D = g.distance_all_pairs()
     total = 0
     for e in E:
         v = e[0]
         w = e[1]
         S = [u for u in g.vertices() if D[u][v] == 1 and D[u][w] == 1]
         total += len(S)
     return total/3

def graph_rank(g):
    return g.adjacency_matrix().rank()

def inverse_degree(g):
    return sum([(1.0/d) for d in g.degree() if d!= 0])

def card_positive_eigenvalues(g):
    return len([lam for lam in g.adjacency_matrix().eigenvalues() if lam > 0])

def card_zero_eigenvalues(g):
    return len([lam for lam in g.adjacency_matrix().eigenvalues() if lam == 0])

def card_negative_eigenvalues(g):
    return len([lam for lam in g.adjacency_matrix().eigenvalues() if lam < 0])

def card_cut_vertices(g):
    return len((g.blocks_and_cut_vertices())[1])

def independent_dominating_set_number(g):
    return g.dominating_set(value_only=True, independent=True)

#returns average of distances between *distinct* vertices, return infinity is graph is not connected
def average_distance(g):
    if not g.is_connected():
        return Infinity
    V = g.vertices()
    D = g.distance_all_pairs()
    n = g.order()
    return sum([D[v][w] for v in V for w in V if v != w])/(n*(n-1))

#return number of leafs or pendants
def card_pendants(g):
    return sum([x for x in g.degree() if x == 1])


def vertex_con(g):
    return g.vertex_connectivity()


def edge_con(g):
    return g.edge_connectivity()

#returns number of bridges in graph
def card_bridges(g):
    gs = g.strong_orientation()
    bridges = []
    for scc in gs.strongly_connected_components():
        bridges.extend(gs.edge_boundary(scc))
    return len(bridges)

#upper bound for the domination number
def alon_spencer(g):
    delta = min(g.degree())
    n = g.order()
    return n*((1+log(delta + 1.0)/(delta + 1)))

#lower bound for residue and, hence, independence number
def caro_wei(g):
    return sum([1.0/(d + 1) for d in g.degree()])

#equals 2*size, the 1st theorem of graph theory
def degree_sum(g):
    return sum(g.degree())

#smallest sum of degrees of non-adjacent degrees, invariant in ore condition for hamiltonicity
#default for complete graph?
def sigma_2(g):
    if g.size() == g.order()*(g.order()-1)/2:
        return g.order()
    Dist = g.distance_all_pairs()
    return min(g.degree(v) + g.degree(w) for v in g for w in g if Dist[v][w] > 1)

#cardinality of the automorphism group of the graph
def order_automorphism_group(g):
    return g.automorphism_group(return_group=False, order=True)

#in sufficient condition for graphs where vizing's independence theorem holds
def brinkmann_steffen(g):
    E = g.edges()
    if len(E) == 0:
        return 0
    Dist = g.distance_all_pairs()
    return min(g.degree(v) + g.degree(w) for v in g for w in g if Dist[v][w] == 1)

#cardinality of the automorphism group of the graph
def order_automorphism_group(g):
    return g.automorphism_group(return_group=False, order=True)

def alpha_critical_optimum(g, alpha_critical_names):

    n = g.order()
    V = g.vertices()
    #g.show()

    alpha_critical_graph_names = []

    #get alpha_critical graphs with order <= n
    for name in alpha_critical_names:
        h = Graph(name)
        if h.order() <= n:
            alpha_critical_graph_names.append(h.graph6_string())

    #print alpha_critical_graphs

    LP = MixedIntegerLinearProgram(maximization=True)
    b = LP.new_variable(nonnegative=True)

    # We define the objective
    LP.set_objective(sum([b[v] for v in g]))

    # For any edge, we define a constraint
    for (u,v) in g.edges(labels=None):
        LP.add_constraint(b[u]+b[v],max=1)
        #LP.add_constraint(b[u]+b[v],min=1)

    #search through all subsets of g with order >= 3
    #and look for *any* subgraph isomorphic to an alpha critical graph
    #for any match we define a constraint

    i = 3
    while i <= n:
        SS = Subsets(Set(V),i)
        for S in SS:
            L = [g6 for g6 in alpha_critical_graph_names if Graph(g6).order() == i]
            #print L
            for g6 in L:
                h = Graph(g6)
                if g.subgraph(S).subgraph_search(h, induced=False):

                    #print S
                    #add constraint
                    alpha = independence_number(h)
                    #print h.graph6_string(), alpha
                    LP.add_constraint(sum([b[j] for j in S]), max = alpha, name = h.graph6_string())
        i = i + 1

    #for c in LP.constraints():
        #print c

    # The .solve() functions returns the objective value
    LP.solve()

    #return LP

    b_sol = LP.get_values(b)
    return b_sol, sum(b_sol.values())


###several invariants and auxiliary functions related to the Independence Decomposition Theorem

#finds all vertices with weight 1 in some max weighted stable set with wieghts in {0,1,1/2}
#had problem that LP solver has small numerical errors, fixed with kludgy if condition
def find_stable_ones_vertices(g):
    F = []
    alpha_f = fractional_alpha(g)
    for v in g.vertices():
        gc = copy(g)
        gc.delete_vertices(closed_neighborhood(gc, v))
        alpha_f_prime = fractional_alpha(gc)
        if abs(alpha_f - alpha_f_prime - 1) < .01:
            F.append(v)
    return F

def find_max_critical_independent_set(g):
    S = find_stable_ones_vertices(g)
    H = g.subgraph(S)
    return H.independent_set()

def critical_independence_number(g):
    return len(find_max_critical_independent_set(g))

def card_independence_irreducible_part(g):
    return len(find_independence_irreducible_part(g))

def find_independence_irreducible_part(g):
    X = find_KE_part(g)
    SX = Set(X)
    Vertices = Set(g.vertices())
    return list(Vertices.difference(SX))

#returns KE part guaranteed by Independence Decomposition Theorem
def find_KE_part(g):
    return closed_neighborhood(g, find_max_critical_independent_set(g))

def card_KE_part(g):
    return len(find_KE_part(g))

def find_independence_irreducible_subgraph(g):
    return g.subgraph(find_independence_irreducible_part(g))

def find_KE_subgraph(g):
    return g.subgraph(find_KE_part(g))


#make invariant from property
def make_invariant_from_property(property, name=None):
    """
    This function takes a property as an argument and returns an invariant
    whose value is 1 if the object has the property, else 0
    Optionally a name for the new property can be provided as a second argument.
    """
    def boolean_valued_invariant(g):
        if property(g):
            return 1
        else:
            return 0

    if name is not None:
        boolean_valued_invariant.__name__ = name
    elif hasattr(property, '__name__'):
        boolean_valued_invariant.__name__ = '{}_value'.format(property.__name__)
    else:
        raise ValueError('Please provide a name for the new function')

    return boolean_valued_invariant

# defined by R. Pepper in an unpublished paper on graph irregularity
def geometric_length_of_degree_sequence(g):
    return sqrt(sum(d^2 for d in g.degree()))

efficiently_computable_invariants = [average_distance, Graph.diameter, Graph.radius,
Graph.girth,  Graph.order, Graph.size, Graph.szeged_index, Graph.wiener_index,
min_degree, max_degree, matching_number, residue, annihilation_number, fractional_alpha,
Graph.lovasz_theta, cvetkovic, cycle_space_dimension, card_center, card_periphery,
max_eigenvalue, kirchhoff_index, largest_singular_value, vertex_con, edge_con,
Graph.maximum_average_degree, Graph.density, welsh_powell, wilf, brooks,
different_degrees, szekeres_wilf, average_vertex_temperature, randic, median_degree,
max_even_minus_even_horizontal, fiedler, laplacian_energy, gutman_energy, average_degree,
degree_variance, number_of_triangles, graph_rank, inverse_degree, sum_temperatures,
card_positive_eigenvalues, card_negative_eigenvalues, card_zero_eigenvalues,
card_cut_vertices, Graph.clustering_average, Graph.connected_components_number,
Graph.spanning_trees_count, card_pendants, card_bridges, alon_spencer, caro_wei,
degree_sum, order_automorphism_group, sigma_2, brinkmann_steffen,
card_independence_irreducible_part, critical_independence_number, card_KE_part,
fractional_covering, eulerian_faces, barrus_q, mean_common_neighbors,
max_common_neighbors, min_common_neighbors, distinct_degrees, barrus_bound, geometric_length_of_degree_sequence]

intractable_invariants = [independence_number, domination_number, chromatic_index,
Graph.clique_number, clique_covering_number, n_over_alpha, chromatic_num,
independent_dominating_set_number]

#for invariants from properties and INVARIANT_PLUS see below

#FAST ENOUGH (tested for graphs on 140921): lovasz_theta, clique_covering_number, all efficiently_computable
#SLOW but FIXED for SpecialGraphs

invariants = efficiently_computable_invariants + intractable_invariants

#removed for speed: Graph.treewidth, card_max_cut

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


#add all properties derived from pairs of invariants
invariant_relation_properties = [has_leq_invariants(f,g) for f in invariants for g in invariants if f != g]


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
pairs_have_unique_common_neighbor, has_star_center, is_complement_of_chordal, has_c4, is_c4_free, is_subcubic]

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
invariants_plus = invariants + invariants_from_properties

# Graph.is_prime removed as faulty 9/2014
# built in Graph.is_transitively_reduced removed 9/2014
# is_line_graph is theoretically efficient - but Sage's implementation is not 9/2014

# weakly_chordal = weakly chordal, i.e., the graph and its complement have no induced cycle of length at least 5



#GRAPH OBJECTS

p3 = graphs.PathGraph(3)
p3.name(new = "p3")

k3_3=graphs.CompleteBipartiteGraph(3,3)
k3_3.name(new = "k3_3")

# The line graph of k3,3
k3_3_line_graph = k3_3.line_graph()
k3_3_line_graph.name(new = "k3_3 line graph")

k5_3=graphs.CompleteBipartiteGraph(5,3)
k5_3.name(new = "k5_3")

# independence_number(x) <= minimum(lovasz_theta(x), 2*e^sum_temperatures(x)) is false
#This is also a counterexample to independence_number(x) <= minimum(floor(lovasz_theta(x)), 2*e^sum_temperatures(x))
k1_9 = graphs.CompleteBipartiteGraph(1,9)
k1_9.name(new = "k1_9")

#two c4's joined at a vertex
c4c4=graphs.CycleGraph(4)
for i in [4,5,6]:
    c4c4.add_vertex()
c4c4.add_edge(3,4)
c4c4.add_edge(5,4)
c4c4.add_edge(5,6)
c4c4.add_edge(6,3)
c4c4.name(new="c4c4")

#two c5's joined at a vertex: eulerian, not perfect, not hamiltonian
c5c5=graphs.CycleGraph(5)
for i in [5,6,7,8]:
    c5c5.add_vertex()
c5c5.add_edge(0,5)
c5c5.add_edge(0,8)
c5c5.add_edge(6,5)
c5c5.add_edge(6,7)
c5c5.add_edge(7,8)
c5c5.name(new="c5c5")

#triangle plus pendant: not hamiltonian, not triangle-free
c3p2=graphs.CycleGraph(3)
c3p2.add_vertex()
c3p2.add_edge(0,3)
c3p2.name(new="c3p2")

K4a=graphs.CompleteGraph(4)
K4b=graphs.CompleteGraph(4)
K4a.delete_edge(0,1)
K4b.delete_edge(0,1)
regular_non_trans = K4a.disjoint_union(K4b)
regular_non_trans.add_edge((0,0),(1,1))
regular_non_trans.add_edge((0,1),(1,0))
regular_non_trans.name(new="regular_non_trans")

c6ee = graphs.CycleGraph(6)
c6ee.add_edges([(1,5), (2,4)])
c6ee.name(new="c6ee")

#c5 plus a chord
c5chord = graphs.CycleGraph(5)
c5chord.add_edge(0,3)
c5chord.name(new="c5chord")

#c6ee plus another chord: hamiltonian, regular, vertex transitive
c6eee = copy(c6ee)
c6eee.add_edge(0,3)
c6eee.name(new="c6eee")

#c8 plus one long vertical chord and 3 parallel horizontal chords
c8chorded = graphs.CycleGraph(8)
c8chorded.add_edge(0,4)
c8chorded.add_edge(1,7)
c8chorded.add_edge(2,6)
c8chorded.add_edge(3,5)
c8chorded.name(new="c8chorded")

#c8 plus 2 parallel chords: hamiltonian, tri-free, not vertex-transitive
c8chords = graphs.CycleGraph(8)
c8chords.add_edge(1,6)
c8chords.add_edge(2,5)
c8chords.name(new="c8chords")

#c8 plus 2 parallel chords: hamiltonian, tri-free, not vertex-transitive
c8chords = graphs.CycleGraph(8)
c8chords.add_edge(1,6)
c8chords.add_edge(2,5)
c8chords.name(new="c8chords")

prism = graphs.CycleGraph(6)
prism.add_edge(0,2)
prism.add_edge(3,5)
prism.add_edge(1,4)
prism.name(new="prism")

prismsub = copy(prism)
prismsub.subdivide_edge(1,4,1)
prismsub.name(new="prismsub")

# ham, not vertex trans, tri-free, not cartesian product
prismy = graphs.CycleGraph(8)
prismy.add_edge(2,5)
prismy.add_edge(0,3)
prismy.add_edge(4,7)
prismy.name(new="prismy")

#c10 with chords, ham, tri-free, regular, planar, vertex transitive
sixfour = graphs.CycleGraph(10)
sixfour.add_edge(1,9)
sixfour.add_edge(0,2)
sixfour.add_edge(3,8)
sixfour.add_edge(4,6)
sixfour.add_edge(5,7)
sixfour.name(new="sixfour")

#unique 24-vertex fullerene: hamiltonian, planar, not vertex transitive
c24 = Graph('WsP@H?PC?O`?@@?_?GG@??CC?G??GG?E???o??B???E???F')
c24.name(new="c24")

#unique 26-atom fullerene: hamiltonian, planar, not vertex trans, radius=5, diam=6
c26 = Graph('YsP@H?PC?O`?@@?_?G?@??CC?G??GG?E??@_??K???W???W???H???E_')
c26.name(new="c26")

"""
The Holton-McKay graph is the smallest planar cubic hamiltonian graph with an edge
that is not contained in a hamiltonian cycle. It has 24 vertices and the edges (0,3)
and (4,7) are not contained in a hamiltonian cycle. This graph was mentioned in
D. A. Holton and B. D. McKay, Cycles in 3-connected cubic planar graphs II, Ars
Combinatoria, 21A (1986) 107-114.

    sage: holton_mckay
    holton_mckay: Graph on 24 vertices
    sage: holton_mckay.is_planar()
    True
    sage: holton_mckay.is_regular()
    True
    sage: max(holton_mckay.degree())
    3
    sage: holton_mckay.is_hamiltonian()
    True
    sage: holton_mckay.radius()
    4
    sage: holton_mckay.diameter()
    6
"""
holton_mckay = Graph('WlCGKS??G?_D????_?g?DOa?C?O??G?CC?`?G??_?_?_??L')
holton_mckay.name(new="holton_mckay")

#z1 is a graph that shows up in a sufficient condition for hamiltonicity
z1 = graphs.CycleGraph(3)
z1.add_edge(0,3)
z1.name(new="z1")

#an example of a bipartite, 1-tough, not van_den_heuvel, not hamiltonian graph
kratsch_lehel_muller = graphs.PathGraph(12)
kratsch_lehel_muller.add_edge(0,5)
kratsch_lehel_muller.add_edge(6,11)
kratsch_lehel_muller.add_edge(4,9)
kratsch_lehel_muller.add_edge(1,10)
kratsch_lehel_muller.add_edge(2,7)
kratsch_lehel_muller.name(new="kratsch_lehel_muller")

#ham, not planar, not anti_tutte
c6xc6 = graphs.CycleGraph(6).cartesian_product(graphs.CycleGraph(6))
c6xc6.name(new="c6xc6")

#non-ham, 2-connected, eulerian (4-regular)
gould = Graph('S~dg?CB?wC_L????_?W?F??c?@gOOOGGK')
gould.name(new="gould")

#two k5s with single edge removed from each and lines joining these 4 points to a new center point, non-hamiltonian
throwing = Graph('J~wWGGB?wF_')
throwing.name(new="throwing")

#k4 plus k2 on one side, open k5 on other, meet at single point in center, non-hamiltonian
throwing2 = Graph("K~wWGKA?gB_N")
throwing2.name(new="throwing2")

#similar to throwing2 with pair of edges swapped, non-hamiltonian
throwing3 = Graph("K~wWGGB?oD_N")
throwing3.name(new="throwing3")

#graph has diameter != radius but is hamiltonian
tent = graphs.CycleGraph(4).join(Graph(1),labels="integers")
tent.name(new="tent")

#c6 with a k4 subgraph, eulerain, diameter = 3, radius=2, hamiltonian
c6subk4 = graphs.CycleGraph(6)
c6subk4.add_edge(1,5)
c6subk4.add_edge(1,4)
c6subk4.add_edge(2,5)
c6subk4.add_edge(2,4)
c6subk4.name(new="c6subk4")

#C5 with chords from one vertex to other 2 (showed up in auto search for CE's): hamiltonian
bridge = Graph("DU{")
bridge.name(new="bridge")

#nico found the smallest hamiltonian overfull graph
non_ham_over = Graph("HCQRRQo")
non_ham_over.name(new="non_ham_over")

ryan = Graph("WxEW?CB?I?_R????_?W?@?OC?AW???O?C??B???G?A?_??R")
ryan.name(new="ryan")

inp = Graph('J?`FBo{fdb?')
inp.name(new="inp")

#p10 joined to 2 points of k4, a CE to conjecture: chromatic_number<=avg degree + 1
p10k4=Graph('MhCGGC@?G?_@_B?B_')
p10k4.name(new="p10k4")

#star on 13 points with added edge: CE to alpha <+ dom + girth^2
s13e = Graph('M{aCCA?_C?O?_?_??')
s13e.name(new="s13e")

#rp CE to alpha<=2*chi+2*residue, has alpha=25,chi=2,residue=10
ryan2=graphs.CirculantGraph(50,[1,3])
ryan2.name(new="circulant_50_1_3")

#CE to alpha <= 2*girth^2+2, star with 22 rays plus extra edge
s22e = graphs.StarGraph(22)
s22e.add_edge(1,2)
s22e.name(new="s22e")

#the unique 100-atom fullerene with minimum independence number of 43 (and IPR, tetrahedral symmetry)
c100 = Graph("~?@csP@@?OC?O`?@?@_?O?A??W??_??_G?O??C??@_??C???G???G@??K???A????O???@????A????A?G??B?????_????C?G???O????@_?????_?????O?????C?G???@_?????E??????G??????G?G????C??????@???????G???????o??????@???????@????????_?_?????W???????@????????C????????G????????G?G??????E????????@_????????K?????????_????????@?@???????@?@???????@_?????????G?????????@?@????????C?C????????W??????????W??????????C??????????@?@?????????G???????????_??????????@?@??????????_???????????O???????????C?G??????????O???????????@????????????A????????????A?G??????????@_????????????W????????????@_????????????E?????????????E?????????????E?????????????B??????????????O?????????????A@?????????????G??????????????OG?????????????O??????????????GC?????????????A???????????????OG?????????????@?_?????????????B???????????????@_???????????????W???????????????@_???????????????F")
c100.name(new="c100")

dc64_g6string ="~?@?JXxwm?OJ@wESEYMMbX{VDokGxAWvH[RkTAzA_Tv@w??wF]?oE\?OAHoC_@A@g?PGM?AKOQ??ZPQ?@rgt??{mIO?NSD_AD?mC\
O?J?FG_FOOEw_FpGA[OAxa?VC?lWOAm_DM@?Mx?Y{A?XU?hwA?PM?PW@?G@sGBgl?Gi???C@_FP_O?OM?VMA_?OS?lSB??PS?`sU\
??Gx?OyF_?AKOCN`w??PA?P[J??@C?@CU_??AS?AW^G??Ak?AwVZg|?Oy_@?????d??iDu???C_?D?j_???M??[Bl_???W??oEV?\
???O??_CJNacABK?G?OAwP??b???GNPyGPCG@???"
dc64 = Graph(dc64_g6string)
dc64.name(new="dc64")

try:
    s = load(os.environ['HOME'] +'/objects-invariants-properties/dc1024_g6string.sobj')
    print "loaded graph dc1024"
    dc1024 = Graph(s)
    dc1024.name(new="dc1024")
except:
    print "couldn't load dc1024_g6string.sobj"

try:
    s = load(os.environ['HOME'] +'/objects-invariants-properties/dc2048_g6string.sobj')
    print "loaded graph dc2048"
    dc2048 = Graph(s)
    dc2048.name(new="dc2048")
except:
    print "couldn't load dc2048_g6string.sobj"

#graph from delavina's jets paper
starfish = Graph('N~~eeQoiCoM?Y?U?F??')
starfish.name(new="starfish")

#difficult graph from INP: order=11, alpha=4, best lower bound < 3
difficult11 = Graph('J?`FBo{fdb?')
difficult11.name(new="difficult11")

#c4 joined to K# at point: not KE, alpha=theta=nu=3, delting any vertex gives KE graph
c5k3=Graph('FheCG')
c5k3.name(new="c5k3")

#mycielskian of a triangle: CE to conj that chi <= max(clique, nu), chi=4, nu = clique = 3
c3mycielski = Graph('FJnV?')
c3mycielski.name(new="c3mycieski")

#4th mycielskian of a triangle, CE to conj chi <= clique + girth, chi = 7, clique = girth = 3
c3mycielski4 = Graph('~??~??GWkYF@BcuIsJWEo@s?N?@?NyB`qLepJTgRXkAkU?JPg?VB_?W[??Ku??BU_??ZW??@u???Bs???Bw???A??F~~_B}?^sB`o[MOuZErWatYUjObXkZL_QpWUJ?CsYEbO?fB_w[?A`oCM??DL_Hk??DU_Is??Al_Dk???l_@k???Ds?M_???V_?{????oB}?????o[M?????WuZ?????EUjO?????rXk?????BUJ??????EsY??????Ew[??????B`o???????xk???????FU_???????\\k????????|_????????}_????????^_?????????')
c3mycielski4.name(new="c3mycielski4")

# a PAW is a traingle with a pendant, same as a Z1
paw=Graph('C{')
paw.name(new="paw")

binary_octahedron = Graph('L]lw??B?oD_Noo')
#2 octahedrons, remove one edge from each, add vertex, connect it to deleted edge vertices
#its regular of degree 4
binary_octahedron.name(new = "binary_octahedron")

#this graph shows that the cartesian product of 2 KE graphs is not necessarily KE
# appears in Abay-Asmerom, Ghidewon, et al. "Notes on the independence number in the Cartesian product of graphs." Discussiones Mathematicae Graph Theory 31.1 (2011): 25-35.
paw_x_paw = paw.cartesian_product(paw)
paw_x_paw.name(new = "paw_x_paw")

#a KITE is a C4 with a chord
kite = Graph('Cn')
kite.name(new="kite")

#a DART is a kite with a pendant
dart = Graph('DnC')
dart.name(new="dart")

#P4 is a path on 4 vertices
p4=Graph('Ch')
p4.name(new="p4")

p5 = graphs.PathGraph(5)
p5.name(new = "p5")

"""
P29 is a CE to independence_number(x) <=degree_sum(x)/sqrt(card_negative_eigenvalues(x)) 
 and to  
<= max_degree(x)^e^card_center(x)
 and to
<= max_degree(x)^2 + card_periphery(x)
"""
p29 = graphs.PathGraph(29)
p29.name(new = "p29")

c9 = graphs.CycleGraph(9)
c9.name(new = "c9")

# CE to independence_number(x) <= (e^welsh_powell(x) - graph_rank(x))^2
c22 = graphs.CycleGraph(22)
c22.name(new = "c22")

# CE to independence_number(x) <= minimum(cvetkovic(x), 2*e^sum_temperatures(x)) 
c34 = graphs.CycleGraph(34)
c34.name(new = "c34")

ce3=Graph("Gr`HOk")
ce3.name(new = "ce3")
#ce3 is a ce to (((is_planar)&(is_regular))&(is_bipartite))->(has_residue_equals_alpha)

ce2=Graph("HdGkCA?")
#ce2 is a ce to ((is_chordal)^(is_forest))->(has_residue_equals_alpha)
ce2.name(new = "ce2")

c6 = graphs.CycleGraph(6)
c6.name(new = "c6")

ce4=Graph("G~sNp?")
ce4.name(new = "ce4")
#ce4 is a ce to ((~(is_planar))&(is_chordal))->(has_residue_equals_alpha)

ce5=Graph("X~}AHKVB{GGPGRCJ`B{GOO`C`AW`AwO`}CGOO`AHACHaCGVACG^")
ce5.name(new = "ce5")
#ce5 is a ce to: (((is_line_graph)&(is_cartesian_product))|(is_split))->(has_residue_equals_alpha)

ce6 = Graph("H??E@cN")
#ce6 is a ce to: (is_split)->((order_leq_twice_max_degree)&(is_chordal))
ce6.name(new = "ce6")

ce7 = Graph("FpGK?")
#counterexample to: (has_residue_equals_alpha)->((is_bipartite)->(order_leq_twice_max_degree))
ce7.name(new = "ce7")

ce8 = Graph('IxCGGC@_G')
#counterexample to: ((has_paw)&(is_circular_planar))->(has_residue_equals_alpha)
ce8.name(new = "ce8")

ce9 = Graph('IhCGGD?G?')
#counterexample to: ((has_H)&(is_forest))->(has_residue_equals_alpha)
ce9.name(new = "ce9")

ce10=Graph('KxkGGC@?G?o@')
#counterexample to: (((is_eulerian)&(is_planar))&(has_paw))->(has_residue_equals_alpha)
ce10.name(new = "ce10")

ce11 = Graph("E|OW")
#counterexample to: (has_alpha_residue_equal_two)->((is_perfect)|(is_regular))
ce11.name(new = "ce11")

ce12 = Graph("Edo_")
#counterexample to: (((is_cubic)&(is_triangle_free))&(is_H_free))->(has_residue_equals_two)
ce12.name(new = "ce12")

ce13 = Graph("ExOG")
#counterexample to: ((diameter_equals_twice_radius)&(is_claw_free))->(has_residue_equals_two)
ce13.name(new = "ce13")

ce14 = Graph('IhCGGC_@?')
#counterexample to: (~(matching_covered))->(has_residue_equals_alpha)
ce14.name(new = "IhCGGC_@?")

"""
CE to independence_number(x) <= 10^order_automorphism_group(x)

    sage: order(ce15)
    57
    sage: independence_number(ce15)
    25
"""
ce15 = Graph("x??C?O?????A?@_G?H??????A?C??EGo?@S?O@?O??@G???CO???CAC_??a?@G?????H???????????O?_?H??G??G??@??_??OA?OCHCO?YA????????A?O???G?O?@????OOC???_@??????MCOC???O_??[Q??@???????O??_G?P?GO@A?G_???A???A@??g???W???@CG_???`_@O??????@?O@?AGO?????C??A??F??????@C????A?E@L?????P@`??")
ce15.name(new = "ce15")

# CE to independence_number(x) <= 2*maximum(welsh_powell(x), max_even_minus_even_horizontal(x))
ce16 = Graph("mG???GP?CC?Aa?GO?o??I??c??O??G?ACCGW@????OC?G@?_A_W_OC@??@?I??O?_AC?Oo?E@_?O??I??B_?@_A@@@??O?OC?GC?CD?C___gAO?G??KOcGCiA??SC????GAVQy????CQ?cCACKC_?A?E_??g_AO@C??c??@@?pY?G?")
ce16.name(new = "ce16")

# CE to independence_number(x) >= 1/2*cvetkovic(x) 
ce17 = Graph("S??wG@@h_GWC?AHG?_gMGY_FaIOk@?C?S")
ce17.name(new = "ce17")

# CE to independence_number(x) >= matching_number - sigma_2
ce18 = Graph("cGO_?CCOB@O?oC?sTDSOCC@O???W??H?b???hO???A@CCKB??I??O??AO@CGA???CI?S?OGG?ACgQa_Cw^GP@AID?Gh??ogD_??dR[?AG?")
ce18.name(new = "ce18")

# CE to independence_number(x) <= maximum(max_even_minus_even_horizontal(x), radius(x)*welsh_powell(x))
ce19 = Graph('J?@OOGCgO{_')
ce19.name(new = "ce19")

# CE to independence_number(x) <= card_center(x) + max_even_minus_even_horizontal(x) + 1
ce20 = Graph('M?CO?k?OWEQO_O]c_')
ce20.name(new = "ce20")

# CE to independence_number(x) <= median_degree(x)^2 + card_periphery(x)
ce21 = Graph('FiQ?_')
ce21.name(new = "ce21")

# CE to independence_number(x) <= brinkmann_steffen(x) + max_even_minus_even_horizontal(x) + 1
ce22 = Graph('Ss?fB_DYUg?gokTEAHC@ECSMQI?OO?GD?')
ce22.name(new = "ce22")

# CE to independence_number(x) <= inverse_degree(x) + order_automorphism_group(x) + 1
ce23 = Graph("HkIU|eA")
ce23.name(new = "ce23")

# CE to independence_number(x) <= ceil(eulerian_faces(x)/diameter(x)) +max_even_minus_even_horizontal(x)
ce24 = Graph('JCbcA?@@AG?')
ce24.name(new = "ce24")

# CE to independence_number(x) <= floor(e^(maximum(max_even_minus_even_horizontal(x), fiedler(x))))
ce25 = Graph('OX??ZHEDxLvId_rgaC@SA')
ce25.name(new = "ce25")

# CE to independence_number(x) <= maximum(card_periphery(x), radius(x)*welsh_powell(x))
ce26 = Graph("NF?_?o@?Oa?BC_?OOaO")
ce26.name(new = "ce26")

# CE to independence_number(x) <= floor(average_distance(x)) + maximum(max_even_minus_even_horizontal(x), brinkmann_steffen(x))
ce27 = Graph("K_GBXS`ysCE_")
ce27.name(new = "ce27")

# CE to independence_number(x) <= minimum(annihilation_number(x), 2*e^sum_temperatures(x))
ce28 = Graph("g??O?C_?`?@?O??A?A????????C?????G?????????A@aA??_???G??GA?@????????_???GHC???CG?_???@??_??OB?C?_??????_???G???C?O?????O??A??????G??")
ce28.name(new = "ce28")

# CE to independence_number(x) <= maximum(2*welsh_powell(x), maximum(max_even_minus_even_horizontal(x), laplacian_energy(x)))
ce29 = Graph("P@g??BSCcIA???COcSO@@O@c")
ce29.name(new = "ce29")

# CE to independence_number(x) <= maximum(order_automorphism_group(x), 2*cvetkovic(x) - matching_number(x))
ce30 = Graph("G~q|{W")
ce30.name(new = "ce30")

# CE to independence_number(x) <= max_even_minus_even_horizontal(x) + min_degree(x) + welsh_powell(x)
ce31 = Graph("VP??oq_?PDOGhAwS??bSS_nOo?OHBqPi?I@AGP?POAi?")
ce31.name(new = "ce31")

# CE to independence_number(x) >= order(x)/szekeres_wilf(x)
ce32 = Graph('H?`@Cbg')
ce32.name(new = "ce32")

# CE to independence_number(x) <= max_even_minus_even_horizontal(x) + minimum(card_positive_eigenvalues(x), card_center(x) + 1)
ce33 = Graph("O_aHgP_kVSGOCXAiODcA_")
ce33.name(new = "ce33")

# CE to independence_number(x) <= card_center(x) + maximum(diameter(x), card_periphery(x))
ce34 = Graph('H?PA_F_')
ce34.name(new = "ce34")
 
# CE to independence_number(x) <= card_center(x) + maximum(diameter(x), card_periphery(x))ce35 = Graph("")
ce35 = Graph("HD`cgGO")
ce35.name(new = "ce35")

# CE to independence_number(x) >= max_degree(x) - order_automorphism_group(x)
ce36 = Graph('ETzw')
ce36.name(new = "ce36")

# CE to independence_number(x) <= maximum(card_center(x), diameter(x)*max_degree(x)) 
ce37 = Graph("~?AA?G?????@@??@?A???????????O??????????G_?A???????????????A?AO?????????G???G?@???@???O?????????????C???????_???????C?_?W???C????????_??????????????????_???????_???O????????D??????????C????????GCC???A??G??????A@??A??@G???_?????@_??????_??G???K??????A????C??????????A???_?A????`??C_O????G????????????A?G???????????????????O?????C??????@???__?@O_G??C????????OA?????????????????????????GA_GA????O???_??O??O?G??G?_C???@?G???O???_?O???_??????C???????????????E_???????????????_@???O??????CC???O?????????OC_????_A????????_?G??????O??????_??????_?I?O??????A???????O?G?O???C@????????????_@????C?????@@???????C???O??A?????_??????A_??????????A?G????AB???A??C?G??????????G???A??@?A???????@???????D?_????B????????????????????g?C???C????G????????@??????@??A????????@????_??_???o?????????@????????????_???????A??????C????A?????C????O????@?@???@?A_????????CA????????????????H???????????????????O????_??OG??Ec?????O??A??_???_???O?C??`?_@??@??????O????G????????????A????@???_?????????_?A???AAG???O????????????????????C???_???@????????????_??H???A??W?O@????@_???O?_A??O????OG???????G?@??G?C?????G?????????@?????????G?O?????G???????_?????????@????@?????????G????????????C?G?????????_C?@?A????G??GA@????????????@?????C??G??????_?????????_@?????@???A?????@?????????????????CG??????_?????@???????@C???O????_`?????OA?G??????????????Q?A?????????????A????@C?????GO??_?C???????O???????@?G?A????O??G???_????_?????A?G_?C?????????C?")
ce37.name(new = "ce37")

# CE to independence_number(x) <= abs(-card_center(x) + min_degree(x)) + max_even_minus_even_horizontal(x)
ce38 = Graph('FVS_O')
ce38.name(new = "ce38")

# CE to independence_number(x) <= abs(-card_center(x) + max_degree(x)) + max_even_minus_even_horizontal(x)
ce39 = Graph("FBAuo")
ce39.name(new = "ce39") 

# CE to independence_number(x) <= floor(inverse_degree(x)) + order_automorphism_group(x) + 1
ce40 = Graph('Htji~Ei')
ce40.name(new = "ce40")

# CE to independence_number(x) <= maximum(girth(x), card_center(x) + card_periphery(x)) 
ce41 = Graph("FhX?G")
ce41.name(new = "ce41")

# CE to independence_number(x) <= card_center(x) + maximum(residue(x), card_periphery(x))
ce42 = Graph('GP[KGC')
ce42.name(new = "ce42") 

# CE to independence_number(x) <= maximum(girth(x), (barrus_bound(x) - order_automorphism_group(x))^2) 
ce43 = Graph("Exi?")
ce43.name(new = "ce43")

# CE to independence_number(x) <= (brinkmann_steffen(x) - szekeres_wilf(x))^2 + max_even_minus_even_horizontal(x)
ce44 = Graph('GGDSsg')
ce44.name(new = "ce44")

# CE to independence_number(x) <= maximum(max_even_minus_even_horizontal(x), radius(x)*szekeres_wilf(x)) 
ce45 = Graph("FWKH?")
ce45.name(new = "ce45")

# CE to independence_number(x) <= maximum(card_periphery(x), radius(x)*szekeres_wilf(x))
ce46 = Graph('F`I`?')
ce46.name(new = "ce46")

# CE to independence_number(x) <= maximum(card_periphery(x), diameter(x) + inverse_degree(x)) 
ce47 = Graph("KVOzWAxewcaE")
ce47.name(new = "ce47")

# CE to independence_number(x) <= maximum(card_periphery(x), max_even_minus_even_horizontal(x) + min_degree(x))
ce48 = Graph('Iq]ED@_s?')
ce48.name(new = "ce48")

# CE to independence_number(x) >= sqrt(card_positive_eigenvalues(x)) 
ce49 = Graph("K^~lmrvv{~~Z")
ce49.name(new = "ce49")

# CE to  independence_number(x) <= max_degree(x) + maximum(max_even_minus_even_horizontal(x), sigma_2(x))
ce50 = Graph('bCaJf?A_??GY_O?KEGA???OMP@PG???G?CO@OOWO@@m?a?WPWI?G_A_?C`OIG?EDAIQ?PG???A_A?C??CC@_G?GDI]CYG??GA_A??')
ce50.name(new = "ce50")

# CE to independence_number(x) >= matching_number(x) - order_automorphism_group(x) - 1 
ce51 = Graph("Ivq~j^~vw")
ce51.name(new = "ce51")

# CE to independence_number(x) >= order(x)/szekeres_wilf(x)
ce52 = Graph('H?QaOiG')
ce52.name(new = "ce52")

# CE to independence_number(x) >= matching_number(x) - sigma_2(x) - 1 
ce53 = Graph("]?GEPCGg]S?`@??_EM@OTp?@E_gm?GW_og?pWO?_??GQ?A?^HIRwH?Y?__BC?G?[PD@Gs[O?GW")
ce53.name(new = "ce53")

# CE to independence_number(x) >= -average_distance(x) + ceil(lovasz_theta(x))
ce54 = Graph('lckMIWzcWDsSQ_xTlFX?AoCbEC?f^xwGHOA_q?m`PDDvicEWP`qA@``?OEySJX_SQHPc_H@RMGiM}`CiG?HCsm_JO?QhI`?ARLAcdBAaOh_QMG?`D_o_FvQgHGHD?sKLEAR^ASOW~uAUQcA?SoD?_@wECSKEc?GCX@`DkC')
ce54.name(new = "ce54")

# CE to independence_number(x) >= -card_periphery(x) + matching_number(x) 
ce55 = Graph("I~~~~~~zw")
ce55.name(new = "ce55")

# CE to independence_number(x) >= lovasz_theta(x)/edge_con(x)
ce56 = Graph('HsaGpOe')
ce56.name(new = "ce56")

# CE to independence_number(x) >= minimum(max_degree(x), floor(lovasz_theta(x))) 
ce57 = Graph("^?H{BDHqHosG??OkHOhE??B[CInU?@j_A?CoA^azGPLcb_@GEYYRPgG?K@gdPAg?d@_?_sGcED`@``O")
ce57.name(new = "ce57")

# CE to independence_number>= barrus_bound(x) - max(card_center(x), card_positive_eigenvalues(x))
ce58 = Graph('Sj[{Eb~on~nls~NJWLVz~~^|{l]b\uFss')
ce58.name(new = "ce58")

# CE to independence_number(x) >= floor(tan(barrus_bound(x) - 1))
ce59 = Graph("RxCWGCB?G?_B?@??_?N??F??B_??w?")
ce59.name(new = "ce59")

# CE to independence_number(x) >= -1/2*diameter(x) + lovasz_theta(x)
ce60 = Graph('wSh[?GCfclJm?hmgA^We?Q_KIXbf\@SgDNxpwHTQIsIB?MIDZukArBAeXE`vqDLbHCwf{fD?bKSVLklQHspD`Lo@cQlEBFSheAH?yW\YOCeaqmOfsZ?rmOSM?}HwPCIAYLdFx?o[B?]ZYb~IK~Z`ol~Ux[B]tYUE`_gnVyHRQ?{cXG?k\BL?vVGGtCufY@JIQYjByg?Q?Qb`SKM`@[BVCKDcMxF|ADGGMBW`ANV_IKw??DRkY\KOCW??P_?ExJDSAg')
ce60.name(new = "ce60")

# CE to independence_number(x) <= maximum(card_negative_eigenvalues(x), max_common_neighbors(x) + max_even_minus_even_horizontal(x))
ce61 = Graph("KsaAA?OOC??C")
ce61.name(new = "ce61")
 
# CE to independence_number(x) >= minimum(floor(lovasz_theta(x)), tan(spanning_trees_count(x)))
ce62 = Graph("qWGh???BLQcAH`aBAGCScC@SoBAAFYAG?_T@@WOEBgRC`oSE`SG@IoRCK[_K@QaQq?c@?__G}ScHO{EcCa?K?o?E?@?C[F_@GpV?K_?_?CSW@D_OCr?b_XOag??C@gGOGh??QFoS?@OHDAKWIX_OBbHGOl??\Cb@?E`WehiP@IGAFC`GaCgC?JjQ???AGJgDJAGsdcqEA_a_q?")
ce62.name(new = "ce62")

# CE to independence_number(x) >= diameter(x)/different_degrees(x) 
ce63 = Graph("KOGkYBOCOAi@")
ce63.name(new = "ce63")

# CE to independence_number(x) >= -max_common_neighbors(x) + min_degree(x)
ce64 = Graph('`szvym|h~RMQLTNNiZzsgQynDR\p~~rTZXi~n`kVvKolVJfP}TVEN}Thj~tv^KJ}D~VqqsNy|NY|ybklZLnz~TfyG')
ce64.name(new = "ce64")

# CE to independence_number(x) >= -10^different_degrees(x) + matching_number(x) 
ce65 = Graph("W~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
ce65.name(new = "ce65")

# CE to independence_number(x) >= girth^max_degree+1
ce66 = Graph("~?@EG??????????@G????_???a???C????????@???A???????G??????C?GCG????????A???C@??????@????O??A??C?????_??O???CA???c??_?_?@????A????@??????C???C?G?O?C???G?????????O?_G?C????G??????_?????@??G???C??????O?GA?????O???@????????A?G?????????_C???????@??G??@??_??IA@???????G?@??????@??_?@????C??G???_????O???P???@???o??????O?????S?O???A???G?????c_?????D?????A???A?????G@???????O???H????O????@@????@K????????C??C?????G??")
ce66.name(new = "ce66")

# CE to independence_number(x) <= maximum(cycle_space_dimension(x), floor(lovasz_theta(x))) 
ce67 = Graph("G??EDw")
ce67.name(new = "ce67")

# CE to independence_number(x) >= minimum(card_positive_eigenvalues(x), 2*card_zero_eigenvalues(x))
ce68 = Graph('HzzP|~]')
ce68.name(new = "ce68")

# CE to independence_number(x) <= maximum(max_degree(x), radius(x)^card_periphery(x)) 
ce69 = Graph("F?BvO")
ce69.name(new = "ce69")

# CE to independence_number(x) >= floor(lovasz_theta(x))/vertex_con(x)
ce70 = Graph('~?@Z??????O?M??`S??A?`?A?????@????`?????A?A?????A@????GO?@@??A_????????O_???I@_??G??A?`?C????????@???????????@??C?@?????O??@??CA??A?D??G?_?????_Q@G????C?_?A??@???O????G?O?G?_?????CoG?G???X??C???_CAG_C??????G?????@?Ao?????C???A??????_??SG??cOC??????????Ao????????_?????G???????D?????C??_?B?????a??_???????G?@?????C??????C?c?????G_?_??G??_Q????C????B?_CG????AGC???G?O??_I????@??????_??a??@?O_G??O??aA@@?????EA???@???????@???????O?O??@??`_G???????GCA?_GO????_?_????????????_??I?@?C???@????????G?aG??????W????@PO@???oC?CO???_??G?@@?CO??K???C@??O???@????D?????A?@G?G?O???_???????Ao??AC???G?_???G????????A??????_?p???W?A?Ao@?????_?????GA??????????????_?C??????@O????_@??O@Gc@??????????A_??????')
ce70.name(new = "ce70")

# CE to independence_number(x) <= maximum(matching_number(x), critical_independence_number(x))
ce71 = Graph('ECYW')
ce71.name(new = "ce71")

# CE to independence_number(x)>=-1/2*x.diameter() + x.lovasz_theta()
ce72 = Graph('fdSYkICGVs_m_TPs`Fmj_|pGhC@@_[@xWawsgEDe_@g`TC{P@pqGoocqOw?HBDS[R?CdG\e@kMCcgqr?G`NHGXgYpVGCoJdOKBJQAsG|ICE_BeMQGOwKqSd\W?CRg')
ce72.name(new = "ce72")

# CE to independence_number(x) >= minimum(floor(lovasz_theta(x)), max_even_minus_even_horizontal(x) + 1)
ce73 = Graph('h???_?CA?A?@AA????OPGoC@????A@?A?_C?C?C_A_???_??_G????HG????c?G_?G??HC??A@GO?G?A@A???_@G_?_G_GC_??E?O?O`??@C?@???O@?AOC?G?H??O?P??C_?O_@??')
ce73.name(new = "ce73")

# CE to independence_number(x) >= minimum(diameter(x), lovasz_theta(x))
ce74 = Graph("FCQb_")
ce74.name(new = "ce74")

# CE to independence_number(x) >= minimum(girth(x), floor(lovasz_theta(x)))
ce75 = Graph('E?Bw')
ce75.name(new = "ce75")

# CE to independence_number(x) <= maximum(average_distance(x), max_even_minus_even_horizontal(x))*sum_temperatures(x)
ce76 = Graph("~?@DS?G???G_?A_?OA?GC??oa?A@?@?K???L?_?S_??CCSA_g???@D?????_?A??EO??GAOO_@C`???O?_CK_???_o_?@O??XA???AS???oE`?A?@?CAa?????C?G??i???C@qo?G?Og?_O?_?@???_G????o?A_@_?O?@??EcA???__?@GgO?O@oG?C?@??CIO?_??G??S?A?@oG_K?@C??@??QOA?C????AOo?p?G???oACAOAC@???OG??qC???C??AC_G?@??GCHG?AC@?_@O?CK?@?B???AI??OO_S_a_O??????AO?OHG?@?????_???EGOG??@?EF@?C?Pc?????C?W_PA?O@?_?@A@??OD_C?@?@?A??CC?_?i@?K?_O_CG??A?")
ce76.name(new = "ce76")

# CE to independence_number(x) <= maximum(matching_number(x), critical_independence_number(x)) 
ce77 = Graph("iF\ZccMAoW`Po_E_?qCP?Ag?OGGOGOS?GOH??oAAS??@CG?AA?@@_??_P??G?SO?AGA??M????SA????I?G?I???Oe?????OO???_S?A??A????ECA??C?A@??O??S?@????_@?_??S???O??")
ce77.name(new = "ce77")

# CE to independence_number(x) <= maximum(max_degree(x), radius(x)^card_periphery(x))
ce78 = Graph("G_aCp[")
ce78.name(new = "ce78")

# CE to independence_number(x) <= residue(x)^2
ce79 = Graph('J?B|~fpwsw_')
ce79.name(new = "ce79")

# CE to independence_number(x) <= 10^(card_center(x)*log(10)/log(sigma_2(x)))
ce80 = Graph('T?????????????????F~~~v}~|zn}ztn}zt^')
ce80.name(new = "ce80")

# CE to independence_number(x) <= diameter(x)^card_periphery(x)
ce81 = Graph('P?????????^~v~V~rzyZ~du{')
ce81.name(new = "ce81")

# CE to independence_number(x) <= radius(x)*residue(x) + girth(x)
ce82 = Graph('O????B~~^Zx^wnc~ENqxY')
ce82.name(new = "ce82")

# CE to independence_number(x) <= minimum(lovasz_theta(x), residue(x)^2)
#independence_number(x) <= minimum(annihilation_number(x), residue(x)^2)
#independence_number(x) <= minimum(fractional_alpha(x), residue(x)^2)
#independence_number(x) <= minimum(cvetkovic(x), residue(x)^2)
#independence_number(x) <= minimum(residue(x)^2, floor(lovasz_theta(x)))
#independence_number(x) <= minimum(size(x), residue(x)^2)
ce83 = Graph('LEYSrG|mrQ[ppi')
ce83.name(new = "ce83")

# CE to independence_number(x) <= maximum(laplacian_energy(x), brinkmann_steffen(x)^2)
ce84 = Graph('~?@r?A??OA?C??????@?A????CC?_?A@????A?@???@?S?O????AO??????G???????C????????C?C???G?????_??????_?G?????O?A?_?O?O@??O???T@@??????O????C_???C?CO???@??@?@???_???O??O??A??O???O?A?OB?C?AD???C`?B?__?_????????Q?C??????????????_???C??_???A?gO??@C???C?EC?O??GG`?O?_?_??O????_?@?GA?_????????????G????????????????????AO_?C?????????P?IO??I??OC???O????A??AC@AO?o????????o@??O?aI?????????_A??O??G??o?????????_??@?????A?O?O?????G?????H???_????????A??a?O@O?_?D???????O@?????G???GG?CA??@?A@?A????GA?@???G??O??A??????AA???????O??_c??@???A?????_????@CG????????????A???A???????A?W???B????@?????HGO???????_@_?????C??????????_a??????_???????@G?@O?@@_??G@???????GG?O??A??????@????_??O_?_??CC?B???O??@????W??`AA????O??_?????????????????_???A??????@G??????I@C?G????????A@?@@?????C???p???????????????????G?_G????Z?A????_??????G????Q????@????????_@O????@???_QC?A??@???o???G???@???????O???CC??O?D?O?@C????@O?G?????A??@C???@????O?????????_??C??????_?@????O??????O?Y?C???_?????A??@OoG???????A???G??????CC??A?A?????????????????GA_???o???G??O??C???_@@??????@?????G??????????O???@O???????????A????S??_o????????A??B??????_??C????C?')
ce84.name(new = "ce84")

# CE to independence_number(x) <= girth(x)^ceil(laplacian_energy(x))
ce85 = Graph('bd_OPG_J_G?apBB?CPk@`X?hB_?QKEo_op`C?|Gc?K_?P@GCoGPTcGCh?CBIlqf_GQ]C_?@jlFP?KSEALWGi?bIS?PjO@?CCA?OG?')
ce85.name(new = "ce85")

# CE to independence_number(x) <= diameter(x)*residue(x) + different_degrees(x)
ce86 = Graph('SK|KWYc|^BJKlaCnMH^ECUoSC[{LHxfMG')
ce86.name(new = "ce86")

# CE to independence_number(x) <= maximum(max_common_neighbors(x), girth(x)^laplacian_energy(x))
ce87 = Graph('~?@iA?B@@b?[a??oHh_gC?@AGD?Wa???_E@_@o_AkGA@_o?_h??GG??cO??g?_?SD?d@IW?s?P_@@cSG?_B??d?CSI?OCQOH_?bo?CAaC???pGC@DO@?PHQOpO??A?_A@K[PC@?__???@OSCOGLO?oOAAA?IOX@??GC?O?P??oA_?KPIK?Q@A?sQC???LA???aQOC_AeG?Q?K_Oo?AB?OU?COD?VoQ?@D????A?_D?CAa?@@G?C??CGHcCA_cB?@c@_O?H??_@?@OWGGCo??AGC??AQ?QOc???Ow_?C[?O@@G_QH?H?O???_I@@PO????FAGk??C?ka@D@I?P?CooC@_O@?agAE??CpG?AA_`OO??_?Q?AiOQEK?GhB@CAOG?G?CC??C@O@GdC__?OIBKO?aOD_?OG???GACH@?b?@?B_???WPA?@_?o?XQQ?ZI_@?O_o_?@O??EDGOBEA??_aOSQsCO@?_DD`O??D?JaoP?G?AOQOCAS?k??S?c@?XW?QCO??_OAGOWc__G?_??G??L@OP?b?O?GCCMAH????????@@?A?C@oDaGG?Wk@H@OM?_A?IOu`SG?E@??W?I@EQA@@_@Wa?@?_??C??AAAiGQG@@?`@oA?_??OgC?K_G??G`?@S@B?A?HWc?HG??`gO???A?W?A?O?MpS??D?GS?GDC_??I@??IPAOdk`?CG??A?pPAgIDlCYCTSDgg?@FW?DI?O_OW?_S??AAQB_OOCF????XS_?@l_kAw__Ea?O?C_CGO??EG??WLb@_H??OCaAET@S?@?I???_??LaO_HCYG@G_G?_?_C???os?_G?OO@s_??_?_GGE`Os??_GCa?DWO?A@?@_CB`MOBCGIC???GKA_c?@BSh??@?RC[?eg?@hOC?_?BeGOaC?AWOSCm@G?A??A?G?Ga_')
ce87.name(new = "ce87")

# CE to independence_number(x) <= maximum(radius(x), max_degree(x))^2
ce88 = Graph('h@`CA???GH?AAG?OW@@????E???O?O???PO?O?_?G??`?O_???@??E?E??O??A?S@???S???????U?GAI???A?DA??C?C@??PA?A???_C_?H?AA??_C??DCO?C???_?AAG??@O?_?G')
ce88.name(new = "ce88")

# CE to independence_number(x) <= max_degree(x) + maximum(max_even_minus_even_horizontal(x), geometric_length_of_degree_sequence(x))
ce89 = Graph("_qH?S@??`??GG??O?_?C?_??@??@??G??C??_??C????O??G???@????O???A???@????C???C?????G????")
ce89.name(new = "ce89")

# CE to independence_number(x) <= maximum(2*welsh_powell(x), max_even_minus_even_horizontal(x)^2)  
ce91 = Graph("q?}x{k\FGNCRacDO`_gWKAq?ED?Qc?IS?Da?@_E?WO_@GOG@B@?Cc?@@OW???qO?@CC@?CA@C?E@?O?KK???E??GC?CO?CGGI??@?cGO??HG??@??G?SC???AGCO?KAG???@O_O???K?GG????WCG??C?C??_C????q??@D??AO???S????CA?a??A?G??IOO????B?A???_??")
ce91.name(new = "ce91")

#a K5 with a pendant, CE to dirac => regular or planar conjecture
k5pendant = Graph('E~}?')
k5pendant.name(new="k5pendant")

#same as H
killer = Graph('EgSG')
killer.name(new="killer")

#alon_seymour graph: CE to the rank-coloring conjecture, 56-regular, vertex_trans, alpha=2, omega=22, chi=chi'=edge_connect=56
alon_seymour=Graph([[0..63], lambda x,y : operator.xor(x,y) not in (0,1,2,4,8,16,32,63)])
alon_seymour.name(new="alon_seymour")

k3 = graphs.CompleteGraph(3)
k3.name(new="k3")

k4 = graphs.CompleteGraph(4)
k4.name(new="k4")

k5 = graphs.CompleteGraph(5)
k5.name(new="k5")

k6 = graphs.CompleteGraph(6)
k6.name(new="k6")

# CE to independence_number(x) >= floor(tan(floor(gutman_energy(x))))
k37 = graphs.CompleteGraph(37)
k37.name(new = "k37")

c4 = graphs.CycleGraph(4)
c4.name(new="c4")

# CE to independence_number(x) <= residue(x)^(degree_sum(x)^density(x))
c102 = graphs.CycleGraph(102)
c102.name(new = "c102")

p2 = graphs.PathGraph(2)
p2.name(new="p2")

p6 = graphs.PathGraph(6)
p6.name(new="p6")

"""
CE to independence_number(x) <= e^(cosh(max_degree(x) - 1))
 and to
independence_number(x) <= max_degree(x)*min_degree(x) + card_periphery(x)
"""
p9 = graphs.PathGraph(9)
p9.name(new = "p9")

# CE to independence_number(x) <= 2*cvetkovic(x)*log(10)/log(x.size())
p102 = graphs.PathGraph(102)
p102.name(new = "p102")

#star with 3 rays, order = 4
k1_3 = graphs.StarGraph(3)
k1_3.name(new="k1_3")

k10 = graphs.CompleteGraph(10)
k10.name(new="k10")

c60 = graphs.BuckyBall()
c60.name(new="c60")

#moser spindle
moser = Graph('Fhfco')
moser.name(new = "moser")

#Holt graph is smallest graph which is edge-transitive but not arc-transitive
holt = graphs.HoltGraph()
holt.name(new = "holt")

golomb = Graph("I?C]dPcww")
golomb.name(new = "golomb")

edge_critical_5=graphs.CycleGraph(5)
edge_critical_5.add_edge(0,3)
edge_critical_5.add_edge(1,4)
edge_critical_5.name(new="edge_critical_5")

#a CE to alpha >= min{e-n+1,diameter}
heather = graphs.CompleteGraph(4)
heather.add_vertex()
heather.add_vertex()
heather.add_edge(0,4)
heather.add_edge(5,4)
heather.name(new="heather")

pete = graphs.PetersenGraph()

#residue = alpha = 3, a CE to conjecture that residue=alpha => is_ore
ryan3=graphs.CycleGraph(15)
for i in range(15):
    for j in [1,2,3]:
        ryan3.add_edge(i,(i+j)%15)
        ryan3.add_edge(i,(i-j)%15)
ryan3.name(new="ryan3")

#sylvester graph: 3-reg, 3 bridges, no perfect matching (why Petersen theorem requires no more than 2 bridges)
sylvester = Graph('Olw?GCD@o??@?@?A_@o`A')
sylvester.name(new="sylvester")

fork=graphs.PathGraph(4)
fork.add_vertex()
fork.add_edge(1,4)
fork.name(new="fork")

#one of the 2 order 11 chromatic edge-critical graphs discovered by brinkmann and steffen
edge_critical_11_1 = graphs.CycleGraph(11)
edge_critical_11_1.add_edge(0,2)
edge_critical_11_1.add_edge(1,6)
edge_critical_11_1.add_edge(3,8)
edge_critical_11_1.add_edge(5,9)
edge_critical_11_1.name(new="edge_critical_11_1")

#one of the 2 order 11 chromatic edge-critical graphs discovered by brinkmann and steffen
edge_critical_11_2 = graphs.CycleGraph(11)
edge_critical_11_2.add_edge(0,2)
edge_critical_11_2.add_edge(3,7)
edge_critical_11_2.add_edge(6,10)
edge_critical_11_2.add_edge(4,9)
edge_critical_11_2.name(new="edge_critical_11_2")

#chromatic_index_critical but not overfull
pete_minus=graphs.PetersenGraph()
pete_minus.delete_vertex(9)
pete_minus.name(new="pete_minus")

bow_tie = Graph(5)
bow_tie.add_edge(0,1)
bow_tie.add_edge(0,2)
bow_tie.add_edge(0,3)
bow_tie.add_edge(0,4)
bow_tie.add_edge(1,2)
bow_tie.add_edge(3,4)
bow_tie.name(new = "bow_tie")



"""
The Haemers graph was considered by Haemers who showed that alpha(G)=theta(G)<vartheta(G).
The graph is a 108-regular graph on 220 vertices. The vertices correspond to the 3-element
subsets of {1,...,12} and two such vertices are adjacent whenever the subsets
intersect in exactly one element.

    sage: haemers
    haemers: Graph on 220 vertices
    sage: haemers.is_regular()
    True
    sage: max(haemers.degree())
    108
"""
haemers = Graph([Subsets(12,3), lambda s1,s2: len(s1.intersection(s2))==1])
haemers.relabel()
haemers.name(new="haemers")

"""
The Pepper residue graph was described by Ryan Pepper in personal communication.
It is a graph which demonstrates that the residue is not monotone. The graph is
constructed by taking the complete graph on 3 vertices and attaching a pendant
vertex to each of its vertices, then taking two copies of this graph, adding a
vertex and connecting it to all the pendant vertices. This vertex has degree
sequence [6, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2] which gives residue equal to 4.
By removing the central vertex with degree 6, you get a graph with degree
sequence [3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1] which has residue equal to 5.

    sage: pepper_residue_graph
    pepper_residue_graph: Graph on 13 vertices
    sage: sorted(pepper_residue_graph.degree(), reverse=True)
    [6, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2]
    sage: residue(pepper_residue_graph)
    4
    sage: residue(pepper_residue_graph.subgraph(vertex_property=lambda v:pepper_residue_graph.degree(v)<6))
    5
"""
pepper_residue_graph = graphs.CompleteGraph(3)
pepper_residue_graph.add_edges([(i,i+3) for i in range(3)])
pepper_residue_graph = pepper_residue_graph.disjoint_union(pepper_residue_graph)
pepper_residue_graph.add_edges([(0,v) for v in pepper_residue_graph.vertices() if pepper_residue_graph.degree(v)==1])
pepper_residue_graph.relabel()
pepper_residue_graph.name(new="pepper_residue_graph")

"""
The Barrus graph was suggested by Mike Barrus in "Havel-Hakimi residues of Unigraphs" (2012) as an example of a graph whose residue (2) is
less than the independence number of any realization of the degree sequence. The degree sequence is [4^8,2].
The realization is the one given by reversing the Havel-Hakimi process.

    sage: barrus_graph
    barrus_graph: Graph on 9 vertices
    sage: residue(barrus_graph)
    2
    sage: independence_number(barrus_graph)
    3
"""
barrus_graph = Graph('HxNEG{W')
barrus_graph.name(new = "barrus_graph")

#CE to conjecture: (is_split)->((is_eulerian)->(is_regular))
#split graph from k4 and e2 that is eulerian but not regular
k4e2split = graphs.CompleteGraph(4)
k4e2split.add_vertices([4,5])
k4e2split.add_edge(4,0)
k4e2split.add_edge(4,1)
k4e2split.add_edge(5,2)
k4e2split.add_edge(5,3)
k4e2split.name(new = "k4e2split")

houseX=graphs.HouseXGraph()
houseX.name(new = "houseX")

triangle_star = Graph("H}qdB@_")
#a counterexample to: (has_residue_equals_alpha)->((is_eulerian)->(alpha_leq_order_over_two))
triangle_star.name(new = "triangle_star")

#flower with n petals
def flower(n):
    g = graphs.StarGraph(2*n)
    for x in range(n):
        v = 2*x+1
        g.add_edge(v,v+1)
    return g

flower_with_3_petals = flower(3)
flower_with_3_petals.name(new = "flower_with_3_petals")

flower_with_4_petals = flower(4)
flower_with_4_petals.name(new = "flower_with_4_petals")

"""
Non-perfect, alpha = 2, order = 6

    sage: pepper_non_perfect_graph.is_perfect()
    false
    sage: independence_number(pepper_non_perfect_graph)
    2
    sage: pepper_non_perfect_graph.order()
    6
"""
pepper_non_perfect_graph = Graph("EdZG")
pepper_non_perfect_graph.name(new = "pepper_non_perfect")

# Gallai Tree graph
gallai_tree = Graph("`hCKGC@?G@?K?@?@_?w?@??C??G??G??c??o???G??@_??F???N????_???G???B????C????W????G????G????C")
gallai_tree.name(new = "gallai_tree")

# Trigonal Antiprism w/ capped top face
trig_antiprism_capped = Graph("Iw?EthkF?")
trig_antiprism_capped.name(new = "trig_antiprism_capped")

"""
From Willis's thesis, page 4
Alpha = Fractional Alpha = 4
    
    sage: independence_number(willis_page4)
    4
    sage: fractional_alpha(willis_page4)
    4
"""
willis_page4 = Graph("GlCKIS")
willis_page4.name(new = "willis_page4")

"""
From Willis's thesis, page 13, Fig. 2.7

    sage: independence_number(willis_page13_fig27)
    4
    sage: willis_page13_fig27.order()
    7
    sage: willis_page13_fig27.size()
    15
"""
willis_page13_fig27 = Graph("Fs\zw")
willis_page13_fig27.name(new = "willis_page13_fig27")

"""
From Willis's thesis, page 9, Figure 2.1
Graph for which the Cvetkovic bound is the best upper bound present in the thesis

    sage: independence_number(willis_page9)
    3
    sage: willis_page9.order()
    8
    sage: willis_page9.size()
    12
    sage: max_degree(willis_page9)
    3
    sage: min_degree(willis_page9)
    3
"""
willis_page9 = Graph("GrCkQK")
willis_page9.name(new = "willis_page9")

"""
From Willis's thesis, page 10, Figure 2.2
Graph for which the Cvetkovic bound is the best upper bound present in the thesis

    sage: independence_number(willis_page10_fig23)
    4
    sage: willis_page10_fig23.order()
    10
    sage: willis_page10_fig23.size()
    15
    sage: max_degree(willis_page10_fig23)
    3
    sage: min_degree(willis_page10_fig23)
    3
"""
willis_page10_fig23 = Graph("G|eKHw")
willis_page10_fig23.name(new = "willis_page10_fig23")

"""
From Willis's thesis, page 10, Figure 2.4
Graph for which the Cvetkovic bound is the best upper bound present in the thesis

    sage: independence_number(willis_page10_fig24)
    9
    sage: willis_page10_fig24.order()
    24
    sage: willis_page10_fig24.size()
    36
    sage: max_degree(willis_page10_fig24)
    3
    sage: min_degree(willis_page10_fig24)
    3
"""
willis_page10_fig24 = Graph("WvOGWK@?G@_B???@_?O?F?????G??W?@K_?????G??@_?@B")
willis_page10_fig24.name(new = "willis_page10_fig24")

"""
From Willis's thesis, page 13, Figure 2.6
Graph for which the fractional independence bound is the best upper bound present in the thesis

    sage: independence_number(willis_page13_fig26)
    3
    sage: willis_page13_fig26.order()
    7
    sage: willis_page13_fig26.size()
    12
    sage: max_degree(willis_page13_fig26)
    4
    sage: min_degree(willis_page13_fig26)
    3
"""
willis_page13_fig26 = Graph("FstpW")
willis_page13_fig26.name(new = "willis_page13_fig26")

"""
From Willis's thesis, page 18, Figure 2.9
Graph for which the minimum degree  is the best upper bound present in the thesis

    sage: independence_number(willis_page18)
    2
    sage: willis_page18.order()
    6
    sage: willis_page18.size()
    12
    sage: max_degree(willis_page18)
    4
    sage: min_degree(willis_page18)
    4
"""
willis_page18 = Graph("E}lw")
willis_page18.name(new = "willis_page18")

"""
From Willis's thesis, page 21, Figure 3.1
Graph for which n/chi is the best lower bound present in the thesis

    sage: independence_number(willis_page21)
    4
    sage: willis_page21.order()
    12
    sage: willis_page21.size()
    20
    sage: max_degree(willis_page21)
    4
    sage: chromatic_num(willis_page21)
    3
"""
willis_page21 = Graph("KoD?Xb?@HBBB")
willis_page21.name(new = "willis_page21")

"""
From Willis's thesis, page 25, Figure 3.2
Graph for which residue is the best lower bound present in the thesis

    sage: independence_number(willis_page25_fig32)
    3
    sage: willis_page25_fig32.order()
    8
    sage: willis_page25_fig32.size()
    15
    sage: max_degree(willis_page25_fig32)
    6
    sage: chromatic_num(willis_page25_fig32)
    4
"""
willis_page25_fig32 = Graph("G@N@~w")
willis_page25_fig32.name(new = "willis_page25_fig32")

"""
From Willis's thesis, page 25, Figure 3.3
Graph for which residue is the best lower bound present in the thesis

    sage: independence_number(willis_page25_fig33)
    4
    sage: willis_page25_fig33.order()
    14
    sage: willis_page25_fig33.size()
    28
    sage: max_degree(willis_page25_fig33)
    4
    sage: chromatic_num(willis_page25_fig33)
    4
"""
willis_page25_fig33 = Graph("Mts?GKE@QDCIQIKD?")
willis_page25_fig33.name(new = "willis_page25_fig33")

# The Lemke Graph
lemke = Graph("G_?ztw")
lemke.name(new = "Lemke")

"""
From Willis's thesis, page 29, Figure 3.6
Graph for which the Harant Bound is the best lower bound present in the thesis

    sage: independence_number(willis_page29)
    4
    sage: willis_page29.order()
    14
    sage: willis_page29.size()
    28
    sage: max_degree(willis_page29)
    4
    sage: chromatic_num(willis_page29)
    4
"""
willis_page29 = Graph("[HCGGC@?G?_@?@_?_?M?@o??_?G_?GO?CC?@?_?GA??_C?@?C?@?A??_?_?G?D?@")
willis_page29.name(new = "willis_page29")

"""
From Willis's thesis, page 35, Figure 5.1
A graph where none of the upper bounds in the thesis give the exact value for alpha

    sage: independence_number(willis_page35_fig51)
    2
    sage: willis_page35_fig51.order()
    10
"""
willis_page35_fig51 = Graph("I~rH`cNBw")
willis_page35_fig51.name(new = "willis_page35_fig51")

"""
From Willis's thesis, page 35, Figure 5.2
A graph where none of the upper bounds in the thesis give the exact value for alpha

    sage: independence_number(willis_page35_fig52)
    2
    sage: willis_page35_fig52.order()
    10
"""
willis_page35_fig52 = Graph("I~zLa[vFw")
willis_page35_fig52.name(new = "willis_page35_fig52")

"""
From Willis's thesis, page 36, Figure 5.3
A graph where none of the upper bounds in the thesis give the exact value for alpha

    sage: independence_number(willis_page36_fig53)
    4
    sage: willis_page36_fig53.order()
    11
"""
willis_page36_fig53 = Graph("JscOXHbWqw?")
willis_page36_fig53.name(new = "willis_page36_fig53")

"""
From Willis's thesis, page 36, Figure 5.4
A graph where none of the upper bounds in the thesis give the exact value for alpha

    sage: independence_number(willis_page36_fig54)
    2
    sage: willis_page36_fig54.order()
    9
    sage: willis_page36_fig54.size()
    24
"""
willis_page36_fig54 = Graph("H~`HW~~")
willis_page36_fig54.name(new = "willis_page36_fig54")

"""
From Willis's thesis, page 36, Figure 5.5
A graph where none of the lower bounds in the thesis give the exact value for alpha

    sage: independence_number(willis_page36_fig55)
    3
    sage: willis_page36_fig54.order()
    7
    sage: willis_page36_fig54.size()
    13
"""
willis_page36_fig55 = Graph("F@^vo")
willis_page36_fig55.name(new = "willis_page36_fig55")

"""
From Willis's thesis, page 37, Figure 5.6
A graph where none of the lower bounds in the thesis give the exact value for alpha

    sage: independence_number(willis_page37_fig56)
    3
    sage: willis_page37_fig56.order()
    7
    sage: willis_page37_fig56.size()
    15
"""
willis_page37_fig56 = Graph("Fimzw")
willis_page37_fig56.name(new = "willis_page37_fig56")

"""
From Willis's thesis, page 37, Figure 5.8
A graph where none of the lower bounds in the thesis give the exact value for alpha

    sage: independence_number(willis_page37_fig58)
    3
    sage: willis_page37_fig58.order()
    9
    sage: willis_page37_fig58.size()
    16
"""
willis_page37_fig58 = Graph("H?iYbC~")
willis_page37_fig58.name(new = "willis_page37_fig58")

"""
From Willis's thesis, page 39, Figure 5.10
A graph where none of the upper or lower bounds in the thesis give the exact value for alpha

    sage: independence_number(willis_page39_fig510)
    5
    sage: willis_page39_fig510.order()
    12
    sage: willis_page39_fig510.size()
    18
"""
willis_page39_fig510 = Graph("Kt?GOKEOGal?")
willis_page39_fig510.name(new = "willis_page39_fig510")

"""
From Willis's thesis, page 39, Figure 5.11
A graph where none of the upper or lower bounds in the thesis give the exact value for alpha

    sage: independence_number(willis_page39_fig511)
    5
    sage: willis_page39_fig511.order()
    12
    sage: willis_page39_fig511.size()
    18
"""
willis_page39_fig511 = Graph("KhCGKCHHACqC")
willis_page39_fig511.name(new = "willis_page39_fig511")

"""
From Willis's thesis, page 40, Figure 5.12
A graph where none of the upper or lower bounds in the thesis give the exact value for alpha

    sage: independence_number(willis_page40_fig512)
    6
    sage: willis_page40_fig512.order()
    14
    sage: willis_page40_fig512.size()
    21
"""
willis_page40_fig512 = Graph("Ms???\?OGdAQJ?J??")
willis_page40_fig512.name(new = "willis_page40_fig512")

"""
From Willis's thesis, page 41, Figure 5.14
A graph where none of the upper or lower bounds in the thesis give the exact value for alpha

    sage: independence_number(willis_page41_fig514)
    5
    sage: willis_page41_fig514.order()
    12
    sage: willis_page41_fig514.size()
    18
"""
willis_page41_fig514 = Graph("Kt?GGGBQGeL?")
willis_page41_fig514.name(new = "willis_page41_fig514")

"""
From Willis's thesis, page 41, Figure 5.15
A graph where none of the upper or lower bounds in the thesis give the exact value for alpha

    sage: independence_number(willis_page41_fig515)
    4
    sage: willis_page41_fig515.order()
    11
    sage: willis_page41_fig515.size()
    22
"""
willis_page41_fig515 = Graph("JskIIDBLPh?")
willis_page41_fig515.name(new = "willis_page41_fig515")

"""
From Elphick-Wocjan page 8
"""
elphick_wocjan_page8 = Graph("F?Azw")
elphick_wocjan_page8.name(new = "Elphick-Wocjan p.8")

"""
From Elphick-Wocjan page 9
"""
elphick_wocjan_page9 = Graph("FqhXw")
elphick_wocjan_page9.name(new = "Elphick-Wocjan p.9")

"""
An odd wheel with 8 vertices
p.175
Rebennack, Steffen, Gerhard Reinelt, and Panos M. Pardalos. "A tutorial on branch and cut algorithms for the maximum stable set problem." International Transactions in Operational Research 19.1-2 (2012): 161-199.

    sage: odd_wheel_8.order()
    8
    sage: odd_wheel_8.size()
    14
"""
odd_wheel_8 = Graph("G|eKMC")
odd_wheel_8.name(new = "odd_wheel_8")

"""
An odd antihole with 7 vertices
p.175
Rebennack, Steffen, Gerhard Reinelt, and Panos M. Pardalos. "A tutorial on branch and cut algorithms for the maximum stable set problem." International Transactions in Operational Research 19.1-2 (2012): 161-199.

    sage: odd_antihole_7.order()
    7
    sage: odd_antihole_7.size()
    14
"""
odd_antihole_7 = Graph("F}hXw")
odd_antihole_7.name(new = "odd_antihole_7")

#GRAPH LISTS

#all with order 3 to 9, a graph is chroamtic_index_critical if it is class 2 removing any edge increases chromatic index

#all with order 3 to 9, a graph is alpha_critical if removing any edge increases independence number
#all alpha critical graphs of orders 2 to 9, 53 in total
alpha_critical_graph_names = ['A_','Bw', 'C~', 'Dhc', 'D~{', 'E|OW', 'E~~w', 'FhCKG', 'F~[KG', 'FzEKW', 'Fn[kG', 'F~~~w', 'GbL|TS', 'G~?mvc', 'GbMmvG', 'Gb?kTG', 'GzD{Vg', 'Gb?kR_', 'GbqlZ_', 'GbilZ_', 'G~~~~{', 'GbDKPG', 'HzCGKFo', 'H~|wKF{', 'HnLk]My', 'HhcWKF_', 'HhKWKF_', 'HhCW[F_', 'HxCw}V`', 'HhcGKf_', 'HhKGKf_', 'Hh[gMEO', 'HhdGKE[', 'HhcWKE[', 'HhdGKFK', 'HhCGGE@', 'Hn[gGE@', 'Hn^zxU@', 'HlDKhEH', 'H~~~~~~', 'HnKmH]N', 'HnvzhEH', 'HhfJGE@', 'HhdJGM@', 'Hj~KHeF', 'HhdGHeB', 'HhXg[EO', 'HhGG]ES', 'H~Gg]f{', 'H~?g]vs', 'H~@w[Vs', 'Hn_k[^o']
alpha_critical_easy = []
for s in alpha_critical_graph_names:
    g = Graph(s)
    g.name(new="alpha_critical_"+ s)
    alpha_critical_easy.append(g)

#all order-7 chromatic_index_critical_graphs (and all are overfull)
L = ['FhCKG', 'FzCKW', 'FzNKW', 'FlSkG', 'Fn]kG', 'FlLKG', 'FnlkG', 'F~|{G', 'FnlLG', 'F~|\\G', 'FnNLG', 'F~^LW', 'Fll\\G', 'FllNG', 'F~l^G', 'F~|^w', 'F~~^W', 'Fnl^W', 'FlNNG', 'F|\\Kg', 'F~^kg', 'FlKMG']
chromatic_index_critical_7 = []
for s in L:
    g=Graph(s)
    g.name(new="chromatic_index_critical_7_" + s)
    chromatic_index_critical_7.append(g)

#class 0 pebbling graphs
import pickle, os, os.path
try:
    class0graphs_dict = pickle.load(open(os.environ['HOME'] + "/objects-invariants-properties/class0graphs_dictionary.pickle","r"))
except:
    class0graphs_dict = {}
class0graphs = []
for d in class0graphs_dict:
    g = Graph(class0graphs_dict[d])
    g.name(new = d)
    class0graphs.append(g)
class0small = [g for g in class0graphs if g.order() < 30]

c5=graphs.CycleGraph(5)
c5.name(new = "c5")

graph_objects = [paw, kite, p4, dart, k3, k4, k5, c6ee, c5chord, graphs.DodecahedralGraph(), c8chorded, c8chords, graphs.ClebschGraph(),  prismy, c24, c26, c60, c6xc6, holton_mckay, sixfour, c4, graphs.PetersenGraph(), p2, graphs.TutteGraph(), non_ham_over, throwing, throwing2, throwing3, kratsch_lehel_muller, graphs.BlanusaFirstSnarkGraph(), graphs.BlanusaSecondSnarkGraph(), graphs.FlowerSnark(), k1_3, ryan3, k10, graphs.MycielskiGraph(5), c3mycielski, s13e, ryan2, s22e, difficult11, graphs.BullGraph(), graphs.ChvatalGraph(), graphs.ClawGraph(), graphs.DesarguesGraph(), graphs.DiamondGraph(), graphs.FlowerSnark(), graphs.FruchtGraph(), graphs.HoffmanSingletonGraph(), graphs.HouseGraph(), graphs.OctahedralGraph(), graphs.ThomsenGraph(), pete , graphs.PappusGraph(), graphs.GrotzschGraph(), graphs.GrayGraph(), graphs.HeawoodGraph(), graphs.HerschelGraph(), graphs.CoxeterGraph(), graphs.BrinkmannGraph(), graphs.TutteCoxeterGraph(), graphs.TutteGraph(), graphs.RobertsonGraph(), graphs.FolkmanGraph(), graphs.Balaban10Cage(), graphs.PappusGraph(), graphs.TietzeGraph(), graphs.SylvesterGraph(), graphs.SzekeresSnarkGraph(), graphs.MoebiusKantorGraph(), ryan, inp, c4c4, regular_non_trans, bridge, p10k4, c100, starfish, c5k3, k5pendant, graphs.ShrikhandeGraph(), sylvester, fork, edge_critical_5, edge_critical_11_1, edge_critical_11_2, pete_minus, c5, bow_tie, pepper_residue_graph, barrus_graph, p5, c6, c9, ce3, ce4, ce5, k4e2split, flower_with_3_petals, flower_with_4_petals, paw_x_paw, graphs.WagnerGraph(), houseX, ce7, triangle_star, ce8, ce9, ce10, p3, binary_octahedron, ce11, prism, ce12, ce13, ce14, pepper_non_perfect_graph, gallai_tree, willis_page4, willis_page13_fig27, willis_page9, willis_page10_fig23, willis_page10_fig24, willis_page13_fig26, willis_page18, willis_page21, willis_page25_fig32, willis_page25_fig33, lemke, k5_3, willis_page29, p9, ce15, willis_page35_fig51, willis_page35_fig52, willis_page36_fig53, willis_page36_fig54, willis_page36_fig55, willis_page37_fig56, willis_page37_fig58, trig_antiprism_capped, willis_page39_fig510, willis_page39_fig511, willis_page40_fig512, willis_page41_fig514, willis_page41_fig515, k3_3_line_graph, ce16, ce17, elphick_wocjan_page8, elphick_wocjan_page9, p29, ce18, ce19, ce20, ce21, ce22, ce23, ce24, ce25, ce26, ce27, c102, p102, ce28, ce29, ce30, ce31, ce32, ce33, ce34, ce35, ce36, ce37, ce38, ce39, ce40, ce41, ce42, ce43, ce44, ce45, ce47, ce48, ce49, ce50, ce51, ce52, ce53, ce54, ce55, ce56, ce57, k37, c34, ce58, ce59, ce60, ce61, ce62, ce63, ce64, ce65, ce66, ce67, ce68, ce69, ce70, ce71, ce72, ce73, k1_9, ce74, ce75, ce76, ce77, ce78, ce79, ce80, ce81, ce82, ce83, ce84, ce85, ce86, ce87, ce88, ce89, ce91, c22, odd_wheel_8, odd_antihole_7] + alpha_critical_easy

alpha_critical_hard = [Graph('Hj\\x{F{')]

chromatic_index_critical_graphs = chromatic_index_critical_7 + [edge_critical_5, edge_critical_11_1, edge_critical_11_2, pete_minus]

#graphs where some computations are especially slow
problem_graphs = [graphs.MeredithGraph(), graphs.SchlaefliGraph(), haemers, c3mycielski4, alon_seymour] + chromatic_index_critical_7 + class0small + alpha_critical_hard
#meredith graph is 4-reg, class2, non-hamiltonian: http://en.wikipedia.org/wiki/Meredith_graph


#graph_objects: all graphs with no duplicates

#obvious way to remove duplicates in list of ALL objects

"""
graph_objects = []
for g in union_objects, idfun=Graph.graph6_string:
    if not g in graph_objects:
        graph_objects.append(g)
"""

#fast way to remove duplicates in list of ALL objects
#from : http://www.peterbe.com/plog/uniqifiers-benchmark


def remove_duplicates(seq, idfun=None):
    # order preserving
    if idfun is None:
        def idfun(x): return x
    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        # in old Python versions:
        # if seen.has_key(marker)
        # but in new ones:
        if marker in seen: continue
        seen[marker] = 1
        result.append(item)
    return result

#could run this occasionally to check there are no duplicates
#graph_objects = remove_duplicates(union_objects, idfun=Graph.graph6_string)



#TESTING

#check for invariant relation that separtates G from class defined by property
def find_separating_invariant_relation(g, objects, property, invariants):
    L = [x for x in objects if (property)(x)]
    for inv1 in invariants:
        for inv2 in invariants:
            if inv1(g) > inv2(g) and all(inv1(x) <= inv2(x) for x in L):
                return inv1.__name__, inv2.__name__
    print "no separating invariants"



#finds "difficult" graphs for necessary conditions, finds graphs which don't have property but which have all necessary conditions
def test_properties_upper_bound_theory(objects, property, theory):
     for g in objects:
         if not property(g) and all(f(g) for f in theory):
             print g.name()

#finds "difficult" graphs for sufficient conditions, finds graphs which dont have any sufficient but do have property
def test_properties_lower_bound_theory(objects, property, theory):
     for g in objects:
         if property(g) and not any(f(g) for f in theory):
             print g.name()

def find_coextensive_properties(objects, properties):
     for p1 in properties:
         for p2 in properties:
             if p1 != p2 and all(p1(g) == p2(g) for g in objects):
                 print p1.__name__, p2.__name__
     print "DONE!"


#load graph property data dictionary, if one exists
try:
    graph_property_data = load(os.environ['HOME'] +'/objects-invariants-properties/graph_property_data.sobj')
    print "loaded graph properties data file"
except IOError:
    print "can't load graph properties sobj file"
    graph_property_data = {}



#this version will open existing data file, and update as needed
def update_graph_property_data(new_objects,properties):
    global graph_property_data
    #try to open existing sobj dictionary file, else initialize empty one
    try:
        graph_property_data = load(os.environ['HOME'] +'/objects-invariants-properties/graph_property_data.sobj')
    except IOError:
        print "can't load properties sobj file"
        graph_property_data = {}

    #check for graph key, if it exists load the current dictionary, if not use empty prop_value_dict as *default*
    for g in new_objects:
        print g.name()
        if g.name not in graph_property_data.keys():
            graph_property_data[g.name()] = {}

        #check for property key, if it exists load the current dictionary, if not initialize an empty dictionary for property
        for prop in properties:
            try:
                graph_property_data[g.name()][prop.__name__]
            except KeyError:
                graph_property_data[g.name()][prop.__name__] = prop(g)

    save(graph_property_data, "graph_property_data.sobj")
    print "DONE"

#load graph property data dictionary, if one exists
try:
    graph_invariant_data = load(os.environ['HOME'] +'/objects-invariants-properties/graph_invariant_data.sobj')
    print "loaded graph invariants data file"
except IOError:
    print "can't load graph invariant sobj file"
    graph_invariant_data = {}


#this version will open existing data file, and update as needed
def update_graph_invariant_data(new_objects,invariants):
    #try to open existing sobj dictionary file, else initialize empty one
    global graph_invariant_data
    try:
        graph_invariant_data = load(os.environ['HOME'] +'/objects-invariants-properties/graph_invariant_data.sobj')
        print "loaded graph invariants data file"
    except IOError:
        print "can't load invariant sobj file"
        graph_invariant_data = {}

    #check for graph key, if it exists load the current dictionary, if not use empty prop_value_dict as *default*
    for g in new_objects:
        print g.name()
        if g.name not in graph_invariant_data.keys():
            graph_invariant_data[g.name()] = {}

        #check for property key, if it exists load the current dictionary, if not initialize an empty dictionary for property
        for inv in invariants:
            try:
                graph_invariant_data[g.name()][inv.__name__]
            except KeyError:
                graph_invariant_data[g.name()][inv.__name__] = inv(g)

    save(graph_invariant_data, "graph_invariant_data.sobj")
    print "DONE"



# THEORY

alpha_upper_bounds = []

alpha_lower_bounds = []
