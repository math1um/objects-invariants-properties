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
    """
    for degrees d_1 >= ... >= d_n
    returns the maximum over all indices i of of the min(i,d_i + 1)

    sage: welsh_powell(k5) = 4
    """
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

# Cut Vertices Theorem as invariant
# Cited in the theorem section
def cut_vertices_thm_invariant(G):
    return (g.order() - (card_cut_vertices(g)/2) - (1/2))

# Radius Pendants Theorem as invariant
# Three Bounds on the Independence Number of a Graph - C. E. Larson, R. Pepper
def radius_pendants_thm_invariant(g):
    return (g.radius() + (card_pendants(g)/2) - 1)

# Median Degree Theorem as invariant
# Three Bounds on the Independence Number of a Graph - C. E. Larson, R. Pepper
def median_degree_thm_invariant(g):
    return (g.order() - (median_degree(g)/2) - 1/2)

# Two Stability Theta Bound
# For graphs with alpha <= 2,
# lovasz_theta <= 2^(2/3)*n^(1/3)
# The Sandwich Theorem by Knuth p. 47
def two_stability_theta_bound(g):
    return 2^(2/3)*g.order()^(1/3)

# Two Stability Theta Bound
# For graphs with alpha <= 2,
# lovasz_theta <= 2^(2/3)*n^(1/3)
# The Sandwich Theorem by Knuth p. 47
def two_stability_theta_bound(g):
    return 2^(2/3)*g.order()^(1/3)

# Lovasz Theta over Root N
# The Sandwich Theorem by Knuth p. 45
def lovasz_theta_over_root_n(g):
    return g.lovasz_theta()/sqrt(g.order())

# Theta * Theta-Complement
# The Sandwich Theorem by Knuth, p. 27
def theta_theta_complement(g):
    return g.lovasz_theta() * g.complement().lovasz_theta()

# Depth = Order - Residue
# This is the number of steps it takes for Havel-Hakimi to terminate
def depth(g):
    return g.order()-residue(g)

# Clearly intractable
# alpha / order
def independence_ratio(g):
    return independence_number(g)/(g.order()+0.0)

# Godsil-Newman Upper Bound theorem as invariant
# Godsil, Chris D., and Mike W. Newman. "Eigenvalue bounds for independent sets." Journal of Combinatorial Theory, Series B 98.4 (2008): 721-734.
def godsil_newman_thm_invariant(g):
    L = max(g.laplacian_matrix().change_ring(RDF).eigenvalues())
    return g.order()*(L-min_degree(g))/L

# Lovasz Theta of the complement of the given graph
def lovasz_theta_complement(g):
    return g.complement().lovasz_theta()

# N over lovasz_theta_complement
# This is a lower bound for lovasz theta
# The Sandwich Theorem by Knuth, p. 27
def n_over_lovasz_theta_complement(g):
    return g.order()/lovasz_theta_complement(g)

# The number of vertices at even distance from v and return the max over all vertices
def max_even(g):
    from sage.graphs.distances_all_pairs import distances_all_pairs
    D = distances_all_pairs(g)
    evens_list = []
    for u in D:
        evens = 0
        for v in D[u]:
            if D[u][v] % 2 == 0:
                evens += 1
        evens_list.append(evens)
    return max(evens_list)

# The number of vertices at even distance from v and return the min over all vertices
def min_even(g):
    from sage.graphs.distances_all_pairs import distances_all_pairs
    D = distances_all_pairs(g)
    evens_list = []
    for u in D:
        evens = 0
        for v in D[u]:
            if D[u][v] % 2 == 0:
                evens += 1
        evens_list.append(evens)
    return min(evens_list)

# The number of vertices at odd distance from v and return the max over all vertices
def max_odd(g):
    from sage.graphs.distances_all_pairs import distances_all_pairs
    D = distances_all_pairs(g)
    odds_list = []
    for u in D:
        odds = 0
        for v in D[u]:
            if D[u][v] % 2 != 0:
                odds += 1
        odds_list.append(odds)
    return max(odds_list)

# The number of vertices at odd distance from v and return the min over all vertices
def min_odd(g):
    from sage.graphs.distances_all_pairs import distances_all_pairs
    D = distances_all_pairs(g)
    odds_list = []
    for u in D:
        odds = 0
        for v in D[u]:
            if D[u][v] % 2 != 0:
                odds += 1
        odds_list.append(odds)
    return min(odds_list)

# AGX Upper Bound Theorem as invariant
#Aouchiche, Mustapha, Gunnar Brinkmann, and Pierre Hansen. "Variable neighborhood search for extremal graphs. 21. Conjectures and results about the independence number." Discrete Applied Mathematics 156.13 (2008): 2530-2542.
def AGX_upper_bound_thm_invariant(g):
    return (g.order() + max_degree(g) - ceil(2 * sqrt(g.order() - 1)))

# AGX Lower Bound Theorem as invariant
# Aouchiche, Mustapha, Gunnar Brinkmann, and Pierre Hansen. "Variable neighborhood search for extremal graphs. 21. Conjectures and results about the independence number." Discrete Applied Mathematics 156.13 (2008): 2530-2542.
def AGX_lower_bound_thm_invariant(g):
    return ceil(2 * sqrt(g.order()))

#returns sum of distances between *distinct* vertices, return infinity is graph is not connected
def transmission(g):
    if not g.is_connected():
        return Infinity
    V = g.vertices()
    D = g.distance_all_pairs()
    n = g.order()
    return sum([D[v][w] for v in V for w in V if v != w])

# removed fiedler for incorrect value calculations
efficiently_computable_invariants = [average_distance, Graph.diameter, Graph.radius,
Graph.girth,  Graph.order, Graph.size, Graph.szeged_index, Graph.wiener_index,
min_degree, max_degree, matching_number, residue, annihilation_number, fractional_alpha,
Graph.lovasz_theta, cvetkovic, cycle_space_dimension, card_center, card_periphery,
max_eigenvalue, kirchhoff_index, largest_singular_value, vertex_con, edge_con,
Graph.maximum_average_degree, Graph.density, welsh_powell, wilf, brooks,
different_degrees, szekeres_wilf, average_vertex_temperature, randic, median_degree,
max_even_minus_even_horizontal, laplacian_energy, gutman_energy, average_degree,
degree_variance, number_of_triangles, graph_rank, inverse_degree, sum_temperatures,
card_positive_eigenvalues, card_negative_eigenvalues, card_zero_eigenvalues,
card_cut_vertices, Graph.clustering_average, Graph.connected_components_number,
Graph.spanning_trees_count, card_pendants, card_bridges, alon_spencer, caro_wei,
degree_sum, order_automorphism_group, sigma_2, brinkmann_steffen,
card_independence_irreducible_part, critical_independence_number, card_KE_part,
fractional_covering, eulerian_faces, barrus_q, mean_common_neighbors,
max_common_neighbors, min_common_neighbors, distinct_degrees, barrus_bound,
geometric_length_of_degree_sequence, cut_vertices_thm_invariant, radius_pendants_thm_invariant,
median_degree_thm_invariant, two_stability_theta_bound, lovasz_theta_over_root_n, theta_theta_complement,
depth, godsil_newman_thm_invariant, lovasz_theta_complement, n_over_lovasz_theta_complement,
max_even, min_even, max_odd, min_odd, AGX_upper_bound_thm_invariant, AGX_lower_bound_thm_invariant,
transmission]

intractable_invariants = [independence_number, domination_number, chromatic_index,
Graph.clique_number, clique_covering_number, n_over_alpha, chromatic_num,
independent_dominating_set_number, independence_ratio, Graph.treewidth]

#for invariants from properties and INVARIANT_PLUS see below

#FAST ENOUGH (tested for graphs on 140921): lovasz_theta, clique_covering_number, all efficiently_computable
#SLOW but FIXED for SpecialGraphs

invariants = efficiently_computable_invariants + intractable_invariants

#removed for speed: Graph.treewidth, card_max_cut
