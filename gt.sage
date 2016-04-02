
# UTILITIES
def memoized(f):
    """
    Enables memoization for functions
    """
    import functools

    # make sure we can also handle instance methods
    if hasattr(f, '__func__'):
        f = f.__func__

    # create a cache for the function
    if not hasattr(f, '_cache'):
        f._cache = {}

    #function that wraps f and handles the caching
    #@wraps makes sure this is done transparently
    @functools.wraps(f)
    def memo(g):
        key = g.graph6_string()
        if key not in f._cache:
            f._cache[key] = f(g)
        return f._cache[key]

    return memo

def add_to_cache(f, g, value, create_cache=True):
    import types
    if type(f) == types.MethodType:
        f = f.__func__
    elif type(f) != types.FunctionType:
        raise ValueError("The argument f is not a function or instance method")

    if hasattr(f, '_cache'):
        f._cache[g.graph6_string()] = value
    elif create_cache:
        f._cache = {g.graph6_string():value}

def function_index(l, f):
    """
    Returns the index of f in l. This works regardless whether f is contained in the
    list, or the __func__ of f (for unbounded methods), or a memoized version of f.
    """
    return [t.__name__ for t in l].index(f.__name__)


#GRAPH UTILITIES

#find first i element independent set I with alpha(G[N[I]]) = |I|
def find_first_removeable_independent_set(g, i):
    V = g.vertices()
    #print V

    for S in Subsets(Set(V),i):
        #print S
        if g.is_independent_set(S):
            #print "in loop"
            T = Set(closed_neighborhood(g,list(S)))
            alpha = independence_number(g.subgraph(T))
            if alpha == i:
                return S
    return Set([])


#iteratively finds removeable sets of size i until no more can be found
def find_all_removable_independent_sets(g,i):
    V = g.vertices()
    I = Set([])

    next = find_first_removeable_independent_set(g,i)
    I = I.union(next)
    #print "first I = {}".format(I)
    #print closed_neighborhood(g,list(I))
    #Vh = [v for v in V if v not in closed_neighborhood(g,list(I))]
    #print "Vh = {}".format(Vh)

    while next.cardinality() > 0:
        Vh = [v for v in V if v not in closed_neighborhood(g,list(I))]
        #print "Vh = {}".format(Vh)
        if len(Vh) > 0:
            h = g.subgraph(Vh)
            next = find_first_removeable_independent_set(h,i)
            #print "next = {}".format(next)
            I = I.union(next)
            #print "I = {}".format(I)
        else:
            next = Set([])

    return I



#find first i element independent set I with alpha(G[N[I]]) = |I|
def find_first_removeable_independent_set(g, i):
    V = g.vertices()
    #print V

    for S in Subsets(Set(V),i):
        #print S
        if g.is_independent_set(S):
            #print "in loop"
            T = Set(closed_neighborhood(g,list(S)))
            alpha = independence_number(g.subgraph(T))
            if alpha == i:
                return S
    return Set([])

#iteratively finds removeable sets of size i until no more can be found
def find_all_removable_independent_sets(g,i):
    V = g.vertices()
    I = Set([])

    next = find_first_removeable_independent_set(g,i)
    I = I.union(next)
    #print "first I = {}".format(I)
    #print closed_neighborhood(g,list(I))
    #Vh = [v for v in V if v not in closed_neighborhood(g,list(I))]
    #print "Vh = {}".format(Vh)

    while next.cardinality() > 0:
        Vh = [v for v in V if v not in closed_neighborhood(g,list(I))]
        #print "Vh = {}".format(Vh)
        if len(Vh) > 0:
            h = g.subgraph(Vh)
            next = find_first_removeable_independent_set(h,i)
            #print "next = {}".format(next)
            I = I.union(next)
            #print "I = {}".format(I)
        else:
            next = Set([])

    return I



#finds independent sets of size i in g unioned with their neighborhoods.
#return LIST of closed neighborhood SETS
def find_lower_bound_sets(g, i):
    V = g.vertices()
    #print V
    lowersets = []

    for S in Subsets(Set(V),i):
        #print S
        if g.is_independent_set(S):
            #print "in loop"
            T = Set(closed_neighborhood(g,list(S)))
            if T not in Set(lowersets):
                lowersets.append(T)
    return lowersets

#check that a set extends to a maximum independent set
def check_independence_extension(g,S):
    V = g.vertices()
    alpha = independence_number(g)
    #print alpha

    if not S.issubset(Set(V)) or not g.is_independent_set(S):
        return False

    N = neighbors_set(g,S)
    X = [v for v in V if v not in S and v not in N]
    h = g.subgraph(X)
    alpha_h = independence_number(h)
    #print alpha_h, len(S)

    return (alpha == alpha_h + len(S))

#find and save names of all alpha critical graphs of given order
def find_alpha_critical_graphs(order):
    graphgen = graphs(order)
    count = 0
    alpha_critical_name_list = []
    while True:
        try:
            g = graphgen.next()
        except StopIteration:
            break

        if g.is_connected():
            count += 1
            if is_alpha_critical(g):
                print "current connected count = {}".format(count)
                print g.graph6_string()
                alpha_critical_name_list.append(g.graph6_string())
    s = "alpha_critical_name_list_{}".format(order)
    save(alpha_critical_name_list, s)
    return alpha_critical_name_list

#tests whether a sequence is the degree sequence of some graph
def is_degree_sequence(L):
    try:
        graphs.DegreeSequence(L)
    except:
        return False
    return True

#ALPHA APPROXIMATIONS

#finds independent sets of size i in g unioned with their neighborhoods.
#return LIST of closed neighborhood SETS
def find_lower_bound_sets(g, i):
    V = g.vertices()
    #print V
    lowersets = []

    for S in Subsets(Set(V),i):
        #print S
        if g.is_independent_set(S):
            #print "in loop"
            T = Set(closed_neighborhood(g,list(S)))
            if T not in Set(lowersets):
                lowersets.append(T)
    return lowersets

def alpha_lower_approximation(g, i):
    n = g.order()

    LP = MixedIntegerLinearProgram(maximization=False)
    x = LP.new_variable(nonnegative=True)

    # We define the objective
    LP.set_objective(sum([x[v] for v in g]))

    # For any subset, we define a constraint
    for j in range(1,i+1):
        for S in find_lower_bound_sets(g, j):
            #print S, S.cardinality()

            LP.add_constraint(sum([x[k] for k in S]), min = j)

    LP.solve()

    #return LP

    x_sol = LP.get_values(x)
    print x_sol
    return sum(x_sol.values())

#input = graph g
#output = bipartite graph with twice as many nodes and edges
#new nodes are labeled n to 2n-1
#assumes nodes in g are labeled [0..n-1]
#same as cartesian product with k2, but output labeling is guarnateed to be integers
def make_bidouble_graph(g):
    n = g.order()
    V = g.vertices()
    gdub = Graph(2*n)
    #print "gdub order = {}".format(gdub.order())

    for (i,j) in g.edges(labels = False):
        #print (i,j)
        gdub.add_edge(i,j+n)
        gdub.add_edge(j,i+n)
    return gdub

#HEURISTIC ALGORITHMS

#takes vertex of max degree, deletes so long as degree > 0, retuens remaining ind set
def MAXINE_independence_heuristic(g):
    V = g.vertices()
    h = g.subgraph(V)
    delta = max(h.degree())

    while delta > 0:
        #print "V is {}".format(V)
        #print "h vertices = {}, h.degree = {}".format(h.vertices(),h.degree())

        max_degree_vertex = V[h.degree().index(delta)]
        #print "max_degree_vertex = {}".format(max_degree_vertex)
        #print "h.neighbors(max_degree_vertex) = {}".format(h.neighbors(max_degree_vertex))
        V.remove(max_degree_vertex)
        h = g.subgraph(V)
        delta = max(h.degree())
        print "delta = {}".format(delta)

    return len(V)

#takes vertex of min degree, adds it to max ind set until no vertices left
def MIN_independence_heuristic(g):
    V = g.vertices()
    I = []
    while V != []:
        #print "V is {}".format(V)
        h = g.subgraph(V)
        #print "h vertices = {}, h.degree = {}".format(h.vertices(),h.degree())
        delta = min(h.degree())
        #print "delta = {}".format(delta)
        min_degree_vertex = V[h.degree().index(delta)]
        #print "min_degree_vertex = {}".format(min_degree_vertex)
        I.append(min_degree_vertex)
        V = [v for v in V if v not in closed_neighborhood(h, min_degree_vertex)]
        #break
    return len(I)


#GRAPH INVARIANTS

def domination_number(g):
    return g.dominating_set(value_only=True)

def min_degree(g):
    return min(g.degree())

def max_degree(g):
    return max(g.degree())

def matching_number(g):
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

efficiently_computable_invariants = [average_distance, Graph.diameter, Graph.radius, Graph.girth,  Graph.order, Graph.size, Graph.szeged_index, Graph.wiener_index, min_degree, max_degree, matching_number, residue, annihilation_number, fractional_alpha, Graph.lovasz_theta, cvetkovic, cycle_space_dimension, card_center, card_periphery, max_eigenvalue, kirchhoff_index, largest_singular_value, vertex_con, edge_con, Graph.maximum_average_degree, Graph.density, welsh_powell, wilf, brooks, different_degrees, szekeres_wilf, average_vertex_temperature, randic, median_degree, max_even_minus_even_horizontal, fiedler, laplacian_energy, gutman_energy, average_degree, degree_variance, number_of_triangles, graph_rank, inverse_degree, sum_temperatures, card_positive_eigenvalues, card_negative_eigenvalues, card_zero_eigenvalues, card_cut_vertices, Graph.clustering_average, Graph.connected_components_number, Graph.spanning_trees_count, card_pendants, card_bridges, alon_spencer, caro_wei, degree_sum, order_automorphism_group, sigma_2, brinkmann_steffen, card_independence_irreducible_part, critical_independence_number, card_KE_part, fractional_covering]

intractable_invariants = [independence_number, domination_number, chromatic_index, Graph.clique_number, clique_covering_number, n_over_alpha, chromatic_num, independent_dominating_set_number]

#for invariants from properties and INVARIANT_PLUS see below

#FAST ENOUGH (tested for graphs on 140921): lovasz_theta, clique_covering_number, all efficiently_computable
#SLOW but FIXED for SpecialGraphs

invariants = efficiently_computable_invariants + intractable_invariants

#removed for speed: Graph.treewidth, card_max_cut

#set precomputed values
#add_to_cache(Graph.treewidth, graphs.BuckyBall(), 10)
add_to_cache(chromatic_index, graphs.MeredithGraph(), 5) #number from http://en.wikipedia.org/wiki/Meredith_graph
add_to_cache(clique_covering_number, graphs.SchlaefliGraph(), 6)
add_to_cache(chromatic_num, graphs.SchlaefliGraph(), 9)  #number from http://en.wikipedia.org/wiki/Schl%C3%A4fli_graph


#GRAPH PROPERTIES




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
    k = vertex_con(g)
    if k < 2:
        return False
    else:
        return True

#part of pebbling class0 sufficient condition
def is_three_connected(g):
    k = vertex_con(g)
    if k < 3:
        return False
    else:
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

#g is factor-critical if order is odd and removal of any vertex gives graph with perfect matching
def is_factor_critical(g):
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

#add all properties derived from pairs of invariants
invariant_relation_properties = [has_leq_invariants(f,g) for f in invariants for g in invariants if f != g]


efficiently_computable_properties = [Graph.is_regular, Graph.is_planar, Graph.is_forest, Graph.is_eulerian, Graph.is_connected, Graph.is_clique, Graph.is_circular_planar, Graph.is_chordal, Graph.is_bipartite, Graph.is_cartesian_product, Graph.is_distance_regular,  Graph.is_even_hole_free, Graph.is_gallai_tree, Graph.is_line_graph, Graph.is_overfull, Graph.is_perfect, Graph.is_split, Graph.is_strongly_regular, Graph.is_triangle_free, Graph.is_weakly_chordal, is_dirac, is_ore, is_haggkvist_nicoghossian, is_generalized_dirac, is_van_den_heuvel, is_two_connected, is_three_connected, is_lindquester, is_claw_free, has_perfect_matching, has_radius_equal_diameter, is_not_forest, is_fan, is_cubic, diameter_equals_twice_radius, diameter_equals_radius, is_locally_connected, matching_covered, is_locally_dirac, is_locally_bipartite, is_locally_two_connected, Graph.is_interval, has_paw, is_paw_free, has_p4, is_p4_free, has_dart, is_dart_free, has_kite, is_kite_free, has_H, is_H_free, has_residue_equals_two, order_leq_twice_max_degree, alpha_leq_order_over_two, is_factor_critical, is_independence_irreducible, has_twin, is_twin_free, is_pebbling_class0, diameter_equals_two, girth_greater_than_2log, is_cycle]

intractable_properties = [Graph.is_hamiltonian, Graph.is_vertex_transitive, Graph.is_edge_transitive, has_residue_equals_alpha, Graph.is_odd_hole_free, Graph.is_semi_symmetric, Graph.is_line_graph, is_planar_transitive, is_class1, is_class2, is_anti_tutte, is_anti_tutte2, has_lovasz_theta_equals_cc, has_lovasz_theta_equals_alpha, is_chvatal_erdos, is_heliotropic_plant, is_geotropic_plant, is_traceable, is_chordal_or_not_perfect, has_alpha_residue_equal_two]

#speed notes
#FAST ENOUGH (tested for graphs on 140921): is_hamiltonian, is_vertex_transitive, is_edge_transitive, has_residue_equals_alpha, is_odd_hole_free, is_semi_symmetric, is_line_graph, is_line_graph, is_anti_tutte, is_planar_transitive
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

#holton-mckay graph: hamiltonian, cubic, planar, radius=4, diameter=6
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
add_to_cache(Graph.lovasz_theta, c100, 46.694)

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

#mycieskian of a triangle: CE to conj that chi <= max(clique, nu), chi=4, nu = clique = 3
c3mycieski = Graph('FJnV?')
c3mycieski.name(new="c3mycieski")

#4th mycielskian of a triangle, CE to conj chi <= clique + girth, chi = 7, clique = girth = 3
c3mycielski4 = Graph('~??~??GWkYF@BcuIsJWEo@s?N?@?NyB`qLepJTgRXkAkU?JPg?VB_?W[??Ku??BU_??ZW??@u???Bs???Bw???A??F~~_B}?^sB`o[MOuZErWatYUjObXkZL_QpWUJ?CsYEbO?fB_w[?A`oCM??DL_Hk??DU_Is??Al_Dk???l_@k???Ds?M_???V_?{????oB}?????o[M?????WuZ?????EUjO?????rXk?????BUJ??????EsY??????Ew[??????B`o???????xk???????FU_???????\\k????????|_????????}_????????^_?????????')
c3mycielski4.name(new="c3mycielski4")
add_to_cache(chromatic_num, c3mycielski4, 7)
add_to_cache(chromatic_index, c3mycielski4, 32)
add_to_cache(clique_covering_number, c3mycielski4, 31)

# a PAW is a traingle with a pendant, same as a Z1
paw=Graph('C{')
paw.name(new="paw")

#a KITE is a C4 with a chord
kite = Graph('Cn')
kite.name(new="kite")

#a DART is a kite with a pendant
dart = Graph('DnC')
dart.name(new="dart")

#P4 is a path on 4 vertices
p4=Graph('Ch')
p4.name(new="p4")

#a K5 with a pendant, CE to dirac => regular or planar conjecture
k5pendant = Graph('E~}?')
k5pendant.name(new="k5pendant")

#same as H
killer = Graph('EgSG')
killer.name(new="killer")

#alon_seymour graph: CE to the rank-coloring conjecture, 56-regular, vertex_trans, alpha=2, omega=22, chi=chi'=edge_connect=56
V = VectorSpace(GF(2),6)
S=[V[i] for i in range(64)]
def count_ones(s):
     count = 0
     for i in range(len(s)):
         if s[i] == 1:
             count += 1
     return count
K=[x for x in S if count_ones(x)==1 or count_ones(x) == 6]
alon_seymour=Graph(64)
for i in range(64):
    alon_seymour.set_vertex(i,S[i])
for i in range(64):
     for j in range(64):
         if i < j:
             if sum([alon_seymour.get_vertex(i),alon_seymour.get_vertex(j)]) not in K:
                 alon_seymour.add_edge(i,j)
alon_seymour.name(new="alon_seymour")
add_to_cache(chromatic_num, alon_seymour, 56)
add_to_cache(chromatic_index, alon_seymour, 56)
add_to_cache(edge_con, alon_seymour, 56)
add_to_cache(vertex_con, alon_seymour, 56)
add_to_cache(kirchhoff_index, alon_seymour, 71.0153846154)
add_to_cache(matching_covered, alon_seymour, True)
add_to_cache(is_locally_two_connected, alon_seymour, True)

k3 = graphs.CompleteGraph(3)
k3.name(new="k3")

k4 = graphs.CompleteGraph(4)
k4.name(new="k4")

k5 = graphs.CompleteGraph(5)
k5.name(new="k5")

k6 = graphs.CompleteGraph(6)
k6.name(new="k6")

c4 = graphs.CycleGraph(4)
c4.name(new="c4")

p2 = graphs.PathGraph(2)
p2.name(new="p2")

p6 = graphs.PathGraph(6)
p6.name(new="p6")

p6 = graphs.PathGraph(6)
p6.name(new="p6")

p6 = graphs.PathGraph(6)
p6.name(new="p6")

p6 = graphs.PathGraph(6)
p6.name(new="p6")

#star with 3 rays, order = 4
s3 = graphs.StarGraph(3)
s3.name(new="s3")

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

#GRAPH LISTS

#all with order 3 to 9, a graph is chroamtic_index_critical if it is class 2 removing any edge increases chromatic index

#all with order 3 to 9, a graph is alpha_critical if removing any edge increases independence number
#all alpha critical graphs of orders 2 to 9, 53 in total
alpha_critical_graph_names = ['A_','Bw', 'C~', 'Dhc', 'D~{', 'E|OW', 'E~~w', 'FhCKG', 'F~[KG', 'FzEKW', 'Fn[kG', 'F~~~w', 'GbL|TS', 'G~?mvc', 'GbMmvG', 'Gb?kTG', 'GzD{Vg', 'Gb?kR_', 'GbqlZ_', 'GbilZ_', 'G~~~~{', 'GbDKPG', 'HzCGKFo', 'H~|wKF{', 'Hj\\x{F{', 'HnLk]My', 'HhcWKF_', 'HhKWKF_', 'HhCW[F_', 'HxCw}V`', 'HhcGKf_', 'HhKGKf_', 'Hh[gMEO', 'HhdGKE[', 'HhcWKE[', 'HhdGKFK', 'HhCGGE@', 'Hn[gGE@', 'Hn^zxU@', 'HlDKhEH', 'H~~~~~~', 'HnKmH]N', 'HnvzhEH', 'HhfJGE@', 'HhdJGM@', 'Hj~KHeF', 'HhdGHeB', 'HhXg[EO', 'HhGG]ES', 'H~Gg]f{', 'H~?g]vs', 'H~@w[Vs', 'Hn_k[^o']
alpha_critical_graphs = []
for s in alpha_critical_graph_names:
    g = Graph(s)
    g.name(new="alpha_critical_"+ s)
    alpha_critical_graphs.append(g)

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

graph_objects = [paw, kite, p4, dart, k3, k4, k5, c6ee, c5chord, graphs.DodecahedralGraph(), c8chorded, c8chords, graphs.ClebschGraph(),  prismy, c24, c26, c60, c6xc6, holton_mckay, sixfour, c4, graphs.PetersenGraph(), p2, graphs.TutteGraph(), non_ham_over, throwing, throwing2, throwing3, kratsch_lehel_muller, graphs.BlanusaFirstSnarkGraph(), graphs.BlanusaSecondSnarkGraph(), graphs.FlowerSnark(), s3, ryan3, k10, graphs.MycielskiGraph(5), c3mycieski, c3mycielski4, alon_seymour, s13e, ryan2, s22e, difficult11, graphs.BullGraph(), graphs.ChvatalGraph(), graphs.ClawGraph(), graphs.DesarguesGraph(), graphs.DiamondGraph(), graphs.FlowerSnark(), graphs.FruchtGraph(), graphs.HoffmanSingletonGraph(), graphs.HouseGraph(), graphs.HouseXGraph(), graphs.OctahedralGraph(), graphs.ThomsenGraph(), graphs.TetrahedralGraph(), pete , graphs.PappusGraph(), graphs.GrotzschGraph(), graphs.GrayGraph(), graphs.HeawoodGraph(), graphs.HerschelGraph(), graphs.SchlaefliGraph(), graphs.CoxeterGraph(), graphs.BrinkmannGraph(), graphs.TutteCoxeterGraph(), graphs.TutteGraph(), graphs.RobertsonGraph(), graphs.FolkmanGraph(), graphs.Balaban10Cage(), graphs.PappusGraph(), graphs.TietzeGraph(), graphs.SylvesterGraph(), graphs.SzekeresSnarkGraph(), graphs.MoebiusKantorGraph(), ryan, inp, c4c4, regular_non_trans, bridge, p10k4, c100, starfish, c5k3, k5pendant, graphs.ShrikhandeGraph(), graphs.MeredithGraph(), sylvester, fork, edge_critical_5, edge_critical_11_1, edge_critical_11_2, pete_minus, c5, bow_tie] + alpha_critical_graphs + chromatic_index_critical_7 + class0small

chromatic_index_critical_graphs = chromatic_index_critical_7 + [edge_critical_5, edge_critical_11_1, edge_critical_11_2, pete_minus]


#graphs were some computations are especially slow
problem_graphs = [graphs.MeredithGraph()]
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
def update_graph_property_data(graph_objects,properties):
    #try to open existing sobj dictionary file, else initialize empty one
    try:
        graph_property_data = load(os.environ['HOME'] +'/objects-invariants-properties/graph_property_data.sobj')
    except IOError:
        print "can't load properties sobj file"
        graph_property_data = {}

    #check for graph key, if it exists load the current dictionary, if not use empty prop_value_dict as *default*
    for g in graph_objects:
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
def update_graph_invariant_data(graph_objects,invariants):
    #try to open existing sobj dictionary file, else initialize empty one
    try:
        graph_invariant_data = load(os.environ['HOME'] +'/objects-invariants-properties/graph_invariant_data.sobj')
        print "loaded graph invariants data file"
    except IOError:
        print "can't load invariant sobj file"
        graph_invariant_data = {}

    #check for graph key, if it exists load the current dictionary, if not use empty prop_value_dict as *default*
    for g in graph_objects:
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

