#GRAPH UTILITIES

def check_independence_extension(g,S):
    """
    Returns True if the set S extends to a maximum independent set of the graph g.

        sage: check_independence_extension(graphs.CycleGraph(6), Set([0,2]))
        True
        sage: check_independence_extension(graphs.CycleGraph(6), Set([0,3]))
        False
    """
    V = g.vertices()
    alpha = g.independent_set(value_only=True)
    #print alpha

    if not S.issubset(Set(V)) or not g.is_independent_set(S):
        return False

    N = neighbors_set(g,S)
    X = [v for v in V if v not in S and v not in N]
    h = g.subgraph(X)
    alpha_h = h.independent_set(value_only=True)
    #print alpha_h, len(S)

    return (alpha == alpha_h + len(S))

def find_alpha_critical_graphs(order):
    """
    Returns a list of the graph6 string of each of the alpha critical graphs of
    the given order. A graph g is alpha critical if alpha(g-e) > alpha(g) for
    every edge e in g. This looks at every graph of the given order, so this
    will be slow for any order larger than 8.

    There is a unique alpha critical graph on 3 and 4 vertices::

        sage: find_alpha_critical_graphs(3)
        ['Bw']
        sage: find_alpha_critical_graphs(4)
        ['C~']

    There are two alpha critical graphs on 5 vertices::

        sage: find_alpha_critical_graphs(5)
        ['Dhc', 'D~{']

    There are two alpha critical graphs on 6 vertices::

        sage: find_alpha_critical_graphs(6)
        ['E|OW', 'E~~w']
    """
    graphgen = graphs(order)
    count = 0
    alpha_critical_name_list = []
    for g in graphgen:
        if g.is_connected():
            count += 1
            if is_alpha_critical(g):
                alpha_critical_name_list.append(g.graph6_string())
    s = "alpha_critical_name_list_{}".format(order)
    save(alpha_critical_name_list, s)
    return alpha_critical_name_list

def is_degree_sequence(L):
    """
    Returns True if the list L is the degree sequence of some graph.

    Since a graph always contains at least two vertices of the same degree, a
    list containing no duplicates cannot be a degree sequence::

        sage: is_degree_sequence([i for i in range(8)])
        False

    A cycle has all degrees equal to two and exists for any order larger than
    3, so a list of twos of length at least 3 is a degree sequence::

        sage: is_degree_sequence([2]*10)
        True
    """
    try:
        graphs.DegreeSequence(L)
    except:
        return False
    return True

#ALPHA APPROXIMATIONS

def find_lower_bound_sets(g, i):
    """
    Returns a list of independent sets of size i unioned with their neighborhoods.
    Since this checks all subsets of size i, this is a potentially slow method!

        sage: l = find_lower_bound_sets(graphs.CycleGraph(6),2)
        sage: l
        [{0, 1, 2, 3, 5},
         {0, 1, 2, 3, 4, 5},
         {0, 1, 3, 4, 5},
         {0, 1, 2, 3, 4},
         {0, 1, 2, 4, 5},
         {1, 2, 3, 4, 5},
         {0, 2, 3, 4, 5}]
        sage: type(l[0])
        <class 'sage.sets.set.Set_object_enumerated_with_category'>
    """
    V = g.vertices()
    lowersets = []

    for S in Subsets(Set(V),i):
        if g.is_independent_set(S):
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

def neighbors_set(g,S):
    N = set()
    for v in S:
        for n in g.neighbors(v):
            N.add(n)
    return list(N)

def closed_neighborhood(g, verts):
    if isinstance(verts, list):
        neighborhood = []
        for v in verts:
            neighborhood += [v] + g.neighbors(v)
        return list(set(neighborhood))
    else:
        return [verts] + g.neighbors(verts)

def is_alpha_critical(g):
    #if not g.is_connected():
        #return False
    alpha = g.independent_set(value_only=True)
    for e in g.edges():
        gc = copy(g)
        gc.delete_edge(e)
        alpha_prime = gc.independent_set(value_only=True)
        if alpha_prime <= alpha:
            return False
    return True

#HEURISTIC ALGORITHMS

#takes vertex of max degree, deletes so long as degree > 0, returns remaining ind set
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

"""
Returns true if the given graph exists in the given list.
It also prints out all graphs in the list that are isomorphic so that duplicates may also be found here.
"""
def does_graph_exist(g, L):
    success = False
    for gL in L:
        if g.is_isomorphic(gL):
            print gL.name()
            success = True
    return success

"""
Returns a list of all pairs of isomorphic graphs in the given list.
"""
import itertools
def find_isomorphic_pairs(l):
    pairs = []
    L = itertools.combinations(l, r = 2)
    for pair in L:
        if pair[0].is_isomorphic(pair[1]):
            pairs.append(pair)
    return pairs

"""
 Returns a Sage graph object generated from the given uncompressed DIMACS file
"""
def read_dimacs_edge_file(filename):
    g = Graph()
    try:
        f = open(filename)
    except IOError:
        print "Couldn't open file:", filename
        exit(-1)
    for line in f:
        if line[0] == 'c':
            continue
        elif line[0] == 'p':
            p, problem, order, size = line.split()
            assert(problem in ("edge", "col")), "Must be an edge problem file"
            order, size = int(order), int(size)
        elif line[0] == 'e':
            e, u, v = line.split()
            g.add_edge(u, v)
    assert(g.order() == order), "Order in problem line does not match generated order"
    assert(g.size() == size), "Size in problem line does not match generated size"
    return g


# Add all DIMACS graphs from the DIMACS subdirectory
def load_dimacs_graphs():
    files = os.listdir("objects-invariants-properties/Objects/DIMACS/")

    for filename in files:
        g = read_dimacs_edge_file("objects-invariants-properties/Objects/DIMACS/" + filename)
        g.name(new = filename[:-4])
        add_to_lists(g, dimacs_graphs, all_graphs)
        print "loaded graph - ", g

# Load the Sloane graphs
def load_sloane_graphs():
    dc64_g6string ="~?@?JXxwm?OJ@wESEYMMbX{VDokGxAWvH[RkTAzA_Tv@w??wF]?oE\?OAHoC_@A@g?PGM?AKOQ??ZPQ?@rgt??{mIO?NSD_AD?mC\
O?J?FG_FOOEw_FpGA[OAxa?VC?lWOAm_DM@?Mx?Y{A?XU?hwA?PM?PW@?G@sGBgl?Gi???C@_FP_O?OM?VMA_?OS?lSB??PS?`sU\
??Gx?OyF_?AKOCN`w??PA?P[J??@C?@CU_??AS?AW^G??Ak?AwVZg|?Oy_@?????d??iDu???C_?D?j_???M??[Bl_???W??oEV?\
???O??_CJNacABK?G?OAwP??b???GNPyGPCG@???"
    dc64 = Graph(dc64_g6string)
    dc64.name(new="dc64")
    add_to_lists(dc64, sloane_graphs, all_graphs)
    print "loaded graph dc64"

    try:
        s = load('objects-invariants-properties/Objects/dc1024_g6string.sobj')
        print "loaded graph dc1024"
        dc1024 = Graph(s)
        dc1024.name(new="dc1024")
        add_to_lists(dc1024, sloane_graphs, all_graphs)
    except:
        print "couldn't load dc1024_g6string.sobj"

    try:
        s = load('objects-invariants-properties/Objects/dc2048_g6string.sobj')
        print "loaded graph dc2048"
        dc2048 = Graph(s)
        dc2048.name(new="dc2048")
        add_to_lists(dc2048, sloane_graphs, all_graphs)

    except:
        print "couldn't load dc2048_g6string.sobj"

def find_all_max_ind_sets(g):
    """
    Finds all the maximum independent sets and stores them in a list
    """
    final_list = []
    V = Set(g.vertices())
    alpha = independence_number(g)

    for s in V.subsets(alpha):
        if g.is_independent_set(s):
            final_list.append(s)

    return final_list

def add_to_lists(graph, *L):
    """
    Adds the specified graph to the arbitrary number of lists given as the second through last argument
    Use this function to build the lists of graphs
    """
    for list in L:
            list.append(graph)

def MIR(n):
    if n < 2:
        raise RuntimeError("MIR is defined for n >= 2")
    if n % 2 == 0:
        g = graphs.PathGraph(2)
    else:
        g = graphs.PathGraph(3)
    while g.order() < n:
        new_v = g.add_vertex()
        for v in g.vertices():
            g.add_edge(v, new_v)
        g.add_edge(new_v, g.add_vertex())
    return g

def Ciliate(q, r):
    if q < 1:
        raise RuntimeError("q must be greater than or equal to 1")
    if r < q:
        raise RuntimeError("r must be greater than or equal to q")
    if q == 1:
        return graphs.PathGraph(2*r)
    if q == r:
        return graphs.CycleGraph(2*q)
    g = graphs.CycleGraph(2*q)
    for v in g.vertices():
        g.add_path([v]+[g.add_vertex() for i in [1..r-q]])
    return g

def Antihole(n):
    if n < 5:
        raise RuntimeError("antihole is defined for n > 5")
    return graphs.CycleGraph(n).complement()
