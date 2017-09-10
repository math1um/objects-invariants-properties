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
            if v != new_v:
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

def Caro_Roditty(n):
    """
    p.171
    Caro, Y., and Y. Roditty. "On the vertex-independence number and star decomposition of graphs." Ars Combinatoria 20 (1985): 167-180.
    """
    g = graphs.CycleGraph(4)
    iters = 1
    while iters < n:
        len_v = len(g.vertices())
        g.add_cycle([len_v..len_v+3])
        last_cycle = g.vertices()[-4:]
        for v in last_cycle:
            g.add_edge(v, v-4)
        iters += 1
    return g

def find_all_triangles(g):
    E = g.edges()
    A = g.adjacency_matrix()
    pos = {v:p for (p,v) in enumerate(g.vertices())}
    triangles = []

    for e in E:
        v,w = (e[0], e[1]) if pos[e[0]] < pos[e[1]] else (e[1], e[0])
        S = [u for u in g.vertices() if A[u][v] == 1 and A[u][w] == 1 and pos[u] > pos[w]]
        for u in S:
            s = Set([u,v,w])
            triangles.append(s)
    return triangles

# the triangles of a graph g are the vertices of the returned auxilliary graph aux_g
# with edges in aux_g between a pair of vertices in aux_g if the corresponding triangles share a vertex of g
def form_triangles_graph(g):
    vertices = find_all_triangles(g)
    edges = []
    for i in range(len(vertices)-1):
        for j in range(i+1, len(vertices)):
            if not((vertices[i].intersection(vertices[j])).is_empty()):
                edges.append((vertices[i],vertices[j]))
    return Graph([vertices,edges])

def max_bipartite_set(g,s,c):
    #print "s is {}".format(s)
    #print "c is {}".format(c)
    if len(c) == 0:
        return s

    v = c[0]
    #print "v is {}".format(v)
    SCopy = copy(s)
    SCopy.append(v)
    Gprime = g.subgraph(SCopy)

    CCopy = copy(c)
    CCopy.remove(v) #CCopy is C with v removed
    if not(Gprime.is_bipartite()):
        #print "{} is not bipartite".format(SCopy)
        return max_bipartite_set(g, s, CCopy)


    temp1 = max_bipartite_set(g, SCopy, CCopy)
    temp2 = max_bipartite_set(g, s, CCopy)

    if len(temp1) > len(temp2):
        return temp1
    else:
        return temp2


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
