
#GRAPH UTILITIES

def check_independence_extension(g,S):
    """
    Return True if the set S extends to a maximum independent set of the graph g.

    INPUT:

    - ``g`` -- Sage Graph

    - ``S`` -- Sage Set

    OUTPUT:

    - Boolean value, True if the set S extends to a maximum independent set of the graph g, False otherwise.

    EXAMPLES:

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

def find_alpha_critical_graphs(order, save = False):
    """
    Return a list of the graph6 string of each of the alpha critical graphs of the given order.

    A graph g is alpha critical if alpha(g-e) > alpha(g) for every edge e in g. This looks at every graph of the given order, so this will be slow for any order larger than 8.

    INPUT:

    -``order``-- integer; order of a graph

    OUTPUT:

    -A list of the graph6 string of each of the alpha critical graphs of the given order (List).

    EXAMPLES:

    sage: find_alpha_critical_graphs(5)
    ['DUW', 'D~{']

    """
    graphgen = graphs(order)
    alpha_critical_name_list = []
    for g in graphgen:
        if g.is_connected():
            if is_alpha_critical(g):
                alpha_critical_name_list.append(g.graph6_string())
    s = "alpha_critical_name_list_{}".format(order)
    if save:
        save(alpha_critical_name_list, s)
    return alpha_critical_name_list

def is_degree_sequence(L):
    """
    Return True if the list L is the degree sequence of some graph.

    Since a graph always contains at least two vertices of the same degree, a list containing no duplicates cannot be a degree sequence.
    A cycle has all degrees equal to two and exists for any order larger than 3, so a list of twos of length at least 3 is a degree sequence.

    INPUT:

    -``L``--Integer; the possible degree sequence of some graph

    OUTPUT:

    -Boolean, True if the list L is the degree sequence of some graph, False if not.

    EXAMPLES:

    sage: is_degree_sequence([i for i in range(8)])
    False

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
    Return a list of independent sets of size i unioned with their neighborhoods.

    Since this checks all subsets of size i, this is a potentially slow method!

    INPUT:

    -``g``-- Sage Set

    -``i``-- Integer

    OUTPUT:

    - A list of independent sets of size i unioned with their neighborhoods. (List of sets)

    EXAMPLES:

    sage: find_lower_bound_sets(graphs.CycleGraph(6),2)
    [{0, 1, 2, 3, 5},
     {0, 1, 2, 3, 4, 5},
     {0, 1, 3, 4, 5},
     {0, 1, 2, 3, 4},
     {0, 1, 2, 4, 5},
     {1, 2, 3, 4, 5},
     {0, 2, 3, 4, 5}]
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
    print(x_sol)
    return sum(x_sol.values())

#input = graph g
#output = bipartite graph with twice as many nodes and edges
#new nodes are labeled n to 2n-1
#assumes nodes in g are labeled [0..n-1]
#same as cartesian product with k2, but output labeling is guarnateed to be integers
def make_bidouble_graph(g):
    """
    Return a bipartite graph with twice as many nodes and edges

    Assumes nodes in g are labeled [0..n-1], same as cartesian product with k2, but output labeling is guaranteed to be integers.

    INPUT:

    -``g``-- Sage Graph

    OUTPUT:

    - A Bipartite graph with twice as many nodes and edges (Sage Graph)

    EXAMPLES:

    sage: l=make_bidouble_graph(graphs.PetersenGraph())

    sage: l.show()
    """
    n = g.order()
    gdub = Graph(2*n)
    #print "gdub order = {}".format(gdub.order())

    for (i,j) in g.edges(labels = False):
        #print (i,j)
        gdub.add_edge(i,j+n)
        gdub.add_edge(j,i+n)
    return gdub

def pineappleGraph(s,t):
    """
    Return a pineapple with an s vertex clique and t pendants.

    INPUT:

    -``s``-- Integer -- Complete graph with s vertex clique

    -``t``-- Integer -- Number of pendants desired on pineapple graph

    OUTPUT:

    - Sage Graph

    EXAMPLES:

    sage:pineapple_3_4 = pineappleGraph(3,4)
    sage:pineapple_3_4.graph6_string()
    'F{aC?'
    
    sage:pineapple_4_2 = pineappleGraph(4,2)
    sage:pineapple_4_2.graph6_string()
    'E~a?'
    """
    G=graphs.CompleteGraph(s)
    for i in range(t):
        G.add_edge((0,i+s))
    return G

def razborovGraphs(n):
    """
    Return the order n^5 Razborov graph
    
    These have chromatic number >= Theta(n^4) and rank <= O(n^3); as such, they have superlinear chromatic-rank gap, disproving a sequence of conjectures.
    
    INPUT:
    
    -``n``-- Integer
    
    OUTPUT:
    
    - Sage Graph
    
    EXAMPLES:

    sage:razborovGraphs(2)
    Graph on 32 vertices

    sage:razborovGraphs(3)
    Graph on 243 vertices

    sage:razborovGraphs(4)
    Graph on 1024 vertices

    REFERENCES:
    
    -Razborov AA, The gap between the chromatic number of a graph and the rank of its adjacency matrix is superlinear, Disc. Math. 108 (1992) pp393--396.
    """
    B = FiniteEnumeratedSet([1..n])
    C=cartesian_product([B,B,B,B,B])
    G=graphs.EmptyGraph()
    for c in C:
        G.add_vertex(c)
    for a in C:
        for b in C:
            x=[]
            for i in [0..4]:
                if a[i]==b[i]:
                    x.append(0)
                else:
                    x.append(1)
            if not x in [[0,0,0,0,0],[1,1,1,0,0],[1,1,0,1,0],[1,1,0,0,1],[1,1,1,1,0],[1,1,1,0,1],[0,0,1,1,1]]:
                G.add_edge(a,b)
    return G

def neighbors_set(g,S):
    """
    Return the set of neighbors of the set of vertices S in a graph g.

    This may include vertices in S if there are neighbor vertices in such.

    INPUT:

    -``g``-- Sage Graph

    -``S``-- Sage set, list or array

    OUTPUT:

    - Array

    """
    N = []
    for v in S:
        for n in g[v]:
            if n not in N:
                N.append(n)
    return N

def external_neighbors_set(g,S):
    """
    Return the set of external neighbors of the set of vertices S in a graph g.

    This does not include vertices in S if there are neighbor vertices in such.

    INPUT:

    -``g``-- Sage Graph

    -``S``-- Sage set, list or array

    OUTPUT:

    - Array

    """
    N = []
    for v in S:
        for n in g[v]:
            if n not in S:
                if n not in N:
                    N.append(n)
    return N

def closed_neighborhood(g, verts):
    if isinstance(verts, list):
        neighborhood = []
        for v in verts:
            neighborhood += [v] + g.neighbors(v)
        return list(set(neighborhood))
    else:
        return [verts] + g.neighbors(verts)

def is_alpha_critical(g):
    """
    Return whether or not a graph g is alpha critical.

    INPUT:

    -``g``-- Sage Graph

    OUTPUT:

    -Boolean value

    """
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

def MAXINE_independence_heuristic(g):
    """
    Return the length of the independent set without the vertex of maximum degree.

    Delets the vertex of maximum degree as long as the degree > 0.

    INPUT:

    -``g``-- Sage Graph

    OUTPUT:

    -Integer, length of independent set
    """
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
        print ("delta = {}".format(delta))

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

def does_graph_exist(g, L):
    """
    Return true if the given graph exists in the given list.

    It also prints out all graphs in the list that are isomorphic so that duplicates may also be found here.

    INPUT:

    -``g``-- Sage Graph

    -``L``-- List

    OUTPUT:

    -Boolean; true if the given graph exists in the given list, false if not.

    -String; name of graphs in the list that are isomorphic to the original graph.

    EXAMPLES:


    """
    success = False
    for gL in L:
        if g.is_isomorphic(gL):
            print(gL.name())
            success = True
    return success

import itertools
def find_isomorphic_pairs(l):
    """
    Return a list of all pairs of isomorphic graphs in the given list.

    INPUT:

    -``l``-- List; list of graphs

    OUTPUT:

    - List of all pairs of isomorphic graphs in the given list.

    EXAMPLES:


    """
    pairs = []
    L = itertools.combinations(l, r = 2)
    for pair in L:
        if pair[0].is_isomorphic(pair[1]):
            pairs.append(pair)
    return pairs

def find_all_max_ind_sets(g):
    """
    Return a list of all the maximum independent sets

    INPUT:

    -``g``-- Sage Graph

    OUTPUT:

    - List of all the maximum independent sets

    EXAMPLES:


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
    Return an arbitrary number of lists with the specified graph appended to all of them.

    Use this function to build the lists of graphs

    INPUT:

    -``g``-- Sage Graph

    -``L``-- List of lists

    OUTPUT:

    - A list of lists with the specified graph appended to all of them.

    EXAMPLES:

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
        g.add_path([v]+[g.add_vertex() for _ in range(r-q)])
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
        g.add_cycle(range(len_v, len_v + 4))
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
        S = [u for u in g.vertices() if g.has_edge(u,v) and g.has_edge(u,v) and pos[u] > pos[w]]
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

# output = closure of input graph
# Useful: a graph is hamiltonian iff its closure is hamiltonian
def closure(graph):
    """
    Test cases:
        sage: closure(graphs.CycleGraph(4)).is_isomorphic(graphs.CompleteGraph(4))
        True
        sage: closure(graphs.CycleGraph(5)).is_isomorphic(graphs.CycleGraph(5))
        True
    """
    from itertools import combinations
    g = graph.copy()
    while(True):
        flag = False
        deg = g.degree()
        for (v,w) in combinations(g.vertices(), 2):
            if (not g.has_edge(v,w)) and deg[v] + deg[w] >= g.order():
                g.add_edge(v,w)
                flag = True
        if flag == False:
            break
    return g

def is_simplical_vertex(g, v):
    """
    Vertex v is a simplical vertex in g if the induced neighborhood of v is a clique
    """
    neighbors = g.neighbors(v)
    induced_neighborhood = g.subgraph(neighbors)
    return induced_neighborhood.is_clique()

# Defined by Sergey Norin at SIAM DM 2018
def is_homogenous_set(g, s):
    """
    Set of vertices s is homogenous if s induces a clique or s is an independent set.
    """
    induced_g = g.subgraph(s)
    return g.is_independent_set(s) or induced_g.is_clique()

def generalized_degree(g,S):
    """
    The cardinality of the union of the neighborhoods of each v in S.
    """
    neighborhood_union = set(w for v in S for w in g.neighbors(v))
    return len(neighborhood_union)

def common_neighbors_of_set(g, s):
    """
    Returns the vertices in g adjacent to every vertex in s
    """
    if not s:
        return []
    comm_neigh = set(g.neighbors(s[0]))
    for v in s[1:]:
        comm_neigh = comm_neigh.intersection(set(g.neighbors(v)))
    return list(comm_neigh)

def common_neighbors(g, v, w):
    """
    returns the Set of common neighbors of v and w in graph g
        sage: common_neighbors(p4, 0, 3)
        {}
        sage: common_neighbors(p4, 0, 2)
        {1}
    """
    Nv = Set(g.neighbors(v))
    Nw = Set(g.neighbors(w))
    return Nv.intersection(Nw)

def extremal_triangle_free_extension(g):
    """
    Returns graph with edges added until no more possible without creating triangles.
    If input is not triangle-free, raises RuntimeError.

    This function is not deterministic; the output may vary among any of possibly many extremal triangle-free extensions.
    The output is also not necessarily the maximal triangle-free extension.
    """
    if not g.is_triangle_free():
        raise RuntimeError("Graph is not triangle-free")

    g2 = g.copy()
    from itertools import combinations
    for (v,w) in combinations(sample(g2.vertices(), k = g2.order()), 2): # Sample so output not deterministic
        if not g2.has_edge(v, w) and all(u not in g2.neighbors(v) for u in g2.neighbors(w)):
            g2.add_edge(v, w)
    return g2

def pyramid_encapsulation(g):
    """
    Returns the pyramid encapsulation of graph g.

    Let a pyramid be a triangle with each edge bisected, and the midpoints
        joined to form an inner triangle on vertices 0,1,2
    For any graph g, make all its vertices adjacent to 0,1,2.

    The pyramid is a pebbling Class1 graph (pebbling number is order + 1).
    The pyramid encapuslation always yields a Class1 graph.
    """
    pyramid = graphs.CompleteGraph(3)
    pyramid.add_vertices([3, 4, 5])
    pyramid.add_edges([[3,1], [3,0], [4,1], [4,2], [5,0], [5,2]])

    pe = pyramid.disjoint_union(g)
    for v in [0, 1, 2]:
        for w in g.vertices():
            pe.add_edge((0, v), (1,w))
    return pe

def cycle_lengths(g):
    """
    Returns set of all cycle lengths in g - without repetition

    If g is acyclic, returns an empty list.
    Performs depth-first search of all possible cycles.
    """
    lengths = set()
    for init_vertex in g.vertices():
        path_stack = [[init_vertex]]
        while path_stack:
            path = path_stack.pop()
            for neighbor in g.neighbors(path[-1]):
                if neighbor not in path:
                    path_stack.append(path + [neighbor])
                elif neighbor == path[0] and len(path) > 2:
                    lengths.add(len(path))
    return lengths

def max_induced_tree(g):
    """
    Returns *a* maximum-size tree which is an induced subgraph of g

    Raises ValueError if g is not connected, since some invariant theorems assume connected.
    """
    if not g.is_connected():
        raise ValueError("Input graph is not connected")

    from itertools import combinations
    for j in xrange(g.order()):
        for subset in combinations(sample(g.vertices(), k = g.order()), j): # randomize so avg.-case time, not worst-case
            sub_g = g.copy()
            sub_g.delete_vertices(subset)
            if sub_g.is_tree():
                return sub_g

def max_induced_forest(g):
    """
    Returns *a* maximum-size induced subgraph of g which is a forest

    Accepts both connected and disconnected graphs as input.
    """
    from itertools import combinations
    for j in xrange(g.order()):
        for subset in combinations(sample(g.vertices(), k = g.order()), j): # randomize so avg.-case time, not worst-case
            sub_g = g.copy()
            sub_g.delete_vertices(subset)
            if sub_g.is_forest():
                return sub_g

def is_matching(s):
    """
    True if set of edges s is a matching, i.e. no edges share a common vertex

    Ignores edges labels; only compares indices 0 and 1 in edge tuples.
    """
    vertex_list = [v for e in s for v in e[:2]] # Ignore any labels
    if len(vertex_list) != len(set(vertex_list)):
        return False
    else:
        return True

def mobius_ladder(k):
    """
    A mobius ladder with parameter k is a cubic graph on 2k vertices which can
    be constructed by taking a cycle on 2k vertices and connecting opposite
    vertices.

    sage: ml10 = mobius_ladder(10)
    sage: ml10
    mobius_ladder_10: Graph on 20 vertices
    sage: ml10.order()
    20
    sage: ml10.degree()
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    sage: ml10.is_apex()
    True
    sage: ml10.is_vertex_transitive()
    True
    """
    g = graphs.CycleGraph(2*k)
    for i in range(k):
        g.add_edge(i, i+k)
    g.name(new = "mobius_ladder_{}".format(k))
    return g

def benoit_boyd_graphs(a, b, c):
    """
    Two triangles pointed at eachother, with opposite vertices connected by paths of a,b,c respective edges. Triangles weighted 0.5, paths 1.0.

    Pg. 927 of Geneviève Benoit and Sylvia Boyd, Finding the Exact Integrality Gap for Small Traveling Salesman Problems.
        Mathematics of Operations Research, 33(4): 921--931, 2008.
    """
    g = Graph(0, weighted = True)
    for i in xrange(0, a):
        g.add_edge(i, i + 1, 1)
    for i in xrange(a + 1, a + b + 1):
        g.add_edge(i, i + 1, 1)
    for i in xrange(a + b + 2, a + b + c + 2):
        g.add_edge(i, i + 1, 1)
    g.add_edges([(0, a + 1, 0.5), (a + 1, a + b + 2, 0.5), (0, a + b + 2, 0.5)])
    g.add_edges([(a, a + b + 1, 0.5), (a + b + 1, a + b + c + 2, 0.5), (a, a + b + c + 2, 0.5)])
    return g

def benoit_boyd_graphs_2(a, b, c):
    """
    Two triangles pointed at eachother, with opposite vertices connected by paths of a,b,c respective edges. Weights more complicated.

    Paths each weighted 1/a, 1/b, 1/c. The triangles are weighted with the sum of the paths they join, e.g. 1/a+1/b or 1/b+1/c.

    Pg. 928 of Geneviève Benoit and Sylvia Boyd, Finding the Exact Integrality Gap for Small Traveling Salesman Problems.
        Mathematics of Operations Research, 33(4): 921--931, 2008.
    """
    g = Graph(0, weighted = True)
    for i in xrange(0, a):
        g.add_edge(i, i + 1, 1/a)
    for i in xrange(a + 1, a + b + 1):
        g.add_edge(i, i + 1, 1/b)
    for i in xrange(a + b + 2, a + b + c + 2):
        g.add_edge(i, i + 1, 1/c)
    g.add_edges([(0, a + 1, 1/a + 1/b), (a + 1, a + b + 2, 1/b + 1/c), (0, a + b + 2, 1/a + 1/c)])
    g.add_edges([(a, a + b + 1, 1/a + 1/b), (a + b + 1, a + b + c + 2, 1/b + 1/c), (a, a + b + c + 2, 1/a + 1/c)])
    return g

def bipartite_double_cover(g):
    """
    Returns the bipatite double cover of a graph ``g``.

    From :wikipedia:`Bipartite double cover`:
    The bipartite double cover of ``g`` may also be known as the
    Kronecker double cover, canonical double cover or the bipartite double of G.
    For every vertex `v_i` of ``g``, there are two vertices `u_i` and `w_i`.
    Two vertices `u_i` and `w_j` are connected by an edge in the double cover if
    and only if `v_i` and `v_j` are connected by an edge in ``g``.

    EXAMPLES:

        sage: bipartite_double_cover(graphs.PetersenGraph()).is_isomorphic(graphs.DesarguesGraph())
        True

        sage: bipartite_double_cover(graphs.CycleGraph(4)).is_isomorphic(graphs.CycleGraph(4).disjoint_union(graphs.CycleGraph(4)))
        True
    """
    return g.tensor_product(graphs.CompleteGraph(2))

#TESTING

#check for invariant relation that separtates G from class defined by property
def find_separating_invariant_relation(g, objects, property, invariants):
    L = [x for x in objects if (property)(x)]
    for inv1 in invariants:
        for inv2 in invariants:
            if inv1(g) > inv2(g) and all(inv1(x) <= inv2(x) for x in L):
                return inv1.__name__, inv2.__name__
    print ("no separating invariants")



#finds "difficult" graphs for necessary conditions, finds graphs which don't have property but which have all necessary conditions
def test_properties_upper_bound_theory(objects, property, theory):
     for g in objects:
         if not property(g) and all(f(g) for f in theory):
             print (g.name())

#finds "difficult" graphs for sufficient conditions, finds graphs which dont have any sufficient but do have property
def test_properties_lower_bound_theory(objects, property, theory):
     for g in objects:
         if property(g) and not any(f(g) for f in theory):
             print (g.name())

def find_coextensive_properties(objects, properties):
     for p1 in properties:
         for p2 in properties:
             if p1 != p2 and all(p1(g) == p2(g) for g in objects):
                 print (p1.__name__, p2.__name__)
     print ("DONE!")
