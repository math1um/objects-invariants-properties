#OTHER UTILITIES

#mean, median and mode were deprecated from the Sage root
#we import them from Numpy as a (hopefully temporary) gack

from numpy import mean

from numpy import median

from statistics import mode


#GRAPH UTILITIES


def anti_degree(g,v): #the number of non-neighbors of v
    n=g.order()
    d=g.degree(v)
    return (n-1)-d

def remove_low_anti_degree_vertices(g,Lbound):
    """
    remove all vertices with anti-degree less than lower bound-1

    if bound is an upper bound for independence number, any vertex v in a
    maximum independent set must have at least lower bound-1 anti-neighbors
    """
    V=g.vertices()
    for v in g.vertices():
        if anti_degree(g,v) < Lbound-1:
            V.remove(v)
    H=g.subgraph(V)
    return H

def non_neighbors(g, v ,w):
    """
    return the vertices that are not v, w nor any neighbor of them

    these are the only vertices that could be in an independent set that contains v and w
    """
    Nv = Set(g.neighbors(v))
    Nw = Set(g.neighbors(w))
    U = Nv.union(Nw)
    V = Set(g.vertices())

    return list(V.difference(U))

def add_non_independent_edges(g, Lbound):
    """
    if v and w don't have lower bound-2 non-neighbors (potential independent set vertices)
    then v and w can't both be an an independent set with bound vertices, so can be made adjacent

    #add v-w edge for each such pair
    """
    h=copy(g)
    V=g.vertices()
    n=g.order()
    for i in srange(n):
        for j in srange(n):
            if i>j: #avoid duplication
                v=V[i]
                w=V[j]
                U = non_neighbors(g,v,w) #a list
                if len(U) < Lbound - 2: #so v,w can't be in an independent set with bound vertices
                    h.add_edge(v,w)
    return h

def find_2_bicritical_part(g):
    """
    returns unique 2-bicritical (=independence irreducible) subgraph

    from Larson, Independence Decomposition Theorem
    """
    X = find_KE_part(g)
    SX = Set(X)
    Vertices = Set(g.vertices())

    return g.subgraph(Vertices.difference(SX))


def independent_set_preprocesser(g, Lbound=1): #default initialization of 1<-alpha<=Infinity
    """
    returns a graph that will have the same independence number

    1. finds maximum critical independent set and removes neighbors
    2. removes vertices that can't be in a maximum independent set because they don't have enough anti-neighbors
    3. adds edges between pairs of vertices that can't both be in a maximum indendent set because they don't have enough common anti-neighbors

    NOTE: this will return just an independent set for a KE graph

    Steps (2) and (3) are discussed in Walteros & Buchanan, 2020, "Why is a Maximum Clique Often Easy in Practice?"
    """

    h = copy(g)

    I=find_max_critical_independent_set(h)
    X=closed_neighborhood(h,I) #the vertices I plus their neighbors (is a KE subgraph)
    SX = Set(X)
    SV = Set(h.vertices())
    SXc = SV.difference(SX) #vertices in the complement Xc of X
    Sh = SXc.union(Set(I)) # critical independent set vertices I plus 2-bicritcal subgraph vertices Xc
    h=g.subgraph(list(Sh)) #this is the 2-bicritical part of g unioned with maximum critical independent set

    while True:

        beginning_loop_order = h.order()
        h = remove_low_anti_degree_vertices(h, Lbound)

        beginning_loop_size = h.size()
        h = add_non_independent_edges(h, Lbound)

        if beginning_loop_order == h.order() and beginning_loop_size == h.size():
            break
    return h

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

def add_to_lists(x, *L):
    """
    Return an arbitrary number of lists with the specified arbitrary object appended to all of them.

    Use this function to build the lists of graphs

    INPUT:

    -``x``-- any Sage object

    -``*L``-- List of lists

    OUTPUT:

    - A list of lists with the specified object appended to all of them.

    EXAMPLES:

    """
    for list in L:
            list.append(x)

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

def is_simplicial_vertex(g, v):
    """
    Vertex v is a simplicial vertex in g if the induced neighborhood of v is a clique
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
    for j in range(g.order()):
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
    for j in range(g.order()):
        for subset in combinations(sample(g.vertices(), k = g.order()), j): # randomize so avg.-case time, not worst-case
            sub_g = g.copy()
            sub_g.delete_vertices(subset)
            if sub_g.is_forest():
                return sub_g

"""
def is_matching(s): #this version works for SETS of edges. another version in GT tests if a GRAPH has only degree one edges

    True if set of edges s is a matching, i.e. no edges share a common vertex

    Ignores edges labels; only compares indices 0 and 1 in edge tuples.

    vertex_list = []
    for e in s:
        vertex_list.append(e[0])
        vertex_list.append(e[1])
    if len(vertex_list) != len(set(vertex_list)):
        return False
    else:
        return True
"""

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
    for i in range(0, a):
        g.add_edge(i, i + 1, 1)
    for i in range(a + 1, a + b + 1):
        g.add_edge(i, i + 1, 1)
    for i in range(a + b + 2, a + b + c + 2):
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
    for i in range(0, a):
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

def make_alpha_critical(g):
    """
    Return a connected alpha_critical_subgraph with same independence number as g.

    Assumes g is connected.

    INPUT:

    -``g``--Sage Graph

    OUTPUT:

    -Sage Graph
    """
    alpha = independence_number(g)

    E = g.edge_iterator(labels=False)
    for (v,w) in E:
        h = copy(g)
        h.delete_edge(v,w)
        if h.is_connected() and (alpha == independence_number(h)):
            g = h
    return g

###DEMING DECOMPOSITION

def make_deming_subgraph(H,Blossom1,Blossom2): #don't want induced edges - just blossom edges
    V1 = Blossom1.vertices()
    S1 = Set(V1)
    V2 = Blossom2.vertices()
    S2 = Set(V2)
    S = S1.union(S2)

    E1 = Blossom1.edges()
    E1 = [(e[0],e[1]) for e in E1]
    SE1 = Set(E1)
    E2 = Blossom2.edges()
    E2 = [(e[0],e[1]) for e in E2]
    SE2 = Set(E2)
    SE = SE1.union(SE2)

    V = list(S)
    E = list(SE)
    return H.subgraph(vertices=V, edges=E)

def make_deming_subgraph_induced(H,Blossom1,Blossom2): #want induced edges - just blossom edges
    V1 = Blossom1.vertices()
    S1 = Set(V1)
    V2 = Blossom2.vertices()
    S2 = Set(V2)
    S = S1.union(S2)
    V = list(S)
    return H.subgraph(vertices=V)

def deming_main(g):
    if not g.has_perfect_matching():
        #print("graph does not have perfect matching!")
        return

    M=[(e[0],e[1]) for e in g.matching()] #M is a perfect matching, with no edge labels

    #print("in deming_main, M = {}".format(M))

    return deming_decomposition(g,M)

def deming_decomposition(g,M):

    Deming_subgraphs = []
    h = copy(g) #g never changes in this function, M never changes either

    S,BlossomTip1,Blossom1,Blossom2 = demings_algorithm(h,M)
    #if Blossom1 = empty, or |S|=n/2 then g must be KE

    while Blossom1.order() != 0: #so there IS a deming subgraph
        V = h.vertices()
        D = make_deming_subgraph_induced(g,Blossom1,Blossom2) #should have perfect matching with M edges

        minD = get_min_deming_subgraph(D)
        Deming_subgraphs.append(minD)

        newV = [v for v in V if v not in minD.vertices()] #remove the vertices in just-found deming subgraph
        #print("in deming_decomposition main loop: newV = {}".format(newV))

        h = g.subgraph(newV) #h may be empty graph here
        newM = [(v,w) for (v,w,u) in h.matching()] #OK if h is empty
        S,BlossomTip1,Blossom1,Blossom2 = demings_algorithm(h,newM) #OK if h is empty (Blossom1 = Graph(0)
        #print("in deming_decomposition main loop: got to end of loop")

    #done with while loop, remaining graph must be KE
    K = g.subgraph(h.vertices()) #the h from the last loop, should be KE with perfect matching
    return Deming_subgraphs, K

def demings_algorithm(g,M):
    #assume g has a perfect matching (if not STOP)
    #1. find perfect matching M
    #2. choose any matching edge xy. let x be red, y be blue
    #3. look for uncolored neighbors of x, color one red and continue
    #4. if no uncolored neighbors then go to any uncolored vertex and continue
    #5. every time you color a red vertex, see if it has a red neighbor. if it does there is blossom
    #6. keep track of prdecessors, trace it back and find the blossow tip z.
    #7. now we'll try to find another blossom in the graph, starting at z, but with z blue
    #7. z was red. now color it blue, color its matched neighbor red
    #8. continue. keep track of the path. if another blossom is found, that's it
    #9. the vertices from the two blossoms and the path must either be an even K4 sub or an even T-sub

    def find_blossom(H,Pred,x,x_end): #find all edges including stem, union them, return graph,

        #print("in find_blossom: Pred = {}, x = {}, x_end = {}".format(Pred,x,x_end))

        BV = [x_end] #blossom vertices,
        BE = [(x,x_end)] #blossom edges, these are the adjacent red vertices

        z = x_end
        while Pred[z] != -1: #adding the blossom "stem"
            y = Pred[z] #x is red, so *some* pred y *must* exist (namely matched blue vertex)
            BV.append(y)
            BE.append((z,y))
            z = y

        BV.append(x) #may be doubles
        z = x
        while Pred[z] != -1: #adding the blossom loop,
            y = Pred[z]
            BV.append(y)
            BE.append((z,y))
            z = y

        newBE = []
        for (v,w) in BE:
            if not (v,w) in newBE and not (w,v) in newBE:
                newBE.append((v,w))

        #print("in find_blossom: Blossom edges = {}".format(newBE))

        return H.subgraph(vertices=list(Set(BV)),edges=newBE)

    def find_blossom_tip(Blossom): #must be degree 1 vertices - or there is a problem

        #print("In Find_BlossomTip: Blossom vertices = {}".format(Blossom.vertices()))

        E = [(v,w) for (v,w,u) in Blossom.edges()]

        g = copy(Blossom)
        gV = g.vertices()

        #print("in find_blossom_tip: min(g.degree())={}".format(min(g.degree())))
        while min(g.degree())<2: #there must be one or PROBLEM
            for v in gV:
                N = g.neighbors(v) #should only be one vertex wtih degree 1
                if len(N)==1:
                    tip = N[0]
                    gV.remove(v) #keep removing degree 1 vertices, keep track of neighbor
                    g = g.subgraph(gV)
                    break

        return tip

    def step1(M,H,S,FLAG,Colors,Pred,BlossomTip1, Blossom1, Blossom2): #finished, or pick heavy edge and extend coloring

        #print("in Step 1")
        #print(H.vertices())

        if len(H.vertices())==0: #no vertices left, S is a maximum independent set
            #print("done!, S = {}".format(S))
            return M,H,S,FLAG,Colors,Pred,BlossomTip1, Blossom1, Blossom2

        else:
            EH = H.edges(labels=False)
            heavy_H = [e for e in M if ((e[0],e[1]) in EH or (e[1],e[0]) in EH)]

            #print("Step 1: heavy_H = {}".format(heavy_H))

            e = heavy_H[0] #just first heavy edge
            x = e[0] #x is random endpoint of heavy edge
            y = e[1]
            Colors[x] = "red" #arbitrary endpoint choice
            Colors[y] = "blue"
            Pred[x]=y #this will be part of stem of Blossom 1, and guarantee a degree 1 vertex

            #print("Step 1: Colors = {}".format(Colors))

            FLAG = 0 #if FLAG = 2, there is one discovered blossom tip

            #GOTO
            return step2(M,H,S,FLAG,Colors,Pred,BlossomTip1, Blossom1, Blossom2)

    def step2(M,H,S,FLAG,Colors,Pred,BlossomTip1, Blossom1, Blossom2): #look at current coloring

        #print("in Step 2")
        #print(H.vertices())
        #print("Step2: Colors = {}".format(Colors))

        VH = H.vertices() #v/
        Red = [u for u in VH if Colors[u]=="red"] #v/
        Uncolored = [v for v in VH if Colors[v]=="uncolored"] #what happens if Uncolored is empty??
        #print("Step2: Uncolored = {}".format(Uncolored))

        EH = H.edges(labels=False)
        RedUncov = [(u,v) for u in Red for v in Uncolored if ((u,v) in EH or (v,u) in EH)] #light edges
        RedRed = [(v,w) for v in Red for w in Red if (v,w) in EH]

        #print("Step 2: RedRed = {}".format(RedRed))
        #print("Step 2: RedUncov = {}".format(RedUncov))

        if len(RedUncov)==0: #if red vertices adjacent only to colored vertices, add them to S. ONLY reduction step

            S = S + Red
            VH = [v for v in VH if Colors[v]=="uncolored"] #new VH, resetting by uncoloring everything
            H = H.subgraph(VH) #new REDUCED H
            # RESET PRED?
            Pred = {v:-1 for v in VH} #initialization of Pred for smaller subgraph

            #GOTO
            return step1(M,H,S,FLAG,Colors,Pred,-1, Graph(0), Graph(0))

        if len(RedUncov) > 0: #so red u is adjacent to uncovered v, uv is a light edge

            e = RedUncov[0] #e is a light edge

            #print("in Step 2, RedUncov > 0, Colors = {}, e = {}".format(Colors, e))

            u = e[0] #u is Red, v is uncolored
            v = e[1] #must exist, as RedUncov is not empty
            #print("in Step 2: u={}, v = {}".format(u,v))
            #there must be a heavy edge v-w where w is uncolored
            heavy_H = [e for e in EH if ((e[0],e[1]) in M or (e[1],e[0]) in M)]

            #print("Step 2: heavy_H = {}".format(heavy_H))
            for e in heavy_H: #find a heavy edge with v as endpoint, set w = other end
                #must exist, as M is perfect matching, v must have matched vertex
                #print(e)
                if e[0] == v:
                    w = e[1]
                if e[1] == v:
                    w = e[0]
            #print("Step 2: w is {}".format(w)) #wait till for loop is finished!

            Colors[v] = "blue" #extend the coloring

            #is this the "w" referenced in the error?? YES. CHECKED!
            #print("Step 2: step before w is referenced. w = {}".format(w))

            Colors[w] = "red"

            #print("Step 2: extending coloring. Colors = {}".format(Colors))

            # RESET PRED?
            #Pred = {v:-1 for v in VH} #initialization

            Pred[v] = u
            Pred[w] = v

            #GOTO
            return step3(M,H,S,FLAG,Colors,Pred,BlossomTip1, Blossom1, Blossom2)

    def step3(M,H,S,FLAG,Colors,Pred,BlossomTip1, Blossom1, Blossom2): #is there a blossom?

        #print("in Step 3: H.vertices = {}".format(H.vertices()))

        VH = H.vertices()
        EH = H.edges(labels=False)
        Red = [v for v in VH if Colors[v]=="red"]
        Uncolored = [v for v in VH if Colors[v]=="uncolored"]
        RedUncov = [(u,v) for u in Red for v in Uncolored if ((u,v) in EH or (v,u) in EH)]
        RedRed = [(v,w) for v in Red for w in Red if (v,w) in EH]

        if len(RedRed) == 0: #no blossom yet, extend S or keep coloring

            #GOTO
            return step2(M,H,S,FLAG,Colors,Pred, BlossomTip1, Blossom1, Blossom2)

        if len(RedRed) > 0: #so THERE IS a blossom

            #print("Step 3: RedRed > 0")

            if FLAG == 0: #we now have FIRST tip

                #print("Step 3: FLAG == 0")

                e = RedRed[0] #first edge in RedRed
                x = e[0] #just names for the endpoints of this edge
                x_end = e[1]

                Blossom1 = find_blossom(H,Pred,x,x_end)
                BlossomTip1 = find_blossom_tip(Blossom1)

                #find BlossomTip1 matched vertex
                #if there is a perfect matching, BlossomTip1 has a heavy edge neighbor - which shouldn't be in blossom

                matched_edges = [(BlossomTip1,w) for w in VH if ((BlossomTip1,w) in M or (w,BlossomTip1) in M)]
                e = matched_edges[0] #matched edegs should have exactly on member
                if e[0]==BlossomTip1:
                    y = e[1]
                else:
                    y = e[0]

                # RESET PRED.? YES, new BlossomTip, going backwards now
                Pred = {v:-1 for v in VH} #initialization
                Pred[y] = BlossomTip1

                #reset Colors
                Colors = {v:"uncolored" for v in VH}
                Colors[BlossomTip1] = "blue"
                Colors[y] = "red"

                FLAG = 2 #1st blossom flag!
                #BlossomTip1 = x

                #GOTO
                return step2(M,H,S,FLAG,Colors,Pred, BlossomTip1, Blossom1, Blossom2)

            if FLAG == 2: #there IS a 2nd blossom - already had one blossom tip, now find the other

                #print("Step 3: FLAG == 2")

                #should be able to just count back from RED NEIGHBOR of other blossom tip

                e = RedRed[0] #this edge will be included in the blossom
                x = e[0]
                x_far = e[1]

                #print("To find_Blossom2: Colors = {}".format(Colors))

                Blossom2 = find_blossom(H,Pred,x,x_far) #we're done now!

                #BlossomTip2 = x

                return M,H,S,FLAG,Colors,Pred,BlossomTip1, Blossom1, Blossom2





    #g may be empty, handling this case
    if g.order() == 0:
        return [],-1,Graph(0),Graph(0)

    V = g.vertices()
    E = g.edges(labels=False)
    M = [(e[0],e[1]) for e in M]
    #print(M)
    M_vertices = get_vertices_from_edges(M) #a list

    S = [] #initialize
    H = g.subgraph(V) #H = G at beginning
    VH = H.vertices()
    EH = H.edges(labels=False)

    heavy_H = [e for e in EH if ((e[0],e[1]) in M or (e[1],e[0]) in M)]
    #light_H = [e for e in EH if not e in M]
    FLAG = -1 #initialization
    Colors = {v:"uncolored" for v in V} #initialization
    #print("in Demings_algorithms: Colors = {}".format(Colors))

    Pred = {v:-1 for v in V} #initialization

    BlossomTip1 = -1 #initialization

    Blossom1 = Graph(0)
    Blossom2 = Graph(0)

    #step1 - start flowing through flow diagram
    #print("in MAIN: ")
    #print(M,H,S,FLAG,Colors,Pred,BlossomTip1,Blossom1,Blossom2)
    M,H,S,FLAG,Colors,Pred,BlossomTip1,Blossom1,Blossom2=step1(M,H,S,FLAG,Colors,Pred,BlossomTip1,Blossom1,Blossom2)

    #print("S,BlossomTip1,Blossom1,Blossom2 = ")
    return S,BlossomTip1,Blossom1,Blossom2

#DEMING AUXILLIARY

def deming_subgraph_min_test(g): #if g is a deming or blossom block, must have for every perfect matching edge xy, that g-{x,y} is KE
    M = g.matching()
    nu = len(M)

    if 2*nu != g.order():
        #print("deming_subgraph_min_test: graph has no perfect matching!")
        return None

    pm_edges = perfect_matching_edges(g)
    for e in pm_edges:
        #print("in deming_subgraph_min_test: e = {}".format(e))

        x = e[0] # xy in perfect matching iff G-{x,y} has perfect matching
        y = e[1]
        V = g.vertices()
        V.remove(x)
        V.remove(y)
        h = g.subgraph(V)
        if not is_KE(h):
            return e #return edge e such that G-e is NOT KE, and we can reapply Deming's algorithm to g-e

    #print("in deming_subgraph_min_test: returning NONE")

    return None #in this case g can't be reduced

def get_min_deming_subgraph(g): #takes deming subgraph as input, keeps reducing

    xy = deming_subgraph_min_test(g)
    #print("in get_min_deming_subgraph, xy = {}".format(xy))
    h = copy(g)

    #if xy == None, then g is reduced
    while xy != None:
        #print("in get_min_deming_subgraph: xy = {}".format(xy))

        V = h.vertices()
        x = xy[0]
        y = xy[1]
        V.remove(x)
        V.remove(y)
        #now we need a Deming subgraph component
        #can use Deming algorithm on D-{x,y} even if its not connected
        h = g.subgraph(V)
        M = [(v,w) for (v,w,u) in h.matching()]
        S,BlossomTip1,Blossom1,Blossom2 = demings_algorithm(h,M)
        h = make_deming_subgraph_induced(g,Blossom1,Blossom2)
        xy = deming_subgraph_min_test(h)

    #print("in get_min_deming_subgraph: h.vertices = {}".format(h.vertices()))

    return h

def is_deming_blossom_pair_graph(g): #has perfect matching, spanning blossom, is min_deming (also alpha = nu-1)

    M = g.matching()
    nu = len(M)

    if 2*nu != g.order():
        #print("in is_deming_blossom_pair: graph has no perfect matching!")
        return None

    #get deming subgraph, make blossom, check that it has cut edge
    S,BlossomTip,Blossom1,Blossom2 = demings_algorithm(g,M)
    h = make_deming_subgraph(g,Blossom1,Blossom2) #not induced, just what you get from the matching

    if not has_cut_vertex(h): #a blossom pair must have a cut vertex
        return False
    if h.order() != g.order(): #the blossom pair must *span* g
        return False

    xy = deming_subgraph_min_test(g) #will be None if g is not reducible
    if xy != None:
        return False

    return True #all tests passed

def is_deming_K4_graph(g): #has perfect matching, spanning but not spanning blossom, is min deming (also alpha = nu-1)

    M = g.matching()
    nu = len(M)

    if 2*nu != g.order():
        #print("in is_deming_blossom_pair: graph has no perfect matching!")
        return None

    #get deming subgraph, make blossom, check that it has cut edge
    S,BlossomTip,Blossom1,Blossom2 = demings_algorithm(g,M)
    h = make_deming_subgraph(g,Blossom1,Blossom2) #not induced, just what you get from the matching

    if h.edge_connectivity() == 1: #in this case deming subgraph is blossom, not K4
        return False
    if h.order() != g.order(): #the blossom pair must *span* g
        return False

    xy = deming_subgraph_min_test(g) #will be None if g is not reducible
    if xy != None:
        return False

    return True #all tests passed

###INDEPENDENCE and MATCHING THEORY

def perfect_matching_edges(g): #if g has a perfect matching, output a list of all edges in *some* perfect matching

    pm_edges = [] #perfect matching edges
    test_edges = [(e[0],e[1]) for e in g.edges()]

    M = g.matching()
    nu = len(M)

    if 2*nu == g.order():
        pm_edges = [(e[0],e[1]) for e in M]
    else:
        #print("in perfect_matching_edges: graph has no perfect matching!")
        return pm_edges #which will still be empty

    test_edges = [(e[0],e[1]) for e in test_edges if ((e[0],e[1]) not in pm_edges and (e[1],e[0]) not in pm_edges)]
    for e in test_edges:
        x = e[0] # xy in perfect matching iff G-{x,y} has perfect matching
        y = e[1]
        V = g.vertices()
        V.remove(x)
        V.remove(y)
        h = g.subgraph(V)
        Mh = h.matching()
        nuh = len(Mh)
        if 2*nuh == h.order():
            pm_edges.append(e)

    return pm_edges


###AUXILLIARY FUNCTIONS

#given deming subgraph D with perfect matching, look for edge xy so that D-{x,y} is not KE
#return None if no xy
#else return xy

def get_vertices_from_edges(edge_set):
    S = Set([])
    for e in edge_set:
        #print e
        if e[0] not in S:
            S = S.union(Set([e[0]]))
        if e[1] not in S:
            S = S.union(Set([e[1]]))
    return list(S)
