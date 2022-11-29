# Functions used as helpers within Properties

def is_v_twin(g, v):
    """
    Return whether N[v]=N[w] for each neighbor w of v.

    INPUT:

    -``g``-- Sage Graph

    -``v``-- Integer

    OUTPUT:

    -Boolean
    """
    Nv=g.neighbors(v)
    Nvp=Nv+[v]
    for w in Nv:
        Nw=g.neighbors(w)
        Nw.append(w)
        if Set(Nvp) == Set(Nw):
            return True
    return False


# NOT a graph property (won't be added to property lists)
def find_twins_of_vertex(g,v):
    """
    Return a list of non-adjacent vertices that have the same neighbors as v, if a pair exists, or None.

    INPUT:

    -``g``-- Sage graph
    -``v``-- a vertex

    OUTPUT:

    -List

    """
    L = []
    V = g.vertices(sort=true)
    D = g.distance_all_pairs()
    for i in range(g.order()):
        w = V[i]
        if D[v][w] == 2 and g.neighbors(v) == g.neighbors(w):
                L.append(w)
    return L


# NOTE: this function returns vertices, not True/False, so its NOT a property, and not added to lists
def find_twin(g):
    """
    If twin vertices are found in graph g, return those vertices.
    Note, this will return two vertices, not a complete set of twin vertices, should more than one set exist.

    Two vertices are twins if they are non-adjacent and have the same neighbors.

    INPUT:

    -``g``-- Sage graph

    OUTPUT:

    -Pair of vertices or None

    """
    V = g.vertices(sort=true)
    for v in V:
        Nv = set(g.neighbors(v))
        for w in V:
            Nw = set(g.neighbors(w))
            if v not in Nw and Nv == Nw:
                return (v,w)
    return None


# update this description!
# NOTE: this function returns vertices, not True/False, so its NOT a property, and not added to lists
def find_neighbor_twins(g):
    """

    Two vertices are twins if they are non-adjacent and have the same neighbors. So same neighbors AND adjacent.

    INPUT:

    -``g``-- Sage graph

    OUTPUT:

    -Pair of vertices or None

    """
    V = g.vertices(sort=true)
    for v in V:
        Nv = g.neighbors(v)
        for w in Nv:
            if set(closed_neighborhood(g,v)) == set(closed_neighborhood(g,w)):
                return (v,w)
    return None


# given graph g and subset S, looks for any neighbor twin of any vertex in T
# if result = T, then no twins, else the result is maximal, but not necessarily unique
# NOTE: this function returns vertices, not True/False, so its NOT a property, and not added to lists
def find_neighbor_twin(g, T):
    """
    Given a graph g, and a subset S, looks for any neighbor twin of any vertex in T.
    If result equals T, no twins were found.  Otherwise, the result is maximal but not unique.

    INPUT:

    -``g``-- Sage graph
    -``T``-- subset of g (?????????????????)

    OUTPUT:

    -none

    """
    gT = g.subgraph(T)
    for v in T:
        condition = False
        Nv = set(g.neighbors(v))
        #print("v = {}, Nv = {}".format(v,Nv))
        NvT = set(gT.neighbors(v))
        for w in Nv:
            NwT = set(g.neighbors(w)).intersection(set(T))
            if w not in T and NvT.issubset(NwT):
                T.append(w)
                condition = True
                #print("TWINS: v = {}, w = {}, sp3 = {}".format(v,w,sp3))
                break
        if condition == True:
            break


# if result = T, then no twins, else the result is maximal, but not necessarily unique
# NOTE: this function returns vertices, not True/False, so its NOT a property, and not added to lists
def iterative_neighbor_twins(g, T):
    T2 = copy(T)
    find_neighbor_twin(g, T)
    while T2 != T:
        T2 = copy(T)
        find_neighbor_twin(g, T)
    return T


# NOTE: this requires a vertex as an input so its NOT a graph property, not in a property list
def has_Havel_Hakimi_property(g, v):
    """
    Return whether the vertex v in the graph g has the Havel-Hakimi property.

    A vertex has the Havel-Hakimi property if it has maximum degree and the minimum degree of its neighbours is at least the maximum degree of its non-neigbors. Graphs with the strong Havel-Hakimi property, M. Barrus, G. Molnar, Graphs and Combinatorics, 2016, http://dx.doi.org/10.1007/s00373-015-1674-7

    INPUT:

    -``g``-- Sage Graph

    -``v``-- Integer

    OUTPUT:

    -Boolean, True if the vertex v in the graph g has the Havel-Hakimi property, False otherwise.

    EXAMPLE:

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
    if len(g.neighbors(v)) == len(g.vertices(sort=true)) - 1: return True

    return (min(g.degree(nv) for nv in g.neighbors(v)) >=
        max(g.degree(nnv) for nnv in g.vertices(sort=true) if nnv != v and nnv not in g.neighbors(v)))


# NOT a graph property (for any fixed k, this IS a graph property)
def is_k_bootstrap_good(G,k):
    """
    Return whether or not there exists a set of k vertices such that G is fully infected.

    Assumes G has at least k vertices.

    If G has more than k vertices, it must be connected to be k_bootstrap_good.

    INPUT:

    -``g``--Sage Graph

    -``k``-- Integer

    OUTPUT:

    -Boolean
    """
    G.relabel()
    for s in itertools.combinations(G.vertices(sort=true), k):
        if k_percolate(G,set(s),k):
            return True
    return False

# NOT a graph property (requires other parameters)
def k_percolate(G,infected,k):
    """
    Return True if the set 'infected' fully k-infects the graph G

    INPUT:

    -``G``--Sage Graph

    -``infected``--List

    -``k``--Integer

    OUTPUT:

    -Boolean
    """
    uninfected = set(G.vertices(sort=true)) - set(infected)
    newInfections = True
    while newInfections:
        newInfections = False
        for v in uninfected:
            if len(set(G.neighbors(v)).intersection(infected)) >= k:
                infected.add(v)
                uninfected-=set([v])
                newInfections = True
                break
    return len(uninfected) == 0


######################################################################################################################
#Below are some factory methods which create properties based on invariants or other properties

def has_equal_invariants(invar1, invar2, name=None):
    """
    This function takes two invariants as an argument and returns the property that these invariants are equal.
    Optionally, a name for the new function can be provided as a third argument.

    INPUT:

    -``invar1, invar2``-- Two mathematical invariants.  Not strictly typed.
    -``name``-- Optional.  String

    OUTPUT:

    -Boolean

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


# NOT a graph property
def has_invariant_equal_to(invar, value, name=None, documentation=None):
    """
    This function takes an invariant and a value as arguments and returns the property
    that the invariant value for a graph is equal to the provided value.

    Optionally a name and documentation for the new function can be provided.

    INPUT:

    -``invar``-- A mathematical invariants.  Not strictly typed.
    -``value``-- Numerical value

    -``name``-- Optional.  String
    -``documentation``-- Optional.  String

    OUTPUT:

    -Boolean

    EXAMPLES:
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

# NOT a graph property
def has_leq_invariants(invar1, invar2, name=None):
    """
     This function takes two invariants as an argument and returns the property that the first invariant is
    less than or equal to the second invariant.
    Optionally a name for the new function can be provided as a third argument.

    INPUT:

    -``invar1, invar2``-- Two mathematical invariants.  Not strictly typed.

    -``name``-- Optional.  String

    OUTPUT:

    -Boolean
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

#NOT a graph property
def localise(f, name=None, documentation=None):
    """
    This function takes a property (i.e., a function taking only a graph as an argument) and
    returns the local variant of that property. The local variant is True if the property is
    True for the neighbourhood of each vertex and False otherwise.
    """
    # create a local version of f
    def localised_function(g):
        return all((f(g.subgraph(g.neighbors(v))) if g.neighbors(v) else True) for v in g.vertices(sort=true))

    # we set a nice name for the new function
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

#any localized property for an efficient property is itself efficient


is_locally_bipartite = localise(Graph.is_bipartite)
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
#PROPERTY LISTS

#TO-DO: add every property above to one of the following lists (or more if appropriate):
# efficiently_computable_properties
# efficiently_computable_sage_properties
# intractable_properties
# intractable_sage_properties

# Syntax: after each defined property, we need:
# add_to_lists(just_defined_property_name, list1, list2, etc)

#have any new Sage properties been defined?
efficiently_computable_sage_properties = [Graph.is_regular, Graph.is_planar,
Graph.is_forest, Graph.is_eulerian, Graph.is_connected, Graph.is_clique,
Graph.is_circular_planar, Graph.is_chordal, Graph.is_bipartite,
Graph.is_cartesian_product,Graph.is_distance_regular,  Graph.is_even_hole_free,
Graph.is_gallai_tree, Graph.is_line_graph, Graph.is_overfull, Graph.is_perfect,
Graph.is_split, Graph.is_strongly_regular, Graph.is_triangle_free,
Graph.is_weakly_chordal, Graph.is_circulant, Graph.has_loops,
Graph.is_asteroidal_triple_free, Graph.is_block_graph, Graph.is_cactus,
Graph.is_cograph, Graph.is_long_antihole_free, Graph.is_long_hole_free, Graph.is_partial_cube,
Graph.is_polyhedral, Graph.is_prime, Graph.is_tree, Graph.is_apex, Graph.is_arc_transitive,
Graph.is_self_complementary]

efficiently_computable_properties = efficiently_computable_properties + efficiently_computable_sage_properties

intractable_sage_properties = []

#have any new Sage intractable properties been defined?
intractable_properties = intractable_properties + intractable_sage_properties

removed_properties = [is_pebbling_class0]

#are all of these already in one of the Sage lists above?
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


properties = efficiently_computable_properties + intractable_properties
properties_plus = efficiently_computable_properties + intractable_properties + invariant_relation_properties

invariants_from_properties = [make_invariant_from_property(property) for property in properties]
invariants_plus = all_invariants + invariants_from_properties

# weakly_chordal = weakly chordal, i.e., the graph and its complement have no induced cycle of length at least 5
