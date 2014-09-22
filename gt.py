
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
    p = MixedIntegerLinearProgram(maximization=True)
    x = p.new_variable(nonnegative=True)
    p.set_objective(sum(x[v] for v in g.vertices()))

    for v in g.vertices():
        p.add_constraint(x[v], max=1)

    for (u,v) in g.edge_iterator(labels=False):
        p.add_constraint(x[u] + x[v], max=1)

    return p.solve()

@memoized
def lovasz_theta(g):
    import cvxopt.base
    import cvxopt.solvers

    cvxopt.solvers.options['show_progress'] = False
    cvxopt.solvers.options['abstol'] = float(1e-10)
    cvxopt.solvers.options['reltol'] = float(1e-10)

    gc = g.complement()
    n = gc.order()
    m = gc.size()

    if n == 1:
        return 1.0

    #the definition of Xrow assumes that the vertices are integers from 0 to n-1, so we relabel the graph
    gc.relabel()

    d = m + n
    c = -1 * cvxopt.base.matrix([0.0]*(n-1) + [2.0]*(d-n))
    Xrow = [i*(1+n) for i in xrange(n-1)] + [b+a*n for (a, b) in gc.edge_iterator(labels=False)]
    Xcol = range(n-1) + range(d-1)[n-1:]
    X = cvxopt.base.spmatrix(1.0, Xrow, Xcol, (n*n, d-1))

    for i in xrange(n-1):
        X[n*n-1, i] = -1.0

    sol = cvxopt.solvers.sdp(c, Gs=[-X], hs=[-cvxopt.base.matrix([0.0]*(n*n-1) + [-1.0], (n,n))])
    v = 1.0 + cvxopt.base.matrix(-c, (1, d-1)) * sol['x']

    # TODO: Rounding here is a total hack, sometimes it can come in slightly
    # under the analytical answer, for example, 2.999998 instead of 3, which
    # screws up the floor() call when checking difficult graphs.
    return round(v[0], 3)

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
    return max(g.spectrum())

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

@memoized
def chromatic_index(g):
    if g.size() == 0:
        return 0
    import sage.graphs.graph_coloring
    return sage.graphs.graph_coloring.edge_coloring(g, value_only=True)

def card_max_cut(g):
    return g.max_cut(value_only=True)

@memoized
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


efficiently_computable_invariants = [Graph.average_distance, Graph.diameter, Graph.radius, Graph.girth,  Graph.order, Graph.size, Graph.szeged_index, Graph.wiener_index, min_degree, max_degree, Graph.average_degree, matching_number, residue, annihilation_number, fractional_alpha, lovasz_theta, cvetkovic, cycle_space_dimension, card_center, card_periphery, max_eigenvalue, kirchhoff_index, largest_singular_value, Graph.vertex_connectivity, Graph.edge_connectivity, Graph.maximum_average_degree, Graph.density, welsh_powell, wilf, brooks]

intractable_invariants = [independence_number, domination_number, chromatic_index, Graph.clique_number, clique_covering_number, n_over_alpha, memoized(Graph.chromatic_number)]

#FAST ENOUGH (tested for graphs on 140921): lovasz_theta, clique_covering_number, all efficiently_computable
#SLOW but FIXED for SpecialGraphs

invariants = efficiently_computable_invariants + intractable_invariants

#removed for speed: Graph.treewidth, card_max_cut

#set precomputed values
#add_to_cache(Graph.treewidth, graphs.BuckyBall(), 10)
add_to_cache(chromatic_index, graphs.MeredithGraph(), 5) #number from http://en.wikipedia.org/wiki/Meredith_graph
add_to_cache(clique_covering_number, graphs.SchlaefliGraph(), 6)
add_to_cache(Graph.chromatic_number, graphs.SchlaefliGraph(), 9)  #number from http://en.wikipedia.org/wiki/Schl%C3%A4fli_graph

#GRAPH PROPERTIES




#sufficient condition for hamiltonicity
def is_dirac(g):
    n = g.order()
    deg = g.degree()
    delta=min(deg)
    if n/2 <= delta:
        return True
    else:
        return False

#sufficient condition for hamiltonicity
def is_ore(g):
    A = g.adjacency_matrix()
    D = g.degree()
    n = g.order()
    for i in range(n):
        for j in range(i):
            if A[i][j]==0:
                if D[i] + D[j] < n:
                    return False
    return True

#sufficient condition for hamiltonicity
def is_haggkvist_nicoghossian(g):
    k = g.vertex_connectivity()
    n = g.order()
    delta = min(g.degree())
    if k >= 2 and delta >= (1.0/3)*(n+k):
        return True
    else:
        return False

#sufficient condition for hamiltonicity
def is_fan(g):
    k = g.vertex_connectivity()
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
    k = g.vertex_connectivity()
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
    k = g.vertex_connectivity()
    if k < 2:
        return False
    else:
        return True

#sufficient condition for hamiltonicity
def is_lindquester(g):
    k = g.vertex_connectivity()
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

def has_claw(g):
    return g.subgraph_search(graphs.ClawGraph(), induced=True) is not None

def is_claw_free(g):
    return not has_claw(g)

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

def has_lovasz_theta_equals_alpha(g):
    if lovasz_theta(g) == independence_number(g):
        return True
    else:
        return False

def has_lovasz_theta_equals_cc(g):
    if lovasz_theta(g) == clique_covering_number(g):
        return True
    else:
        return False

#z1 is a triangle with pendant, that show's up in hedetniemi sufficient condition for hamiltonicity
def has_z1(g):
    return g.subgraph_search(z1, induced=True) is not None

def is_z1_free(g):
    return not has_z1(g)

#sufficient condition for hamiltonicity
def is_chvatal_erdos(g):
    return independence_number(g) <= g.vertex_connectivity()

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
    nu = matching_number(g)
    E = g.edges()
    for e in E:
        g.delete_edge(e)
        nu2 = matching_number(g)
        if nu != nu2:
            return False
        g.add_edge(e)
    return True

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
is_locally_two_connected = localise(is_two_connected)

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

efficiently_computable_properties = [Graph.is_regular, Graph.is_planar, Graph.is_forest, Graph.is_eulerian, Graph.is_connected, Graph.is_clique, Graph.is_circular_planar, Graph.is_chordal, Graph.is_bipartite, Graph.is_cartesian_product, Graph.is_distance_regular,  Graph.is_even_hole_free, Graph.is_gallai_tree, Graph.is_line_graph, Graph.is_overfull, Graph.is_perfect, Graph.is_split, Graph.is_strongly_regular, Graph.is_triangle_free, Graph.is_weakly_chordal, is_dirac, is_ore, is_haggkvist_nicoghossian, is_generalized_dirac, is_van_den_heuvel, is_two_connected, is_lindquester, is_claw_free, has_perfect_matching, has_radius_equal_diameter, is_not_forest, has_empty_KE_part, is_fan, is_cubic, diameter_equals_twice_radius, has_z1, is_z1_free, diameter_equals_radius, is_locally_connected, matching_covered, is_locally_dirac, is_locally_bipartite, is_locally_two_connected]

intractable_properties = [Graph.is_hamiltonian, Graph.is_vertex_transitive, Graph.is_edge_transitive, has_residue_equals_alpha, Graph.is_odd_hole_free, Graph.is_semi_symmetric, Graph.is_line_graph, is_planar_transitive, is_class1, is_class2, is_anti_tutte, is_anti_tutte2, has_lovasz_theta_equals_cc, has_lovasz_theta_equals_alpha, is_chvatal_erdos ]

#speed notes
#FAST ENOUGH (tested for graphs on 140921): is_hamiltonian, is_vertex_transitive, is_edge_transitive, has_residue_equals_alpha, is_odd_hole_free, is_semi_symmetric, is_line_graph, is_line_graph, is_anti_tutte, is_planar_transitive
#SLOW but FIXED for SpecialGraphs: is_class1, is_class2

properties = efficiently_computable_properties + intractable_properties

# Graph.is_prime removed as faulty 9/2014
# built in Graph.is_transitively_reduced removed 9/2014
# is_line_graph is theoretically efficient - but Sage's implementation is not 9/2014

# weakly_chordal = weakly chordal, i.e., the graph and its complement have no induced cycle of length at least 5



#GRAPH OBJECTS


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
tent = graphs.CycleGraph(4).join(Graph(1),verbose_relabel=false)
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

#GRAPH LISTS

hamiltonian_objects = [graphs.CompleteGraph(3), graphs.CompleteGraph(4), graphs.CompleteGraph(5), c6ee, c5chord, graphs.DodecahedralGraph(), c8chorded, c8chords, graphs.ClebschGraph(), graphs.CycleGraph(4), prismy, c24, c26, graphs.BuckyBall(), c6xc6, holton_mckay, sixfour]

non_hamiltonian_objects = [graphs.PetersenGraph(), graphs.PathGraph(2), graphs.TutteGraph(), non_ham_over, throwing, throwing2, throwing3, kratsch_lehel_muller ]

residue_equals_alpha_objects = [graphs.StarGraph(3)]

other_graphs = [graphs.BullGraph(), graphs.ChvatalGraph(), graphs.ClawGraph(), graphs.DesarguesGraph(), graphs.DiamondGraph(), graphs.DodecahedralGraph(), graphs.FlowerSnark(), graphs.FruchtGraph(), graphs.HoffmanSingletonGraph(), graphs.HouseGraph(), graphs.HouseXGraph(), graphs.OctahedralGraph(), graphs.ThomsenGraph(), graphs.TetrahedralGraph(), graphs.PetersenGraph(), graphs.PappusGraph(), graphs.GrotzschGraph(), graphs.GrayGraph(), graphs.HeawoodGraph(), graphs.HerschelGraph(), graphs.SchlaefliGraph(), graphs.CoxeterGraph(), graphs.BrinkmannGraph(), graphs.TutteCoxeterGraph(), graphs.TutteGraph(), graphs.RobertsonGraph(), graphs.FolkmanGraph(), graphs.Balaban10Cage(), graphs.BullGraph(), graphs.BuckyBall(), graphs.PappusGraph(), graphs.TietzeGraph(), graphs.SylvesterGraph(), graphs.SzekeresSnarkGraph(), graphs.MoebiusKantorGraph(), ryan, inp, c4c4, regular_non_trans, bridge, z1, p10k4]

#graphs were some computations are especially slow
problem_graphs = [graphs.MeredithGraph()]

#meredith graph is 4-reg, class2, non-hamiltonian: http://en.wikipedia.org/wiki/Meredith_graph

union_objects = hamiltonian_objects + non_hamiltonian_objects + residue_equals_alpha_objects + other_graphs + problem_graphs

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

graph_objects = remove_duplicates(union_objects, idfun=Graph.graph6_string)
