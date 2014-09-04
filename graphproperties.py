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
        for j in range(n):
            if i>j and A[i][j]==0:
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
    Dist = g.distance_matrix()
    L = []
    V = g.vertices()
    n = g.order()
    for i in range(n):
        for j in range (n):
            if i>j and Dist[i][j]==2:
                L.append(max(D[i],D[j]))
    if len(L) == 0:
        return True
    if min(L) >= n/2.0:
        return True
    else:
        return False

#sufficient condition for hamiltonicity
def is_planar_transitive(g):
    if g.order() > 2 and g.is_planar() and g.is_vertex_transitive():
        return True
    else:
        return False

def neighbors_set(g,S):
    N = Set([])
    for v in S:
        T = Set(g.neighbors(v))
        N = N.union(T)
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
def is_van_den_Heuvel(g):
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

        return D + M

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
    D = g.distance_matrix()
    n = g.order()
    V = g.vertices()
    for i in range(n):
        for j in range(n):
            if i > j and D[i][j] == 2:
                if len(neighbors_set(g,[V[i],V[j]])) < (2*n-1)/3.0:
                    return False
    return True

def is_complete(g):
    D = g.distance_matrix()
    n = g.order()
    e = g.size()
    if not g.has_multiple_edges():
        return e == n*(n-1)/2

    for i in range(n):
        for j in range(n):
            if i>j and D[i][j] == 0:
                return False
    return True


properties = [Graph.is_hamiltonian, Graph.is_vertex_transitive, Graph.is_transitively_reduced, Graph.is_regular, Graph.is_planar, Graph.is_forest, Graph.is_eulerian, Graph.is_connected, Graph.is_clique, Graph.is_circular_planar, Graph.is_chordal, Graph.is_bipartite, Graph.is_cartesian_product, Graph.is_distance_regular, Graph.is_edge_transitive, Graph.is_even_hole_free, Graph.is_gallai_tree, Graph.is_line_graph, Graph.is_overfull, Graph.is_perfect, Graph.is_split, Graph.is_strongly_regular, Graph.is_triangle_free, Graph.is_weakly_chordal,Graph.is_odd_hole_free, is_dirac, is_ore, is_haggkvist_nicoghossian, is_fan, is_planar_transitive, is_generalized_dirac, is_van_den_Heuvel, is_two_connected, is_lindquester]

# Graph.is_prime removed as faulty 9/2014

