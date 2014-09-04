def dominationNumber(g):
    return len(g.dominating_set())

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
    x = p.new_variable()
    p.set_objective(sum(x[v] for v in g.vertices()))

    for v in g.vertices():
        p.add_constraint(x[v], max=1)

    for (u,v) in g.edge_iterator(labels=False):
        p.add_constraint(x[u] + x[v], max=1)

    return p.solve()

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

invariants = [dominationNumber, Graph.average_distance, Graph.diameter, Graph.radius, Graph.girth,Graph.order, Graph.size, Graph.szeged_index, Graph.wiener_index, min_degree, max_degree, Graph.average_degree, matching_number, residue, annihilation_number, fractional_alpha, lovasz_theta, cvetkovic]
