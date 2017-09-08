def read_dimacs_edge_file(filename):
    """
    Returns a Sage graph object generated from the given uncompressed DIMACS file
    """
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
