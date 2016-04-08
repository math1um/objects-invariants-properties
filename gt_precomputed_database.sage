"""
This file implements the creation and usage of a database of precomputed values
for the graphs, invariants and properties defined in gt.sage.

EXAMPLE::

    sage: load("gt.sage")
    sage: load("gt_precomputed_database.sage")
    sage: update_invariant_database(invariants, graph_objects, timeout=5)
    sage: update_property_database(properties, graph_objects, timeout=5)
"""

import sqlite3

def get_connection(database=None):
    if database is None:
        database = "gt_precomputed_database.db"
    return sqlite3.connect(database)

def create_tables(database=None):
    conn = get_connection(database)
    conn.execute("CREATE TABLE IF NOT EXISTS inv_values (invariant TEXT, graph TEXT, value FLOAT)")
    conn.execute("CREATE TABLE IF NOT EXISTS prop_values (property TEXT, graph TEXT, value BOOLEAN)")

def invariants_as_dict(database=None):
    d = {}
    conn = get_connection(database)
    result = conn.execute("SELECT invariant,graph,value FROM inv_values")
    for (i,g,v) in result:
        if g in d:
            d[g][i] = v
        else:
            d[g] = {i:v}
    conn.close()
    return d

def precomputed_invariants_for_conjecture(database=None):
    return (invariants_as_dict(database), (lambda g: g.canonical_label().graph6_string()), (lambda f: f.__name__))

def properties_as_dict(database=None):
    d = {}
    conn = get_connection(database)
    result = conn.execute("SELECT property,graph,value FROM prop_values")
    for (p,g,v) in result:
        if g in d:
            d[g][p] = v
        else:
            d[g] = {p:v}
    conn.close()
    return d

def precomputed_properties_for_conjecture(database=None):
    return (properties_as_dict(database), (lambda g: g.canonical_label().graph6_string()), (lambda f: f.__name__))

def compute_invariant_value(invariant, graph, computation_results):
    value = float(invariant(graph))
    computation_results[(invariant.__name__, graph.canonical_label().graph6_string())] = value

def update_invariant_database(invariants, graphs, timeout=60, database=None):
    import multiprocessing

    # get the values which are already in the database
    current = invariants_as_dict(database)

    # open a connection with the database
    conn = get_connection(database)

    # create a manager to get the results from the worker thread to the main thread
    manager = multiprocessing.Manager()
    computation_results = manager.dict()

    for inv in invariants:
        for g in graphs:
            # first we check to see if the value is already known
            g_key = g.canonical_label().graph6_string()
            if g_key in current:
                if inv.__name__ in current[g_key]:
                    continue

            # start a worker thread to compute the value
            p = multiprocessing.Process(target=compute_invariant_value, args=(inv, g, computation_results))
            p.start()

            # give the worker thread some time to calculate the value
            p.join(timeout)

            # if the worker thread is not finished we kill it. Otherwise we store the value in the database
            if p.is_alive():
                print "Computation of {} for {} did not end in time... killing!".format(inv.__name__, g.name())

                p.terminate()
                p.join()
            else:
                #computation did end, so we add the value to the database
                if (inv.__name__, g.canonical_label().graph6_string()) in computation_results:
                    value = computation_results[(inv.__name__, g.canonical_label().graph6_string())]
                    conn.execute("INSERT INTO inv_values(invariant, graph, value) VALUES (?,?,?)",(inv.__name__, g.canonical_label().graph6_string(), value))
                    # commit the data so we don't lose anything if we abort early
                    conn.commit()
                else:
                    # the computation might have crashed
                    print "Computation of {} for {} failed!".format(inv.__name__, g.name())
    # close the connection
    conn.close()

def store_invariant_value(invariant, graph, value, overwrite=False, database=None):
    current = invariants_as_dict(database)
    i_key = invariant.__name__
    g_key = graph.canonical_label().graph6_string()

    if not overwrite:
        if g_key in current:
            if i_key in current[g_key]:
                if value!=current[g_key][i_key]:
                    print "Stored value differs from provided value: {} vs. {}".format(current[g_key][i_key], value)
                return

    conn = get_connection(database)
    conn.execute("INSERT INTO inv_values(invariant, graph, value) VALUES (?,?,?)",(i_key, g_key, float(value)))
    conn.commit()
    conn.close()

def list_missing_invariants(invariants, graphs, database=None):
    # get the values which are already in the database
    current = invariants_as_dict(database)

    for inv in invariants:
        for g in graphs:
            g_key = g.canonical_label().graph6_string()
            if g_key in current:
                if inv.__name__ in current[g_key]:
                    continue
            print "{} for {} is missing.".format(inv.__name__, g.name())

def compute_property_value(property, graph, computation_results):
    value = bool(property(graph))
    computation_results[(property.__name__, graph.canonical_label().graph6_string())] = value

def update_property_database(properties, graphs, timeout=60, database=None):
    import multiprocessing

    # get the values which are already in the database
    current = properties_as_dict(database)

    # open a connection with the database
    conn = get_connection(database)

    # create a manager to get the results from the worker thread to the main thread
    manager = multiprocessing.Manager()
    computation_results = manager.dict()

    for prop in properties:
        for g in graphs:
            # first we check to see if the value is already known
            g_key = g.canonical_label().graph6_string()
            if g_key in current:
                if prop.__name__ in current[g_key]:
                    continue

            # start a worker thread to compute the value
            p = multiprocessing.Process(target=compute_property_value, args=(prop, g, computation_results))
            p.start()

            # give the worker thread some time to calculate the value
            p.join(timeout)

            # if the worker thread is not finished we kill it. Otherwise we store the value in the database
            if p.is_alive():
                print "Computation of {} for {} did not end in time... killing!".format(prop.__name__, g.name())

                p.terminate()
                p.join()
            else:
                #computation did end, so we add the value to the database
                if (prop.__name__, g.canonical_label().graph6_string()) in computation_results:
                    value = computation_results[(prop.__name__, g.canonical_label().graph6_string())]
                    conn.execute("INSERT INTO prop_values(property, graph, value) VALUES (?,?,?)",(prop.__name__, g.canonical_label().graph6_string(), value))
                    # commit the data so we don't lose anything if we abort early
                    conn.commit()
                else:
                    # the computation might have crashed
                    print "Computation of {} for {} failed!".format(prop.__name__, g.name())
    # close the connection
    conn.close()

def store_property_value(property, graph, value, overwrite=False, database=None):
    current = properties_as_dict(database)
    p_key = property.__name__
    g_key = graph.canonical_label().graph6_string()

    if not overwrite:
        if g_key in current:
            if p_key in current[g_key]:
                if value!=current[g_key][p_key]:
                    print "Stored value differs from provided value: {} vs. {}".format(current[g_key][p_key], value)
                return

    conn = get_connection(database)
    conn.execute("INSERT INTO prop_values(property, graph, value) VALUES (?,?,?)",(p_key, g_key, bool(value)))
    conn.commit()
    conn.close()

def list_missing_properties(properties, graphs, database=None):
    # get the values which are already in the database
    current = properties_as_dict(database)

    for prop in properties:
        for g in graphs:
            g_key = g.canonical_label().graph6_string()
            if g_key in current:
                if prop.__name__ in current[g_key]:
                    continue
            print "{} for {} is missing.".format(prop.__name__, g.name())
