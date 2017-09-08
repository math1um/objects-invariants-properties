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
    """
    Returns a connection to the database. If no name is provided, this method
    will by default open a connection with a database called gt_precomputed_database.db
    located in the current working directory.
    """
    if database is None:
        database = "gt_precomputed_database.db"
    return sqlite3.connect(database)

def create_tables(database=None):
    """
    Sets up the database for use by the other methods, i.e., this method creates
    the necessary tables in the database. It is safe to run this method even if
    the tables already exist. If no database name is provided, this method will
    default to the default database of get_connection().
    """
    conn = get_connection(database)
    conn.execute("CREATE TABLE IF NOT EXISTS inv_values (invariant TEXT, graph TEXT, value FLOAT, UNIQUE(invariant, graph))")
    conn.execute("CREATE TABLE IF NOT EXISTS prop_values (property TEXT, graph TEXT, value BOOLEAN, UNIQUE(property, graph))")
    conn.close()

def invariants_as_dict(database=None):
    """
    Returns a dictionary containing for each graph a dictionary containing for
    each invariant the invariant value for that invariant and graph. If no
    database name is provided, this method will default to the default database
    of get_connection().
    """
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
    """
    Returns a tuple of length 3 that can be used for the conjecture method of
    conjecturing.py. If no database name is provided, this method will default
    to the default database of get_connection().
    """
    return (invariants_as_dict(database), (lambda g: g.canonical_label(algorithm='sage').graph6_string()), (lambda f: f.__name__))

def properties_as_dict(database=None):
    """
    Returns a dictionary containing for each graph a dictionary containing for
    each property the property value for that property and graph. If no
    database name is provided, this method will default to the default database
    of get_connection().
    """
    d = {}
    conn = get_connection(database)
    result = conn.execute("SELECT property,graph,value FROM prop_values")
    for (p,g,v) in result:
        if g in d:
            d[g][p] = bool(v)
        else:
            d[g] = {p:bool(v)}
    conn.close()
    return d

def precomputed_properties_for_conjecture(database=None):
    """
    Returns a tuple of length 3 that can be used for the propertyBasedConjecture
    method of conjecturing.py. If no database name is provided, this method will
    default to the default database of get_connection().
    """
    return (properties_as_dict(database), (lambda g: g.canonical_label(algorithm='sage').graph6_string()), (lambda f: f.__name__))

def compute_invariant_value(invariant, graph, computation_results):
    """
    Computes the value of invariant for graph and stores the result in the
    dictionary computation_results. This method is not intended to be called
    directly. It will be called by the method update_invariant_database as a
    separate process.
    """
    value = float(invariant(graph))
    computation_results[(invariant.__name__, graph.canonical_label(algorithm='sage').graph6_string())] = value

def update_invariant_database(invariants, graphs, timeout=60, database=None, verbose=False):
    """
    Tries to compute the invariant value of each invariant in invariants for each
    graph in graphs and stores it in the database. If the value is already in the
    database it is not recomputed. If the computation does not end in timeout
    seconds, then it is terminated. The default value for the timeout is 60 (one
    minute). If no database name is provided, this method will default to the
    default database of get_connection().
    """
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
            g_key = g.canonical_label(algorithm='sage').graph6_string()
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
                if (inv.__name__, g.canonical_label(algorithm='sage').graph6_string()) in computation_results:
                    value = computation_results[(inv.__name__, g.canonical_label(algorithm='sage').graph6_string())]
                    conn.execute("INSERT INTO inv_values(invariant, graph, value) VALUES (?,?,?)",(inv.__name__, g.canonical_label(algorithm='sage').graph6_string(), value))
                    # commit the data so we don't lose anything if we abort early
                    conn.commit()
                else:
                    # the computation might have crashed
                    print "Computation of {} for {} failed!".format(inv.__name__, g.name())
        if verbose:
            print "Finished {}".format(inv.__name__)
    # close the connection
    conn.close()

def store_invariant_value(invariant, graph, value, overwrite=False, database=None, epsilon= 0.00000001, verbose=False):
    """
    Stores the given value in the database for the given invariant and graph.
    This method can be used to store hard to compute invariant values which are
    already known. If overwrite is False, then no value in the database will be
    overwritten and a warning will be printed if the provided value differs from
    the value which is currently in the database. If no database name is provided,
    this method will default to the default database of get_connection().
    """
    i_key = invariant.__name__
    g_key = graph.canonical_label(algorithm='sage').graph6_string()
    conn = get_connection(database)
    result = conn.execute("SELECT value FROM inv_values WHERE invariant=? AND graph=?",
                        (i_key, g_key)).fetchone()
    conn.close()

    if not overwrite:
        if result is not None:
            stored_value = result[0]
            if value!=stored_value and abs(value - stored_value) > epsilon:
                print "Stored value of {} for {} differs from provided value: {} vs. {}".format(i_key, graph.name(), stored_value, value)
            elif verbose:
                print "Value of {} for {} is already in the database".format(i_key, graph.name())
            return

    conn = get_connection(database)
    if result is None:
        conn.execute("INSERT INTO inv_values(invariant, graph, value) VALUES (?,?,?)",(i_key, g_key, float(value)))
    else:
        conn.execute("UPDATE inv_values SET value=(?) WHERE invariant = (?) AND graph = (?)",(float(value), i_key, g_key))
    conn.commit()
    conn.close()
    if verbose:
        print "Inserted value of {} for {}: {}".format(i_key, graph.name(), float(value))

def list_missing_invariants(invariants, graphs, database=None):
    """
    Prints a list of all invariant and graph pairs from invariants and graphs
    which are not in the database. If no database name is provided, this method
    will default to the default database of get_connection().
    """
    # get the values which are already in the database
    current = invariants_as_dict(database)

    for inv in invariants:
        for g in graphs:
            g_key = g.canonical_label(algorithm='sage').graph6_string()
            if g_key in current:
                if inv.__name__ in current[g_key]:
                    continue
            print "{} for {} is missing.".format(inv.__name__, g.name())

def verify_invariant_values(invariants, graphs, epsilon= 0.00000001 ,timeout=60, database=None):
    """
    Tries to compute the invariant value of each invariant in invariants for each
    graph in graphs and compares it to the value stored in the database. If the
    computation does not end in timeout seconds, then it is terminated. The
    default value for the timeout is 60 (one minute). If no database name is
    provided, this method will default to the default database of get_connection().
    """
    import multiprocessing

    # get the values which are already in the database
    current = invariants_as_dict(database)

    # create a manager to get the results from the worker thread to the main thread
    manager = multiprocessing.Manager()
    computation_results = manager.dict()

    for inv in invariants:
        for g in graphs:
            # first we check to see if the value is already known
            g_key = g.canonical_label(algorithm='sage').graph6_string()
            if g_key in current:
                if inv.__name__ in current[g_key]:
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
                         #computation did end, so we verify the value
                         if (inv.__name__, g.canonical_label(algorithm='sage').graph6_string()) in computation_results:
                             value = computation_results[(inv.__name__, g.canonical_label(algorithm='sage').graph6_string())]
                             if value != current[g.canonical_label(algorithm='sage').graph6_string()][inv.__name__] and abs(value - current[g.canonical_label(algorithm='sage').graph6_string()][inv.__name__]) > epsilon:
                                 print "Stored value of {} for {} differs from computed value: {} vs. {}".format(
                                            inv.__name__, g.name(),
                                            current[g.canonical_label(algorithm='sage').graph6_string()][inv.__name__],
                                            value)
                         else:
                             # the computation might have crashed
                             print "Computation of {} for {} failed!".format(inv.__name__, g.name())

def compute_property_value(property, graph, computation_results):
    """
    Computes the value of property for graph and stores the result in the
    dictionary computation_results. This method is not intended to be called
    directly. It will be called by the method update_property_database as a
    separate process.
    """
    value = bool(property(graph))
    computation_results[(property.__name__, graph.canonical_label(algorithm='sage').graph6_string())] = value

def update_property_database(properties, graphs, timeout=60, database=None, verbose=False):
    """
    Tries to compute the property value of each property in properties for each
    graph in graphs and stores it in the database. If the value is already in the
    database it is not recomputed. If the computation does not end in timeout
    seconds, then it is terminated. The default value for the timeout is 60 (one
    minute). If no database name is provided, this method will default to the
    default database of get_connection().
    """
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
            g_key = g.canonical_label(algorithm='sage').graph6_string()
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
                if (prop.__name__, g.canonical_label(algorithm='sage').graph6_string()) in computation_results:
                    value = computation_results[(prop.__name__, g.canonical_label(algorithm='sage').graph6_string())]
                    conn.execute("INSERT INTO prop_values(property, graph, value) VALUES (?,?,?)",(prop.__name__, g.canonical_label(algorithm='sage').graph6_string(), value))
                    # commit the data so we don't lose anything if we abort early
                    conn.commit()
                else:
                    # the computation might have crashed
                    print "Computation of {} for {} failed!".format(prop.__name__, g.name())
        if verbose:
            print "Finished {}".format(prop.__name__)
    # close the connection
    conn.close()

def store_property_value(property, graph, value, overwrite=False, database=None, verbose=False):
    """
    Stores the given value in the database for the given property and graph.
    This method can be used to store hard to compute property values which are
    already known. If overwrite is False, then no value in the database will be
    overwritten and a warning will be printed if the provided value differs from
    the value which is currently in the database. If no database name is provided,
    this method will default to the default database of get_connection().
    """
    p_key = property.__name__
    g_key = graph.canonical_label(algorithm='sage').graph6_string()
    conn = get_connection(database)
    result = conn.execute("SELECT value FROM prop_values WHERE property=? AND graph=?",
                        (p_key, g_key)).fetchone()
    conn.close()

    if not overwrite:
        if result is not None:
            stored_value = result[0]
            if value!=stored_value:
                print "Stored value of {} for {} differs from provided value: {} vs. {}".format(p_key, graph.name(), stored_value, value)
            elif verbose:
                print "Value of {} for {} is already in the database".format(p_key, graph.name())
            return

    conn = get_connection(database)
    if result is None:
        conn.execute("INSERT INTO prop_values(property, graph, value) VALUES (?,?,?)",(p_key, g_key, bool(value)))
    else:
        conn.execute("UPDATE prop_values SET value=(?) WHERE property = (?) AND graph = (?)",(bool(value), p_key, g_key))
    conn.commit()
    conn.close()
    if verbose:
        print "Inserted value of {} for {}: {}".format(p_key, graph.name(), bool(value))

def list_missing_properties(properties, graphs, database=None):
    """
    Prints a list of all property and graph pairs from properties and graphs
    which are not in the database. If no database name is provided, this method
    will default to the default database of get_connection().
    """
    # get the values which are already in the database
    current = properties_as_dict(database)

    for prop in properties:
        for g in graphs:
            g_key = g.canonical_label(algorithm='sage').graph6_string()
            if g_key in current:
                if prop.__name__ in current[g_key]:
                    continue
            print "{} for {} is missing.".format(prop.__name__, g.name())

def verify_property_values(properties, graphs, timeout=60, database=None):
    """
    Tries to compute the property value of each property in properties for each
    graph in graphs and compares it to the value stored in the database. If the
    computation does not end in timeout seconds, then it is terminated. The
    default value for the timeout is 60 (one minute). If no database name is
    provided, this method will default to the default database of get_connection().
    """
    import multiprocessing

    # get the values which are already in the database
    current = properties_as_dict(database)

    # create a manager to get the results from the worker thread to the main thread
    manager = multiprocessing.Manager()
    computation_results = manager.dict()

    for prop in properties:
        for g in graphs:
            # first we check to see if the value is already known
            g_key = g.canonical_label(algorithm='sage').graph6_string()
            if g_key in current:
                if prop.__name__ in current[g_key]:
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
                         #computation did end, so we verify the value
                         if (prop.__name__, g.canonical_label(algorithm='sage').graph6_string()) in computation_results:
                             value = computation_results[(prop.__name__, g.canonical_label(algorithm='sage').graph6_string())]
                             if value != current[g.canonical_label(algorithm='sage').graph6_string()][prop.__name__]:
                                 print "Stored value of {} for {} differs from computed value: {} vs. {}".format(
                                            prop.__name__, g.name(),
                                            current[g.canonical_label(algorithm='sage').graph6_string()][prop.__name__],
                                            value)
                         else:
                             # the computation might have crashed
                             print "Computation of {} for {} failed!".format(prop.__name__, g.name())

def dump_database(folder="db", database=None):
    """
    Writes the specified database to a series of SQL files in the specified folder.
    If no folder is given, then this method defaults to a folder named db. If no
    database name is provided, this method will default to the default database
    of get_connection().
    """
    import os

    #make sure all necessary folders exist
    if not os.path.exists(os.path.join(folder, 'invariants')):
        os.makedirs(os.path.join(folder, 'invariants'))
    if not os.path.exists(os.path.join(folder, 'properties')):
        os.makedirs(os.path.join(folder, 'properties'))

    conn = get_connection(database)
    #dump the table with invariant values
    current = invariants_as_dict(database)
    invs = set()

    for g in current.keys():
        invs.update(current[g].keys())

    for inv in sorted(invs):
        f = open(os.path.join(folder, 'invariants', inv + '.sql'), 'w')
        q = "SELECT 'INSERT INTO \"inv_values\" VALUES('||quote(invariant)||','||quote(graph)||','||quote(value)||')' FROM 'inv_values' WHERE invariant=? ORDER BY graph ASC"
        query_res = conn.execute(q, (inv,))
        for row in query_res:
            s = row[0]
            #fix issue with sqlite3 not being able to read its own output
            if s[-5:] == ',Inf)':
                s = s[:-5] + ',1e999)'
            elif s[-6:] == ',-Inf)':
                s = s[:-6] + ',-1e999)'
            f.write("{};\n".format(s))
        f.close()

    #dump the table with property values
    current = properties_as_dict(database)
    props = set()

    for g in current.keys():
        props.update(current[g].keys())

    for prop in sorted(props):
        f = open(os.path.join(folder, 'properties', prop + '.sql'), 'w')
        q = "SELECT 'INSERT INTO \"prop_values\" VALUES('||quote(property)||','||quote(graph)||','||quote(value)||')' FROM 'prop_values' WHERE property=? ORDER BY graph ASC"
        query_res = conn.execute(q, (prop,))
        for row in query_res:
            f.write("{};\n".format(row[0]))
        f.close()
