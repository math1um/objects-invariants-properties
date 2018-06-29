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

def compute_invariant_value(invariant, graph, g_key):
    """
    Computes the value of invariant for graph and stores the result in the
    dictionary computation_results. This method is not intended to be called
    directly. It will be called by the method update_invariant_database as a
    separate process.
    """
    try:
        return float(invariant(graph))
    except Exception as e:
        print "Error while computing {} for {}".format(invariant.__name__, graph.name())
        print type(e), e.message

def update_invariant_database(invariants, graphs, timeout=60, database=None, verbose=False):
    """
    Tries to compute the invariant value of each invariant in invariants for each
    graph in graphs and stores it in the database. If the value is already in the
    database it is not recomputed. If the computation does not end in timeout
    seconds, then it is terminated. The default value for the timeout is 60 (one
    minute). If no database name is provided, this method will default to the
    default database of get_connection().
    """
    # get the values which are already in the database
    current = invariants_as_dict(database)

    # open a connection with the database
    conn = get_connection(database)

    for inv in invariants:
        for g in graphs:
            # first we check to see if the value is already known
            g_key = g.canonical_label(algorithm='sage').graph6_string()
            if g_key in current:
                if inv.__name__ in current[g_key]:
                    continue

            result = None
            try:
                alarm(timeout)
                result = compute_invariant_value(inv, g, g_key)
            except (AlarmInterrupt, KeyboardInterrupt):
                # Computation did not end. We interrupt/kill it.
                print "Computation of {} for {} did not end in time... killing!".format(inv.__name__, g.name())
            else:
                #computation did end, so we add the value to the database
                cancel_alarm()
                if result != None:
                    conn.execute("INSERT INTO inv_values(invariant, graph, value) VALUES (?,?,?)",(inv.__name__, g_key, result))
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

def verify_invariant_values(invariants, graphs, epsilon= 0.00000001, timeout=60, database=None):
    """
    Tries to compute the invariant value of each invariant in invariants for each
    graph in graphs and compares it to the value stored in the database. If the
    computation does not end in timeout seconds, then it is terminated. The
    default value for the timeout is 60 (one minute). If no database name is
    provided, this method will default to the default database of get_connection().
    """
    # get the values which are already in the database
    current = invariants_as_dict(database)

    # create a manager to get the results from the worker thread to the main thread
    for inv in invariants:
        for g in graphs:
            # first we check to see if the value is already known
            g_key = g.canonical_label(algorithm='sage').graph6_string()
            if g_key in current:
                if inv.__name__ in current[g_key]:
                    result = None
                    try:
                        alarm(timeout)
                        result = compute_invariant_value(inv, g, g_key)
                    except AlarmInterrupt:
                        # Computation did not end. We interrupt/kill it.
                        print "Computation of {} for {} did not end in time... killing!".format(inv.__name__, g.name())
                    else:
                        #computation did end, so we verify the value
                        cancel_alarm()
                        if result:
                            if result != current[g_key][inv.__name__] and abs(value - current[g_key][inv.__name__]) > epsilon:
                                print "Stored value of {} for {} differs from computed value: {} vs. {}".format(
                                           inv.__name__, g.name(),
                                           current[g_key][inv.__name__],
                                           result)
                        else:
                            # the computation might have crashed
                            print "Computation of {} for {} failed!".format(inv.__name__, g.name())

def precomputed_graphs_by_invariants(graphs, invariants, database = None):
    """
    Returns a copy of graphs filtered to only graphs with values precomputed for all invariants in invariants

    If no database name is provided, this method will default to the default database of get_connection().
    """
    # get the values which are already in the database
    precomputed = invariants_as_dict(database)

    fully_precomputed_graphs = []
    for g in graphs:
        g_key = g.canonical_label(algorithm='sage').graph6_string()
        if all(g_key in precomputed and inv.__name__ in precomputed[g_key] for inv in inv):
            fully_precomputed_graphs.append(g)
    return fully_precomputed_graphs

def compute_property_value(property, graph, g_key):
    """
    Computes the value of property for graph and returns it, if succesful.
    This method is not intended to be called directly. It will be called by the
    method update_property_database as a separate process.
    """
    try:
        value = bool(property(graph))
        return value
    except Exception as e:
        print "Error while computing {} for {}".format(property.__name__, graph.name())
        print type(e), e.message

def update_property_database(properties, graphs, timeout=60, database=None, verbose=False):
    """
    Tries to compute the property value of each property in properties for each
    graph in graphs and stores it in the database. If the value is already in the
    database it is not recomputed. If the computation does not end in timeout
    seconds, then it is terminated. The default value for the timeout is 60 (one
    minute). If no database name is provided, this method will default to the
    default database of get_connection().
    """
    # get the values which are already in the database
    current = properties_as_dict(database)

    # open a connection with the database
    conn = get_connection(database)

    for prop in properties:
        for g in graphs:
            # first we check to see if the value is already known
            g_key = g.canonical_label(algorithm='sage').graph6_string()
            if g_key in current:
                if prop.__name__ in current[g_key]:
                    continue

            result = None
            try:
                alarm(timeout)
                result = compute_property_value(prop, g, g_key)
            except (AlarmInterrupt, KeyboardInterrupt):
                # Computation did not end. We interrupt/kill it.
                print "Computation of {} for {} did not end in time... killing!".format(prop.__name__, g.name())
            else:
                # computation did end, so we add the value to the database
                cancel_alarm()
                if result != None:
                    conn.execute("INSERT INTO prop_values(property, graph, value) VALUES (?,?,?)",(prop.__name__, g_key, result))
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
    # get the values which are already in the database
    current = properties_as_dict(database)

    for prop in properties:
        for g in graphs:
            # first we check to see if the value is already known
            g_key = g.canonical_label(algorithm='sage').graph6_string()
            if g_key in current:
                if prop.__name__ in current[g_key]:
                    result = None
                    try:
                        alarm(timeout)
                        result = compute_property_value(prop, g, g_key)
                    except AlarmInterrupt:
                        # Computation did not end. We interrupt/kill it.
                        print "Computation of {} for {} did not end in time... killing!".format(prop.__name__, g.name())
                    else:
                        #computation did end, so we verify the value
                        cancel_alarm()
                        if result:
                            if result != current[g_key][prop.__name__]:
                                print "Stored value of {} for {} differs from computed value: {} vs. {}".format(
                                           prop.__name__, g.name(),
                                           current[g_key][prop.__name__],
                                           result)
                        else:
                            # the computation might have crashed
                            print "Computation of {} for {} failed!".format(prop.__name__, g.name())

def precomputed_graphs_by_properties(graphs, properties, database = None):
    """
    Returns a copy of graphs filtered to only graphs with values precomputed for all properties in properties

    If no database name is provided, this method will default to the default database of get_connection().
    """
    # get the values which are already in the database
    precomputed = properties_as_dict(database)

    fully_precomputed_graphs = []
    for g in graphs:
        g_key = g.canonical_label(algorithm='sage').graph6_string()
        if all(g_key in precomputed and prop.__name__ in precomputed[g_key] for prop in properties):
            fully_precomputed_graphs.append(g)
    return fully_precomputed_graphs

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

### Conjecturing ###
def precomputed_only_property_conjecture(objects, properties, mainProperty, precomputed_db = None,
                                         time = 5, debug = False, verbose = False, sufficient = True, operators = None, theory = None):
    """
    Runs the conjecturing program for given properties using only the objects with precomputed values for all properties

    Requires the package conjecturing and the file gt_precomputed_database.sage to be loaded.

    If no database name is provided, precomputed_db will default to the default database of get_connection().
    See documentation for propertyBasedConjecture in conjecturing for details on other parameters.
    If verbose = True (default False), we also print when filtering is complete and control is passed to conjecturing.

    The slowest part of this function is finding the graphs' canonical labels when generating the filtered list. If planning to make multiple
    runs of conjecturing, it would be best to use precomputed_graphs_by_properties() in gt_precomputed_database.sage to generate your own
    list of precomputed graphs and then use that with the standard propertyBasedConjecture() method.
    """
    import time as clock
    if verbose: print clock.asctime(clock.localtime(clock.time())) + " Filtering start."
    fully_precomputed_graphs = precomputed_graphs_by_properties(objects, properties, precomputed_db)
    if verbose:
        print clock.asctime(clock.localtime(clock.time())) + " Filtering finished. " + str(len(fully_precomputed_graphs)) + " graphs remaining for conjecture."
        print clock.asctime(clock.localtime(clock.time())) + " Conjecturing start."
    conjectures = propertyBasedConjecture(fully_precomputed_graphs, properties, mainProperty, time = time, debug = debug, verbose = verbose,
                                          sufficient = sufficient, operators = operators, theory = theory,
                                          precomputed = precomputed_properties_for_conjecture(precomputed_db) )
    if verbose: print clock.asctime(clock.localtime(clock.time())) + " Conjecturing finished."
    return conjectures
    
def precomputed_only_invariant_conjecture(objects, invariants, mainInvariant, precomputed_db = None,
                                         variableName = 'x', time = 5, debug = False, verbose = False, upperBound = True, 
                                         operators = None, theory = None):
    """
    Runs the conjecturing program for given invariants using only the objects with precomputed values for all invariants

    Requires the package conjecturing and the file gt_precomputed_database.sage to be loaded.

    If no database name is provided, precomputed_db will default to the default database of get_connection().
    See documentation for conjecture in conjecturing for details on other parameters.
    If verbose = True (default False), we also print when filtering is complete and control is passed to conjecturing.

    The slowest part of this function is finding the graphs' canonical labels when generating the filtered list. If planning to make multiple
    runs of conjecturing, it would be best to use precomputed_graphs_by_properties() in gt_precomputed_database.sage to generate your own
    list of precomputed graphs and then use that with the standard conjecture() method.
    """
    import time as clock
    if verbose: print clock.asctime(clock.localtime(clock.time())) + " Filtering start."
    fully_precomputed_graphs = precomputed_graphs_by_invariants(objects, invariants, precomputed_db)
    if verbose:
        print clock.asctime(clock.localtime(clock.time())) + " Filtering finished. " + str(len(fully_precomputed_graphs)) + " graphs remaining for conjecture."
        print clock.asctime(clock.localtime(clock.time())) + " Conjecturing start."
    conjectures = conjecture(fully_precomputed_graphs, invariants, mainInvariant, variableName = variableName, time = time, debug = debug, 
                                          verbose = verbose, upperBound = upperBound, operators = operators, theory = theory,
                                          precomputed = precomputed_invariants_for_conjecture(precomputed_db) )
    if verbose: print clock.asctime(clock.localtime(clock.time())) + " Conjecturing finished."
    return conjectures
