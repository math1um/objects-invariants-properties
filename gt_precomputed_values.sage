"""
This file stores some hard to compute values in the database of precomputed values
for the graphs, invariants and properties defined in gt.sage.

EXAMPLE::

    sage: load("gt.sage")
    sage: load("gt_precomputed_database.sage")
    sage: load("gt_precomputed_values.sage")
    sage: store_values()
"""

def store_values(overwrite=False, database=None, verbose=False):
    """
    Stores some hard to compute values in the specified database. If no database
    name is provided, this defaults to the default database of get_connection().
    By default this method does not overwrite existing values and prints a warning
    if the existing value differs from the value that is provided here.
    """
    # Taken from http://en.wikipedia.org/wiki/Meredith_graph
    store_invariant_value(chromatic_index, graphs.MeredithGraph(), 5, overwrite=overwrite, database=database, verbose=verbose)

    #taken from http://en.wikipedia.org/wiki/Schl%C3%A4fli_graph
    store_invariant_value(clique_covering_number, graphs.SchlaefliGraph(), 6, overwrite=overwrite, database=database, verbose=verbose)
    store_invariant_value(chromatic_num, graphs.SchlaefliGraph(), 9, overwrite=overwrite, database=database, verbose=verbose)

    store_invariant_value(chromatic_num, c3mycielski4, 7, overwrite=overwrite, database=database, verbose=verbose)
    store_invariant_value(chromatic_index, c3mycielski4, 32, overwrite=overwrite, database=database, verbose=verbose)
    store_invariant_value(clique_covering_number, c3mycielski4, 31, overwrite=overwrite, database=database, verbose=verbose)

    store_invariant_value(chromatic_num, alon_seymour, 56, overwrite=overwrite, database=database, verbose=verbose)
    store_invariant_value(chromatic_index, alon_seymour, 56, overwrite=overwrite, database=database, verbose=verbose)
    store_invariant_value(edge_con, alon_seymour, 56, overwrite=overwrite, database=database, verbose=verbose)
    store_invariant_value(vertex_con, alon_seymour, 56, overwrite=overwrite, database=database, verbose=verbose)
    store_invariant_value(kirchhoff_index, alon_seymour, 71.0153846154, overwrite=overwrite, database=database, verbose=verbose)
    store_property_value(matching_covered, alon_seymour, True, overwrite=overwrite, database=database, verbose=verbose)
    store_property_value(is_locally_two_connected, alon_seymour, True, overwrite=overwrite, database=database, verbose=verbose)

    # Miscomputed by Sage as 9.999
    store_invariant_value(Graph.lovasz_theta, starfish, 10, overwrite=overwrite, database=database, verbose=verbose)

    # Miscomputed by Sage as 6.9999
    store_invariant_value(Graph.lovasz_theta, sylvester, 7, overwrite=overwrite, database=database, verbose=verbose)

    # c100 is hamiltonian. Cite: Aldred, Robert EL, et al. "Nonhamiltonian 3-connected cubic planar graphs." SIAM Journal on Discrete Mathematics 13.1 (2000): 25-32.
    store_property_value(Graph.is_hamiltonian, c100, True, overwrite=overwrite, database=database, verbose=verbose)
    store_invariant_value(Graph.lovasz_theta, c100, 46.694, overwrite=overwrite, database=database, verbose=verbose)

    # Calculated haemers *is* 3-connected using new property definition on 160418
    store_property_value(is_three_connected, haemers, True, overwrite=overwrite, database=database, verbose=verbose)
