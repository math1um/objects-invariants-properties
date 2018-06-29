# THEORY
all_invariant_theorems = []
all_property_theorems = []

alpha_upper_bounds = []
alpha_lower_bounds = []

hamiltonian_sufficient = []

#####
# ALPHA UPPER BOUNDS
#####

# R. Pepper. Binding independence. Ph. D. Dissertation. University of Houston. Houston, TX, 2004.
alpha_annihilation_bound = annihilation_number
add_to_lists(alpha_annihilation_bound, alpha_upper_bounds, all_invariant_theorems)

# Nemhauser, George L., and Leslie Earl Trotter. "Vertex packings: structural properties and algorithms." Mathematical Programming 8.1 (1975): 232-248.
# Nemhauser, George L., and Leslie E. Trotter. "Properties of vertex packing and independence system polyhedra." Mathematical Programming 6.1 (1974): 48-61.
alpha_fractional_bound = fractional_alpha
add_to_lists(alpha_fractional_bound, alpha_upper_bounds, all_invariant_theorems)

# D. M. Cvetkovic, M. Doob, and H. Sachs. Spectra of graphs. Academic Press, New York, 1980.
alpha_cvetkovic_bound = cvetkovic
add_to_lists(alpha_cvetkovic_bound, alpha_upper_bounds, all_invariant_theorems)

# Trivial
alpha_trivial_bound = Graph.order
add_to_lists(alpha_trivial_bound, alpha_upper_bounds, all_invariant_theorems)

# Lovasz Theta
alpha_lovasz_theta_bound = Graph.lovasz_theta
add_to_lists(alpha_lovasz_theta_bound, alpha_upper_bounds, all_invariant_theorems)

# R. Pepper. Binding independence. Ph. D. Dissertation. University of Houston. Houston, TX, 2004.
def alpha_kwok_bound(g):
    return order(g) - (g.size()/max_degree(g))
add_to_lists(alpha_kwok_bound, alpha_upper_bounds, all_invariant_theorems)

# P. Hansen and M. Zheng. Sharp Bounds on the order, size, and stability number of graphs. NETWORKS 23 (1993), no. 2, 99-102.
def alpha_hansen_bound(g):
    return floor(1/2 + sqrt(1/4 + order(g)**2 - order(g) - 2*size(g)))
add_to_lists(alpha_hansen_bound, alpha_upper_bounds, all_invariant_theorems)

# Matching Number - Folklore
def alpha_matching_number_bound(g):
    return order(g) - matching_number(g)
add_to_lists(alpha_matching_number_bound, alpha_upper_bounds, all_invariant_theorems)

# Min-Degree Theorm
def alpha_min_degree_bound(g):
    return order(g) - min_degree(g)
add_to_lists(alpha_min_degree_bound, alpha_upper_bounds, all_invariant_theorems)

# Cut Vertices Theorem
# Three Bounds on the Independence Number of a Graph - C. E. Larson, R. Pepper
def alpha_cut_vertices_bound(g):
    return (g.order() - (card_cut_vertices(g)/2) - (1/2))
add_to_lists(alpha_cut_vertices_bound, alpha_upper_bounds, all_invariant_theorems)

# Median Degree Theorem
# Three Bounds on the Independence Number of a Graph - C. E. Larson, R. Pepper
def alpha_median_degree_bound(g):
    return (g.order() - (median_degree(g)/2) - 1/2)
add_to_lists(alpha_median_degree_bound, alpha_upper_bounds, all_invariant_theorems)

# Godsil-Newman Upper Bound theorem
# Godsil, Chris D., and Mike W. Newman. "Eigenvalue bounds for independent sets." Journal of Combinatorial Theory, Series B 98.4 (2008): 721-734.
def alpha_godsil_newman_bound(g):
    L = max(g.laplacian_matrix().change_ring(RDF).eigenvalues())
    return g.order()*(L-min_degree(g))/L
add_to_lists(alpha_godsil_newman_bound, alpha_upper_bounds, all_invariant_theorems)

# AGX Upper Bound Theorem
#Aouchiche, Mustapha, Gunnar Brinkmann, and Pierre Hansen. "Variable neighborhood search for extremal graphs. 21. Conjectures and results about the independence number." Discrete Applied Mathematics 156.13 (2008): 2530-2542.
def alpha_AGX_upper_bound(g):
    return (g.order() + max_degree(g) - ceil(2 * sqrt(g.order() - 1)))
add_to_lists(alpha_AGX_upper_bound, alpha_upper_bounds, all_invariant_theorems)

def alpha_li_zhang_1_bound(g):
    """
    From:
    A note on eigenvalue bounds for independence numbers of non-regular graphs
    By Yusheng Li, Zhen Zhang
    """
    return (((max_degree(g)) - min_degree(g) - min_eigenvalue(g)) / max_degree(g)) * g.order()
add_to_lists(alpha_li_zhang_1_bound, alpha_upper_bounds, all_invariant_theorems)

def alpha_li_zhang_2_bound(g):
    """
    From:
    A note on eigenvalue bounds for independence numbers of non-regular graphs
    By Yusheng Li, Zhen Zhang
    """
    return ((max_eigenvalue(g) - min_eigenvalue(g) + max_degree(g) - 2 * min_degree(g)) / (max_eigenvalue(g) - min_eigenvalue(g) + max_degree(g) - min_degree(g))) * g.order()
add_to_lists(alpha_li_zhang_2_bound, alpha_upper_bounds, all_invariant_theorems)

def alpha_haemers_bound(g):
    """
    From: W. Haemers, Interlacing eigenvalues and graphs, Linear Algebra Appl. 226/228 (1995) 593–616.
    """
    return ((-max_eigenvalue(g) * min_eigenvalue(g)) / (min_degree(g)**2 - (max_eigenvalue(g) * min_eigenvalue(g)))) * g.order()
add_to_lists(alpha_haemers_bound, alpha_upper_bounds, all_invariant_theorems)

#####
# LOWER BOUNDS
#####

# Radius Pendants Theorem
# Three Bounds on the Independence Number of a Graph - C. E. Larson, R. Pepper
def alpha_radius_pendants_bound(g):
    return (g.radius() + (card_pendants(g)/2) - 1)
add_to_lists(alpha_radius_pendants_bound, alpha_lower_bounds, all_invariant_theorems)

# AGX Lower Bound Theorem
# Aouchiche, Mustapha, Gunnar Brinkmann, and Pierre Hansen. "Variable neighborhood search for extremal graphs. 21. Conjectures and results about the independence number." Discrete Applied Mathematics 156.13 (2008): 2530-2542.
def alpha_AGX_lower_bound(g):
    return ceil(2 * sqrt(g.order()))
add_to_lists(alpha_AGX_lower_bound, alpha_lower_bounds, all_invariant_theorems)

def alpha_max_degree_minus_triangles_bound(g):
    return max_degree(g) - g.triangles_count()
add_to_lists(alpha_max_degree_minus_triangles_bound, alpha_lower_bounds, all_invariant_theorems)

def alpha_order_brooks_bound(g):
    return ceil(order(x)/brooks(x))
add_to_lists(alpha_order_brooks_bound, alpha_lower_bounds, all_invariant_theorems)

def alpha_szekeres_wilf_bound(g):
    return ceil(order(x)/szekeres_wilf(x))
add_to_lists(alpha_szekeres_wilf_bound, alpha_lower_bounds, all_invariant_theorems)

def alpha_welsh_powell_bound(g):
    return ceil(g.order()/welsh_powell(g))
add_to_lists(alpha_welsh_powell_bound, alpha_lower_bounds, all_invariant_theorems)

def alpha_staton_girth_bound(g):
    """
    Hopkins, Glenn, and William Staton. "Girth and independence ratio." Can. Math. Bull. 25.2 (1982): 179-186.
    """
    if g.girth() < 6:
        return 1
    else:
        d = max_degree(g)
        return order(g) * (2* d - 1) / (d^2 + 2 * d - 1)
add_to_lists(alpha_staton_girth_bound, alpha_lower_bounds, all_invariant_theorems)

def alpha_staton_triangle_free_bound(g):
    """
    Staton, William. "Some Ramsey-type numbers and the independence ratio." Transactions of the American Mathematical Society 256 (1979): 353-370.
    """
    if g.is_triangle_free() and (max_degree(g) > 2):
        return (5 * g.order() ) / ((5 * max_degree(g)) - 1)
    return 1
add_to_lists(alpha_staton_triangle_free_bound, alpha_lower_bounds, all_invariant_theorems)

alpha_average_distance_bound = Graph.average_distance
add_to_lists(alpha_average_distance_bound, alpha_lower_bounds, all_invariant_theorems)

alpha_radius_bound = Graph.radius
add_to_lists(alpha_radius_bound, alpha_lower_bounds, all_invariant_theorems)

alpha_residue_bound = residue
add_to_lists(alpha_residue_bound, alpha_lower_bounds, all_invariant_theorems)

alpha_max_even_minus_even_horizontal_bound = max_even_minus_even_horizontal
add_to_lists(alpha_max_even_minus_even_horizontal_bound, alpha_lower_bounds, all_invariant_theorems)

alpha_critical_independence_number_bound = critical_independence_number
add_to_lists(alpha_critical_independence_number_bound, alpha_lower_bounds, all_invariant_theorems)

def alpha_max_degree_minus_number_of_triangles_bound(g):
    return max_degree(g) - g.triangles_count()
add_to_lists(alpha_max_degree_minus_number_of_triangles_bound, alpha_lower_bounds, all_invariant_theorems)

def alpha_HHRS_bound(g):
    """
    Returns 1 if max_degree > 3 or if g has k4 as a subgraph
    ONLY WORKS FOR CONNECTED GRAPHS becasue that is what we are focussing on, disconnected graphs just need to count the bad compenents

    Harant, Jochen, et al. "The independence number in graphs of maximum degree three." Discrete Mathematics 308.23 (2008): 5829-5833.
    """
    assert(g.is_connected() == true)
    if not is_subcubic(g):
        return 1
    if has_k4(g):
        return 1
    return (4*g.order() - g.size() - (1 if is_bad(g) else 0) - subcubic_tr(g)) / 7
add_to_lists(alpha_HHRS_bound, alpha_lower_bounds, all_invariant_theorems)

def alpha_seklow_bound(g):
    """
    Returns the Seklow bound from:
    Selkow, Stanley M. "A probabilistic lower bound on the independence number of graphs." Discrete Mathematics 132.1-3 (1994): 363-365.
    """
    v_sum = 0
    for v in g.vertices():
        d = g.degree(v)
        v_sum += ((1/(d + 1)) * (1 + max(0, (d/(d + 1) - sum([(1/(g.degree(w) + 1)) for w in g.neighbors(v)])))))
    return v_sum
add_to_lists(alpha_seklow_bound, alpha_lower_bounds, all_invariant_theorems)

def alpha_harant_bound(g):
    """
    From:
    A lower bound on the independence number of a graph
    Jochen Harant
    """
    return (caro_wei(g)**2) / (caro_wei(g) - sum([(g.degree(e[0]) - g.degree(e[1]))**2 * (1/g.degree(e[0]))**2 * (1/g.degree(e[1]))**2 for e in g.edges()]))
add_to_lists(alpha_harant_bound, alpha_lower_bounds, all_invariant_theorems)

def alpha_harant_schiermeyer_bound(g):
    """
    From:
    On the independence number of a graph in terms of order and size
    By J. Harant and I. Schiermeyerb
    """
    order = g.order()
    t = 2*g.size() + order + 1
    return (t - sqrt(t**2 - 4 * order**2)) / 2
add_to_lists(alpha_harant_schiermeyer_bound, alpha_lower_bounds, all_invariant_theorems)

def alpha_shearer_bound(g):
    """
    From:
    Shearer, James B. "The independence number of dense graphs with large odd girth." Electron. J. Combin 2.2 (1995).
    """
    girth = g.girth()

    if girth == +Infinity:
        return 1.0
    if is_even(girth):
        return 1.0

    k = ((girth - 1) / 2.0)
    v_sum = sum([g.degree(v)**(1/(k - 1.0)) for v in g.vertices()])
    return 2**(-((k - 1.0)/k)) * v_sum**((k - 1.0)/k)
add_to_lists(alpha_shearer_bound, alpha_lower_bounds, all_invariant_theorems)


####
# HAMILTONICITY SUFFICIENT CONDITIONS
####
# Chvátal, V., & Erdös, P. (1972). A note on hamiltonian circuits. Discrete Mathematics, 2(2), 111-113.
add_to_lists(is_chvatal_erdos, hamiltonian_sufficient, all_property_theorems)

# R.J Faudree, Ronald J Gould, Michael S Jacobson, R.H Schelp, Neighborhood unions and hamiltonian properties in graphs, Journal of Combinatorial Theory, Series B, Volume 47, Issue 1, 1989, Pages 1-9
add_to_lists(is_generalized_dirac, hamiltonian_sufficient, all_property_theorems)

# Häggkvist, Roland & Nicoghossian, G. G. (1981). A remark on hamiltonian cycles. Journal of Combinatorial Theory, 30(1), 118-120
add_to_lists(is_haggkvist_nicoghossian, hamiltonian_sufficient, all_property_theorems)

# Fan, G. H. (1984). New sufficient conditions for cycles in graphs. Journal of Combinatorial Theory, 37(3), 221-227.
add_to_lists(is_genghua_fan, hamiltonian_sufficient, all_property_theorems)

# Lindquester, T. E. (1989). The effects of distance and neighborhood union conditions on hamiltonian properties in graphs. Journal of Graph Theory, 13(3), 335-352.
# Ore, Ø. (1960), "Note on Hamilton circuits", American Mathematical Monthly, 67 (1): 55, doi:10.2307/2308928, JSTOR 2308928.
def is_lindquester_or_is_ore(g):
    return is_lindquester(g) or is_ore(g)
add_to_lists(is_lindquester_or_is_ore, hamiltonian_sufficient, all_property_theorems)

# Trivial / "belongs to folklore"
def is_cycle_or_is_clique(g):
    return is_cycle(g) or g.is_clique()
add_to_lists(is_cycle_or_is_clique, hamiltonian_sufficient, all_property_theorems)

# Geng-Hua Fan. "New Sufficient Conditions for Cycles in Graphs". Journal of Combinatorial Theory 37.3(1984):221-227.
def sigma_dist2_geq_half_n(g):
    return sigma_dist2(g) >= g.order()/2
add_to_lists(sigma_dist2_geq_half_n, hamiltonian_sufficient, all_property_theorems)

# Bauer, Douglas, et al. "Long cycles in graphs with large degree sums." Discrete Mathematics 79.1 (1990): 59-70.
add_to_lists(is_bauer, hamiltonian_sufficient, all_property_theorems)

# Jung, H. A. "On maximal circuits in finite graphs." Annals of Discrete Mathematics. Vol. 3. Elsevier, 1978. 129-144.
add_to_lists(is_jung, hamiltonian_sufficient, all_property_theorems)

# S. Goodman and S. Hedetniemi, Sufficient Conditions for a Graph to Be Hamiltonian. Journal of Combinatorial Theory 16: 175--180, 1974.
def is_two_connected_claw_free_paw_free(g):
    return is_two_connected(g) and is_claw_free_paw_free(g)
add_to_lists(is_claw_free_paw_free, hamiltonian_sufficient, all_property_theorems)

# Ronald Gould, Updating the Hamiltonian problem — a survey. Journal of Graph Theory 15.2: 121-157, 1991.
add_to_lists(is_oberly_sumner, hamiltonian_sufficient, all_property_theorems)
add_to_lists(is_oberly_sumner_bull, hamiltonian_sufficient, all_property_theorems)
add_to_lists(is_oberly_sumner_p4, hamiltonian_sufficient, all_property_theorems)
add_to_lists(is_matthews_sumner, hamiltonian_sufficient, all_property_theorems)
add_to_lists(is_broersma_veldman_gould, hamiltonian_sufficient, all_property_theorems)

# Chvátal, Václav. "On Hamilton's ideals." Journal of Combinatorial Theory, Series B 12.2 (1972): 163-168.
add_to_lists(chvatals_condition, hamiltonian_sufficient, all_property_theorems)


####
# HAMILTONICITY NECESSARY CONDITIONS
####
