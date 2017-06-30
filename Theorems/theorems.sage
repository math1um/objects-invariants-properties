# THEORY

#####
# UPPER BOUND
#####

# R. Pepper. Binding independence. Ph. D. Dissertation. University of Houston. Houston, TX, 2004.
annihilation_thm = annihilation_number

# Nemhauser, George L., and Leslie Earl Trotter. "Vertex packings: structural properties and algorithms." Mathematical Programming 8.1 (1975): 232-248.
# Nemhauser, George L., and Leslie E. Trotter. "Properties of vertex packing and independence system polyhedra." Mathematical Programming 6.1 (1974): 48-61.
fractional_thm = fractional_alpha

# D. M. Cvetkovic, M. Doob, and H. Sachs. Spectra of graphs. Academic Press, New York, 1980.
cvetkovic_thm = cvetkovic

# Trivial
trivial_thm = Graph.order

# R. Pepper. Binding independence. Ph. D. Dissertation. University of Houston. Houston, TX, 2004.
def kwok_thm(g):
    return order(g) - (size(g)/max_degree(g))

# P. Hansen and M. Zheng. Sharp Bounds on the order, size, and stability number of graphs. NETWORKS 23 (1993), no. 2, 99-102.
def hansen_thm(g):
    return floor(1/2 + sqrt(1/4 + order(g)**2 - order(g) - 2*size(g)))

# Matching Number - Folklore
def matching_number_thm(g):
    return order(g) - matching_number(g)

# Min-Degree Theorm
def min_degree_thm(g):
    return order(g) - min_degree(g)

# Cut Vertices Theorem
# Three Bounds on the Independence Number of a Graph - C. E. Larson, R. Pepper
def cut_vertices_thm(g):
    return (g.order() - (card_cut_vertices(g)/2) - (1/2))

# Median Degree Theorem
# Three Bounds on the Independence Number of a Graph - C. E. Larson, R. Pepper
def median_degree_thm(g):
    return (g.order() - (median_degree(g)/2) - 1/2)

# Godsil-Newman Upper Bound theorem
# Godsil, Chris D., and Mike W. Newman. "Eigenvalue bounds for independent sets." Journal of Combinatorial Theory, Series B 98.4 (2008): 721-734.
def godsil_newman_thm(g):
    L = max(g.laplacian_matrix().change_ring(RDF).eigenvalues())
    return g.order()*(L-min_degree(g))/L

# AGX Upper Bound Theorem
#Aouchiche, Mustapha, Gunnar Brinkmann, and Pierre Hansen. "Variable neighborhood search for extremal graphs. 21. Conjectures and results about the independence number." Discrete Applied Mathematics 156.13 (2008): 2530-2542.
def AGX_upper_bound_thm(g):
    return (g.order() + max_degree(g) - ceil(2 * sqrt(g.order() - 1)))

alpha_upper_bounds = [annihilation_thm, fractional_thm, cvetkovic_thm, trivial_thm, kwok_thm, hansen_thm, matching_number_thm, min_degree_thm, cut_vertices_thm, median_degree_thm, godsil_newman_thm, AGX_upper_bound_thm]

#####
# LOWER BOUNDS
#####

# Radius Pendants Theorem
# Three Bounds on the Independence Number of a Graph - C. E. Larson, R. Pepper
def radius_pendants_thm(g):
    return (g.radius() + (card_pendants(g)/2) - 1)

# AGX Lower Bound Theorem
# Aouchiche, Mustapha, Gunnar Brinkmann, and Pierre Hansen. "Variable neighborhood search for extremal graphs. 21. Conjectures and results about the independence number." Discrete Applied Mathematics 156.13 (2008): 2530-2542.
def AGX_lower_bound_thm(g):
    return ceil(2 * sqrt(g.order()))

alpha_lower_bounds = [radius_pendants_thm, AGX_lower_bound_thm]
