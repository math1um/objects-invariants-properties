# Graph Lists

import pickle, os

graph_objects = []
alpha_critical_easy = []
alpha_critical_hard = []
chromatic_index_critical = []
chromatic_index_critical_7 = []
class0graphs = []
class0small = []
counter_examples = []
problem_graphs = []
sloane_graphs = []
non_connected_graphs = []
dimacs_graphs = []
all_graphs = []

# HexahedralGraph is CE to (((is_planar)&(is_regular))&(is_bipartite))->(has_residue_equals_alpha)
# WagnerGraph is a graph for which the Cvetkovic bound is the best upper bound present in the Willis Thesis
# OctohedralGraph is a graph for which the minimum degree is the best upper bound present in the Willis thesis
# BidiakisCube is a graph where none of the upper or lower bounds in the Willis thesis give the exact value for alpha
# TetrahedralGraph and MoserSpindle in the alpha critical list as "C~" and "FzEKW" respectively
# MeredithGraph and SchlaefliGraph are in the Problem Graphs list

# Removed graphs.McLaughlinGraph() and graphs.LocalMcLaughlinGraph() due to needing extra packages.

sage_graphs = [graphs.BullGraph(), graphs.ButterflyGraph(), graphs.ClawGraph(),
graphs.DiamondGraph(), graphs.HouseGraph(), graphs.HouseXGraph(), graphs.Balaban10Cage(),
graphs.Balaban11Cage(), graphs.BidiakisCube(),
graphs.BiggsSmithGraph(), graphs.BlanusaFirstSnarkGraph(), graphs.BlanusaSecondSnarkGraph(),
graphs.BrinkmannGraph(), graphs.BrouwerHaemersGraph(), graphs.BuckyBall(),
graphs.ChvatalGraph(), graphs.ClebschGraph(), graphs.CameronGraph(),
graphs.CoxeterGraph(), graphs.DesarguesGraph(), graphs.DejterGraph(), graphs.DoubleStarSnark(),
graphs.DurerGraph(), graphs.DyckGraph(), graphs.EllinghamHorton54Graph(),
graphs.EllinghamHorton78Graph(), graphs.ErreraGraph(), graphs.F26AGraph(), graphs.FlowerSnark(),
graphs.FolkmanGraph(), graphs.FosterGraph(), graphs.FranklinGraph(), graphs.FruchtGraph(),
graphs.GoldnerHararyGraph(), graphs.GossetGraph(), graphs.GrayGraph(), graphs.GrotzschGraph(),
graphs.HallJankoGraph(), graphs.HarborthGraph(), graphs.HarriesGraph(), graphs.HarriesWongGraph(),
graphs.HeawoodGraph(), graphs.HerschelGraph(), graphs.HigmanSimsGraph(), graphs.HoffmanGraph(),
graphs.HoffmanSingletonGraph(), graphs.HoltGraph(), graphs.HortonGraph(),
graphs.IoninKharaghani765Graph(), graphs.JankoKharaghaniTonchevGraph(), graphs.KittellGraph(),
graphs.KrackhardtKiteGraph(), graphs.Klein3RegularGraph(), graphs.Klein7RegularGraph(),
graphs.LjubljanaGraph(), graphs.M22Graph(),
graphs.MarkstroemGraph(), graphs.McGeeGraph(),
graphs.MoebiusKantorGraph(), graphs.NauruGraph(), graphs.PappusGraph(),
graphs.PoussinGraph(), graphs.PerkelGraph(), graphs.PetersenGraph(), graphs.RobertsonGraph(),
graphs.ShrikhandeGraph(), graphs.SimsGewirtzGraph(),
graphs.SousselierGraph(), graphs.SylvesterGraph(), graphs.SzekeresSnarkGraph(),
graphs.ThomsenGraph(), graphs.TietzeGraph(), graphs.TruncatedIcosidodecahedralGraph(),
graphs.TruncatedTetrahedralGraph(), graphs.Tutte12Cage(), graphs.TutteCoxeterGraph(),
graphs.TutteGraph(), graphs.WagnerGraph(), graphs.WatkinsSnarkGraph(), graphs.WellsGraph(),
graphs.WienerArayaGraph(), graphs.JankoKharaghaniGraph(1800),
graphs.JankoKharaghaniGraph(936),
graphs.HexahedralGraph(), graphs.DodecahedralGraph(), graphs.OctahedralGraph(), graphs.IcosahedralGraph()]

#These built in graphs are nameless so here they are given names

temp1 = graphs.Cell120()
temp1.name(new = "Cell120")

temp2 = graphs.Cell600()
temp2.name(new = "Cell600")

temp3 = graphs.MathonStronglyRegularGraph(0)
temp3.name(new = "Mathon Strongly Regular Graph 0")

temp4 = graphs.MathonStronglyRegularGraph(1)
temp4.name(new = "Mathon Strongly Regular Graph 1")

temp5 = graphs.MathonStronglyRegularGraph(2)
temp5.name(new = "Mathon Strongly Regular Graph 2")

for graph in sage_graphs + [temp1, temp2, temp3, temp4, temp5]:
    add_to_lists(graph, graph_objects, all_graphs)

add_to_lists(graphs.WorldMap(), non_connected_graphs, all_graphs)

# Meredith graph is 4-reg, class2, non-hamiltonian: http://en.wikipedia.org/wiki/Meredith_graph
add_to_lists(graphs.MeredithGraph(), problem_graphs, all_graphs)
add_to_lists(graphs.SchlaefliGraph(), problem_graphs, all_graphs)

# A graph is alpha_critical if removing any edge increases independence number
# All alpha critical graphs of orders 2 to 9, 53 in total

# "E|OW" is CE to (has_alpha_residue_equal_two)->((is_perfect)|(is_regular))

alpha_critical_graph_names = ['A_','Bw', 'C~', 'Dhc', 'D~{', 'E|OW', 'E~~w', 'FhCKG', 'F~[KG',
'FzEKW', 'Fn[kG', 'F~~~w', 'GbL|TS', 'G~?mvc', 'GbMmvG', 'Gb?kTG', 'GzD{Vg', 'Gb?kR_', 'GbqlZ_',
'GbilZ_', 'G~~~~{', 'GbDKPG', 'HzCGKFo', 'H~|wKF{', 'HnLk]My', 'HhcWKF_', 'HhKWKF_', 'HhCW[F_',
'HxCw}V`', 'HhcGKf_', 'HhKGKf_', 'Hh[gMEO', 'HhdGKE[', 'HhcWKE[', 'HhdGKFK', 'HhCGGE@', 'Hn[gGE@',
'Hn^zxU@', 'HlDKhEH', 'H~~~~~~', 'HnKmH]N', 'HnvzhEH', 'HhfJGE@', 'HhdJGM@', 'Hj~KHeF', 'HhdGHeB',
'HhXg[EO', 'HhGG]ES', 'H~Gg]f{', 'H~?g]vs', 'H~@w[Vs', 'Hn_k[^o']

for s in alpha_critical_graph_names:
    g = Graph(s)
    g.name(new="alpha_critical_"+ s)
    add_to_lists(g, alpha_critical_easy, graph_objects, all_graphs)

# All order-7 chromatic_index_critical_graphs (and all are overfull)
n7_chromatic_index_critical_names = ['FhCKG', 'FzCKW', 'FzNKW', 'FlSkG', 'Fn]kG', 'FlLKG', 'FnlkG', 'F~|{G', 'FnlLG', 'F~|\\G',
'FnNLG', 'F~^LW', 'Fll\\G', 'FllNG', 'F~l^G', 'F~|^w', 'F~~^W', 'Fnl^W', 'FlNNG', 'F|\\Kg',
'F~^kg', 'FlKMG']

for s in n7_chromatic_index_critical_names:
    g=Graph(s)
    g.name(new="chromatic_index_critical_7_" + s)
    add_to_lists(g, chromatic_index_critical, chromatic_index_critical_7, problem_graphs, all_graphs)

# Class 0 pebbling graphs
try:
    class0graphs_dict = pickle.load(open("objects-invariants-properties/Objects/class0graphs_dictionary.pickle","r"))
except:
    class0graphs_dict = {}

for d in class0graphs_dict:
    g = Graph(class0graphs_dict[d])
    g.name(new = d)
    add_to_lists(g, class0graphs, all_graphs)

class0small = [g for g in class0graphs if g.order() < 30]
for g in class0small:
    add_to_lists(g, problem_graphs)

alpha_critical_hard = [Graph('Hj\\x{F{')]
for g in alpha_critical_hard:
    add_to_lists(g, problem_graphs, all_graphs)

# Graph objects

p3 = graphs.PathGraph(3)
p3.name(new = "p3")
add_to_lists(p3, graph_objects, all_graphs)

p4 = graphs.PathGraph(4)
p4.name(new="p4")
add_to_lists(p4, graph_objects, all_graphs)

p5 = graphs.PathGraph(5)
p5.name(new = "p5")
add_to_lists(p5, graph_objects, all_graphs)

p6 = graphs.PathGraph(6)
p6.name(new="p6")
add_to_lists(p6, graph_objects, all_graphs)

"""
CE to independence_number(x) <= e^(cosh(max_degree(x) - 1))
 and to
independence_number(x) <= max_degree(x)*min_degree(x) + card_periphery(x)
"""
p9 = graphs.PathGraph(9)
p9.name(new = "p9")
add_to_lists(p9, graph_objects, counter_examples, all_graphs)

"""
P29 is a CE to independence_number(x) <=degree_sum(x)/sqrt(card_negative_eigenvalues(x))
 and to
<= max_degree(x)^e^card_center(x)
 and to
<= max_degree(x)^2 + card_periphery(x)
"""
p29 = graphs.PathGraph(29)
p29.name(new = "p29")
add_to_lists(p29, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= 2*cvetkovic(x)*log(10)/log(x.size())
p102 = graphs.PathGraph(102)
p102.name(new = "p102")
add_to_lists(p102, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x)<=welsh_powell(x)^e^different_degrees(x)
p6707 = graphs.PathGraph(6707)
p6707.name(new = "p6707")

c4 = graphs.CycleGraph(4)
c4.name(new="c4")
add_to_lists(c4, graph_objects, all_graphs)

c6 = graphs.CycleGraph(6)
c6.name(new = "c6")
add_to_lists(c6, graph_objects, all_graphs)

# CE to independence_number(x) <= (e^welsh_powell(x) - graph_rank(x))^2
c22 = graphs.CycleGraph(22)
c22.name(new = "c22")
add_to_lists(c22, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= minimum(cvetkovic(x), 2*e^sum_temperatures(x))
c34 = graphs.CycleGraph(34)
c34.name(new = "c34")
add_to_lists(c34, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= residue(x)^(degree_sum(x)^density(x))
c102 = graphs.CycleGraph(102)
c102.name(new = "c102")
add_to_lists(c102, graph_objects, counter_examples, all_graphs)

"""
Sage defines circulant graphs without the cycle, whereas the paper defines it with the cycle
From:
Brimkov, Valentin. "Algorithmic and explicit determination of the Lovasz number for certain circulant graphs." Discrete Applied Mathematics 155.14 (2007): 1812-1825.
"""
c13_2 = graphs.CirculantGraph(13, 2)
c13_2.add_cycle([0..12])
c13_2.name(new = "c13_2")
add_to_lists(c13_2, graph_objects, all_graphs)

k10 = graphs.CompleteGraph(10)
k10.name(new="k10")
add_to_lists(k10, graph_objects, all_graphs)

# CE to independence_number(x) >= floor(tan(floor(gutman_energy(x))))
k37 = graphs.CompleteGraph(37)
k37.name(new = "k37")
add_to_lists(k37, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= minimum(lovasz_theta(x), 2*e^sum_temperatures(x))
#   and to
# independence_number(x) <= minimum(floor(lovasz_theta(x)), 2*e^sum_temperatures(x))
#   and to
# independence_number(x) >= -brinkmann_steffen(x) + 1/2*card_center(x)
k1_9 = graphs.CompleteBipartiteGraph(1,9)
k1_9.name(new = "k1_9")
add_to_lists(k1_9, graph_objects, counter_examples, all_graphs)

# The line graph of k3,3
k3_3_line_graph = graphs.CompleteBipartiteGraph(3, 3).line_graph()
k3_3_line_graph.name(new = "k3_3 line graph")
add_to_lists(k3_3_line_graph, graph_objects, all_graphs)

k5_3=graphs.CompleteBipartiteGraph(5,3)
k5_3.name(new = "k5_3")
add_to_lists(k5_3, graph_objects, all_graphs)

# CE to independence_number(x) <= diameter^(max_degree-1)
# diameter is 16, Delta=3, alpha = 341
bt2_8 = graphs.BalancedTree(2,8)
bt2_8.name(new = "bt2_8")
add_to_lists(bt2_8, graph_objects, counter_examples, all_graphs)

#two c4's joined at a vertex
c4c4=graphs.CycleGraph(4)
for i in [4,5,6]:
    c4c4.add_vertex()
c4c4.add_edge(3,4)
c4c4.add_edge(5,4)
c4c4.add_edge(5,6)
c4c4.add_edge(6,3)
c4c4.name(new="c4c4")
add_to_lists(c4c4, graph_objects, all_graphs)

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
add_to_lists(c5c5, graph_objects, all_graphs)

K4a=graphs.CompleteGraph(4)
K4b=graphs.CompleteGraph(4)
K4a.delete_edge(0,1)
K4b.delete_edge(0,1)
regular_non_trans = K4a.disjoint_union(K4b)
regular_non_trans.add_edge((0,0),(1,1))
regular_non_trans.add_edge((0,1),(1,0))
regular_non_trans.name(new="regular_non_trans")
add_to_lists(regular_non_trans, graph_objects, all_graphs)

c6ee = graphs.CycleGraph(6)
c6ee.add_edges([(1,5), (2,4)])
c6ee.name(new="c6ee")
add_to_lists(c6ee, graph_objects, all_graphs)

#c6ee plus another chord: hamiltonian, regular, vertex transitive
c6eee = copy(c6ee)
c6eee.add_edge(0,3)
c6eee.name(new="c6eee")
add_to_lists(c6eee, graph_objects, all_graphs)

#c8 plus one long vertical chord and 3 parallel horizontal chords
c8chorded = graphs.CycleGraph(8)
c8chorded.add_edge(0,4)
c8chorded.add_edge(1,7)
c8chorded.add_edge(2,6)
c8chorded.add_edge(3,5)
c8chorded.name(new="c8chorded")
add_to_lists(c8chorded, graph_objects, all_graphs)

#c8 plus 2 parallel chords: hamiltonian, tri-free, not vertex-transitive
c8chords = graphs.CycleGraph(8)
c8chords.add_edge(1,6)
c8chords.add_edge(2,5)
c8chords.name(new="c8chords")
add_to_lists(c8chords, graph_objects, all_graphs)

prismsub = graphs.CycleGraph(6)
prismsub.add_edge(0,2)
prismsub.add_edge(3,5)
prismsub.add_edge(1,4)
prismsub.subdivide_edge(1,4,1)
prismsub.name(new="prismsub")
add_to_lists(prismsub, graph_objects, all_graphs)

# ham, not vertex trans, tri-free, not cartesian product
prismy = graphs.CycleGraph(8)
prismy.add_edge(2,5)
prismy.add_edge(0,3)
prismy.add_edge(4,7)
prismy.name(new="prismy")
add_to_lists(prismy, graph_objects, all_graphs)

#c10 with chords, ham, tri-free, regular, planar, vertex transitive
sixfour = graphs.CycleGraph(10)
sixfour.add_edge(1,9)
sixfour.add_edge(0,2)
sixfour.add_edge(3,8)
sixfour.add_edge(4,6)
sixfour.add_edge(5,7)
sixfour.name(new="sixfour")
add_to_lists(sixfour, graph_objects, all_graphs)

#unique 24-vertex fullerene: hamiltonian, planar, not vertex transitive
c24 = Graph('WsP@H?PC?O`?@@?_?GG@??CC?G??GG?E???o??B???E???F')
c24.name(new="c24")
add_to_lists(c24, graph_objects, all_graphs)

#unique 26-atom fullerene: hamiltonian, planar, not vertex trans, radius=5, diam=6
c26 = Graph('YsP@H?PC?O`?@@?_?G?@??CC?G??GG?E??@_??K???W???W???H???E_')
c26.name(new="c26")
add_to_lists(c26, graph_objects, all_graphs)

#the unique 100-atom fullerene with minimum independence number of 43 (and IPR, tetrahedral symmetry)
c100 = Graph("~?@csP@@?OC?O`?@?@_?O?A??W??_??_G?O??C??@_??C???G???G@??K???A????O???@????A????A?G??B?????_????C?G???O????@_?????_?????O?????C?G???@_?????E??????G??????G?G????C??????@???????G???????o??????@???????@????????_?_?????W???????@????????C????????G????????G?G??????E????????@_????????K?????????_????????@?@???????@?@???????@_?????????G?????????@?@????????C?C????????W??????????W??????????C??????????@?@?????????G???????????_??????????@?@??????????_???????????O???????????C?G??????????O???????????@????????????A????????????A?G??????????@_????????????W????????????@_????????????E?????????????E?????????????E?????????????B??????????????O?????????????A@?????????????G??????????????OG?????????????O??????????????GC?????????????A???????????????OG?????????????@?_?????????????B???????????????@_???????????????W???????????????@_???????????????F")
c100.name(new="c100")
add_to_lists(c100, graph_objects, all_graphs)

"""
The Holton-McKay graph is the smallest planar cubic hamiltonian graph with an edge
that is not contained in a hamiltonian cycle. It has 24 vertices and the edges (0,3)
and (4,7) are not contained in a hamiltonian cycle. This graph was mentioned in
D. A. Holton and B. D. McKay, Cycles in 3-connected cubic planar graphs II, Ars
Combinatoria, 21A (1986) 107-114.

    sage: holton_mckay
    holton_mckay: Graph on 24 vertices
    sage: holton_mckay.is_planar()
    True
    sage: holton_mckay.is_regular()
    True
    sage: max(holton_mckay.degree())
    3
    sage: holton_mckay.is_hamiltonian()
    True
    sage: holton_mckay.radius()
    4
    sage: holton_mckay.diameter()
    6
"""
holton_mckay = Graph('WlCGKS??G?_D????_?g?DOa?C?O??G?CC?`?G??_?_?_??L')
holton_mckay.name(new="holton_mckay")
add_to_lists(holton_mckay, graph_objects, all_graphs)

#an example of a bipartite, 1-tough, not van_den_heuvel, not hamiltonian graph
kratsch_lehel_muller = graphs.PathGraph(12)
kratsch_lehel_muller.add_edge(0,5)
kratsch_lehel_muller.add_edge(6,11)
kratsch_lehel_muller.add_edge(4,9)
kratsch_lehel_muller.add_edge(1,10)
kratsch_lehel_muller.add_edge(2,7)
kratsch_lehel_muller.name(new="kratsch_lehel_muller")
add_to_lists(kratsch_lehel_muller, graph_objects, all_graphs)

#ham, not planar, not anti_tutte
c6xc6 = graphs.CycleGraph(6).cartesian_product(graphs.CycleGraph(6))
c6xc6.name(new="c6xc6")
add_to_lists(c6xc6, graph_objects, all_graphs)

c7xc7 = graphs.CycleGraph(7).cartesian_product(graphs.CycleGraph(7))
c7xc7.name(new="c7xc7")
add_to_lists(c7xc7, graph_objects, all_graphs)

# Product Graphs, fig. 1.13
c6xk2 = graphs.CycleGraph(6).cartesian_product(graphs.CompleteGraph(2))
c6xk2.name(new = "c6xk2")
add_to_lists(c6xk2, graph_objects, all_graphs)

# Product Graphs, fig. 1.13
k1_4xp3 = graphs.CompleteBipartiteGraph(1, 4).cartesian_product(graphs.PathGraph(3))
k1_4xp3.name(new = "k1_4xp3")
add_to_lists(k1_4xp3, graph_objects, all_graphs)

# Product Graphs, fig. 1.14
p4xk3xk2 = graphs.PathGraph(4).cartesian_product(graphs.CompleteGraph(3)).cartesian_product(graphs.CompleteGraph(2))
p4xk3xk2.name(new = "p4xk3xk2")
add_to_lists(p4xk3xk2, graph_objects, all_graphs)

# Product Graphs, fig. 4.1
p3xk2xk2 = graphs.PathGraph(3).cartesian_product(graphs.CompleteGraph(2)).cartesian_product(graphs.CompleteGraph(2))
p3xk2xk2.name(new = "p3xk2xk2")
add_to_lists(p3xk2xk2, graph_objects, all_graphs)

# Product Graphs, fig. 5.1
p4Xp5 = graphs.PathGraph(4).strong_product(graphs.PathGraph(5))
p4Xp5.name(new = "p4Xp5")
add_to_lists(p4Xp5, graph_objects, all_graphs)

# Product Graphs, fig. 5.4
p5txp3 = graphs.PathGraph(5).tensor_product(graphs.PathGraph(3))
p5txp3.name(new = "p5txp3")
add_to_lists(p5txp3, non_connected_graphs, all_graphs)

# Product Graphs, Fig 6.1
k3lxp3 = graphs.CompleteGraph(3).lexicographic_product(graphs.PathGraph(3))
k3lxp3.name(new = "k3lxp3")
add_to_lists(k3lxp3, graph_objects, all_graphs)

# Product Graphs, Fig 6.1
p3lxk3 = graphs.PathGraph(3).lexicographic_product(graphs.CompleteGraph(3))
p3lxk3.name(new = "p3lxk3")
add_to_lists(p3lxk3, graph_objects, all_graphs)

"""
Referenced p.15
Mathew, K. Ashik, and Patric RJ Östergård. "New lower bounds for the Shannon capacity of odd cycles." Designs, Codes and Cryptography (2015): 1-10.
"""
c5Xc5 = graphs.CycleGraph(5).strong_product(graphs.CycleGraph(5))
c5Xc5.name(new = "c5Xc5")
add_to_lists(c5Xc5, graph_objects, all_graphs)

#non-ham, 2-connected, eulerian (4-regular)
gould = Graph('S~dg?CB?wC_L????_?W?F??c?@gOOOGGK')
gould.name(new="gould")
add_to_lists(gould, graph_objects, all_graphs)

#two k5s with single edge removed from each and lines joining these 4 points to a new center point, non-hamiltonian
throwing = Graph('J~wWGGB?wF_')
throwing.name(new="throwing")
add_to_lists(throwing, graph_objects, all_graphs)

#k4 plus k2 on one side, open k5 on other, meet at single point in center, non-hamiltonian
throwing2 = Graph("K~wWGKA?gB_N")
throwing2.name(new="throwing2")
add_to_lists(throwing2, graph_objects, all_graphs)

#similar to throwing2 with pair of edges swapped, non-hamiltonian
throwing3 = Graph("K~wWGGB?oD_N")
throwing3.name(new="throwing3")
add_to_lists(throwing3, graph_objects, all_graphs)

#graph has diameter != radius but is hamiltonian
tent = graphs.CycleGraph(4).join(Graph(1),labels="integers")
tent.name(new="tent")
add_to_lists(tent, graph_objects, all_graphs)

# C5 with chords from one vertex to other 2 (showed up in auto search for CE's): hamiltonian
bridge = Graph("DU{")
bridge.name(new="bridge")
add_to_lists(bridge, graph_objects, all_graphs)

# nico found the smallest hamiltonian overfull graph
non_ham_over = Graph("HCQRRQo")
non_ham_over.name(new="non_ham_over")
add_to_lists(non_ham_over, graph_objects, all_graphs)

# From Ryan Pepper
ryan = Graph("WxEW?CB?I?_R????_?W?@?OC?AW???O?C??B???G?A?_??R")
ryan.name(new="ryan")
add_to_lists(ryan, graph_objects, all_graphs)

# Ryan Pepper
# CE to independence_number(x) <= 2 * chromatic_number(x) + 2 * residue(x)
# has alpha=25,chi=2,residue=10
ryan2=graphs.CirculantGraph(50,[1,3])
ryan2.name(new="circulant_50_1_3")
add_to_lists(ryan2, graph_objects, counter_examples, all_graphs)

# From Ryan Pepper
# CE to independence_number(x) >= diameter(x) - 1 for regular graphs
pepper_1_gadget = Graph('Ot???CA?WB`_B@O_B_B?A')
pepper_1_gadget.name(new = "pepper_1_gadget")
add_to_lists(pepper_1_gadget, graph_objects, counter_examples, all_graphs)

# p10 joined to 2 points of k4
# CE to chromatic_number <= avg_degree + 1
p10k4=Graph('MhCGGC@?G?_@_B?B_')
p10k4.name(new="p10k4")
add_to_lists(p10k4, graph_objects, counter_examples, all_graphs)

# star on 13 points with added edge:
# CE to independence_number(x) <= dom + girth(x)^2
s13e = Graph('M{aCCA?_C?O?_?_??')
s13e.name(new="s13e")
add_to_lists(s13e, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= 2 * girth(x)^2 + 2
# Star with 22 rays plus extra edge
s22e = graphs.StarGraph(22)
s22e.add_edge(1,2)
s22e.name(new="s22e")
add_to_lists(s22e, graph_objects, counter_examples, all_graphs)

# Graph from delavina's jets paper
starfish = Graph('N~~eeQoiCoM?Y?U?F??')
starfish.name(new="starfish")
add_to_lists(starfish, graph_objects, all_graphs)

# difficult graph from INP: order=11, alpha=4, best lower bound < 3
difficult11 = Graph('J?`FBo{fdb?')
difficult11.name(new="difficult11")
add_to_lists(difficult11, graph_objects, all_graphs)

# c4 joined to K# at point: not KE, alpha=theta=nu=3, delting any vertex gives KE graph
c5k3=Graph('FheCG')
c5k3.name(new="c5k3")
add_to_lists(c5k3, graph_objects, all_graphs)

# mycielskian of a triangle:
# CE to chi <= max(clique, nu)
# chi=4, nu = clique = 3
c3mycielski = Graph('FJnV?')
c3mycielski.name(new="c3mycieski")
add_to_lists(c3mycielski, problem_graphs , counter_examples, all_graphs)

# 4th mycielskian of a triangle,
# CE to chi <= clique + girth
# chi = 7, clique = girth = 3
c3mycielski4 = Graph('~??~??GWkYF@BcuIsJWEo@s?N?@?NyB`qLepJTgRXkAkU?JPg?VB_?W[??Ku??BU_??ZW??@u???Bs???Bw???A??F~~_B}?^sB`o[MOuZErWatYUjObXkZL_QpWUJ?CsYEbO?fB_w[?A`oCM??DL_Hk??DU_Is??Al_Dk???l_@k???Ds?M_???V_?{????oB}?????o[M?????WuZ?????EUjO?????rXk?????BUJ??????EsY??????Ew[??????B`o???????xk???????FU_???????\\k????????|_????????}_????????^_?????????')
c3mycielski4.name(new="c3mycielski4")
add_to_lists(c3mycielski4, graph_objects, counter_examples, all_graphs)

# A PAW is a traingle with a pendant
# Shows up in a sufficient condition for hamiltonicity
paw=Graph('C{')
paw.name(new="paw")
add_to_lists(paw, graph_objects, all_graphs)

# 2 octahedrons, remove one edge from each, add vertex, connect it to deleted edge vertices
# its regular of degree 4
binary_octahedron = Graph('L]lw??B?oD_Noo')
binary_octahedron.name(new = "binary_octahedron")
add_to_lists(binary_octahedron, graph_objects, all_graphs)

# this graph shows that the cartesian product of 2 KE graphs is not necessarily KE
# appears in Abay-Asmerom, Ghidewon, et al. "Notes on the independence number in the Cartesian product of graphs." Discussiones Mathematicae Graph Theory 31.1 (2011): 25-35.
paw_x_paw = paw.cartesian_product(paw)
paw_x_paw.name(new = "paw_x_paw")
add_to_lists(paw_x_paw, graph_objects, all_graphs)

#a DART is a kite with a pendant
dart = Graph('DnC')
dart.name(new="dart")
add_to_lists(dart, graph_objects, all_graphs)

# CE to ((is_chordal)^(is_forest))->(has_residue_equals_alpha)
ce2=Graph("HdGkCA?")
ce2.name(new = "ce2")
add_to_lists(ce2, graph_objects, counter_examples, all_graphs)

# CE to ((~(is_planar))&(is_chordal))->(has_residue_equals_alpha)
ce4=Graph("G~sNp?")
ce4.name(new = "ce4")
add_to_lists(ce4, graph_objects, counter_examples, all_graphs)

# CE to (((is_line_graph)&(is_cartesian_product))|(is_split))->(has_residue_equals_alpha)
ce5=Graph("X~}AHKVB{GGPGRCJ`B{GOO`C`AW`AwO`}CGOO`AHACHaCGVACG^")
ce5.name(new = "ce5")
add_to_lists(ce5, graph_objects, counter_examples, all_graphs)

# CE to (is_split)->((order_leq_twice_max_degree)&(is_chordal))
ce6 = Graph("H??E@cN")
ce6.name(new = "ce6")
add_to_lists(ce6, graph_objects, counter_examples, all_graphs)

# CE to (has_residue_equals_alpha)->((is_bipartite)->(order_leq_twice_max_degree))
ce7 = Graph("FpGK?")
ce7.name(new = "ce7")
add_to_lists(ce7, graph_objects, counter_examples, all_graphs)

# CE to ((has_paw)&(is_circular_planar))->(has_residue_equals_alpha)
ce8 = Graph('IxCGGC@_G')
ce8.name(new = "ce8")
add_to_lists(ce8, graph_objects, counter_examples, all_graphs)

# CE to ((has_H)&(is_forest))->(has_residue_equals_alpha)
ce9 = Graph('IhCGGD?G?')
ce9.name(new = "ce9")
add_to_lists(ce9, graph_objects, counter_examples, all_graphs)

# CE to (((is_eulerian)&(is_planar))&(has_paw))->(has_residue_equals_alpha)
ce10=Graph('KxkGGC@?G?o@')
ce10.name(new = "ce10")
add_to_lists(ce10, graph_objects, counter_examples, all_graphs)

# CE to (((is_cubic)&(is_triangle_free))&(is_H_free))->(has_residue_equals_two)
ce12 = Graph("Edo_")
ce12.name(new = "ce12")
add_to_lists(ce12, graph_objects, counter_examples, all_graphs)

# CE to ((diameter_equals_twice_radius)&(is_claw_free))->(has_residue_equals_two)
ce13 = Graph("ExOG")
ce13.name(new = "ce13")
add_to_lists(ce13, graph_objects, counter_examples, all_graphs)

# CE to (~(matching_covered))->(has_residue_equals_alpha)
ce14 = Graph('IhCGGC_@?')
ce14.name(new = "IhCGGC_@?")
add_to_lists(ce14, graph_objects, counter_examples, all_graphs)

"""
CE to independence_number(x) <= 10^order_automorphism_group(x)

    sage: order(ce15)
    57
    sage: independence_number(ce15)
    25
"""
ce15 = Graph("x??C?O?????A?@_G?H??????A?C??EGo?@S?O@?O??@G???CO???CAC_??a?@G?????H???????????O?_?H??G??G??@??_??OA?OCHCO?YA????????A?O???G?O?@????OOC???_@??????MCOC???O_??[Q??@???????O??_G?P?GO@A?G_???A???A@??g???W???@CG_???`_@O??????@?O@?AGO?????C??A??F??????@C????A?E@L?????P@`??")
ce15.name(new = "ce15")
add_to_lists(ce15, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= 2*maximum(welsh_powell(x), max_even_minus_even_horizontal(x))
ce16 = Graph("mG???GP?CC?Aa?GO?o??I??c??O??G?ACCGW@????OC?G@?_A_W_OC@??@?I??O?_AC?Oo?E@_?O??I??B_?@_A@@@??O?OC?GC?CD?C___gAO?G??KOcGCiA??SC????GAVQy????CQ?cCACKC_?A?E_??g_AO@C??c??@@?pY?G?")
ce16.name(new = "ce16")
add_to_lists(ce16, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) >= 1/2*cvetkovic(x)
ce17 = Graph("S??wG@@h_GWC?AHG?_gMGY_FaIOk@?C?S")
ce17.name(new = "ce17")
add_to_lists(ce17, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) >= matching_number - sigma_2
ce18 = Graph("cGO_?CCOB@O?oC?sTDSOCC@O???W??H?b???hO???A@CCKB??I??O??AO@CGA???CI?S?OGG?ACgQa_Cw^GP@AID?Gh??ogD_??dR[?AG?")
ce18.name(new = "ce18")
add_to_lists(ce18, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= maximum(max_even_minus_even_horizontal(x), radius(x)*welsh_powell(x))
ce19 = Graph('J?@OOGCgO{_')
ce19.name(new = "ce19")
add_to_lists(ce19, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= card_center(x) + max_even_minus_even_horizontal(x) + 1
ce20 = Graph('M?CO?k?OWEQO_O]c_')
ce20.name(new = "ce20")
add_to_lists(ce20, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= median_degree(x)^2 + card_periphery(x)
ce21 = Graph('FiQ?_')
ce21.name(new = "ce21")
add_to_lists(ce21, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= brinkmann_steffen(x) + max_even_minus_even_horizontal(x) + 1
ce22 = Graph('Ss?fB_DYUg?gokTEAHC@ECSMQI?OO?GD?')
ce22.name(new = "ce22")
add_to_lists(ce22, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= inverse_degree(x) + order_automorphism_group(x) + 1
ce23 = Graph("HkIU|eA")
ce23.name(new = "ce23")
add_to_lists(ce23, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= ceil(eulerian_faces(x)/diameter(x)) +max_even_minus_even_horizontal(x)
ce24 = Graph('JCbcA?@@AG?')
ce24.name(new = "ce24")
add_to_lists(ce24, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= floor(e^(maximum(max_even_minus_even_horizontal(x), fiedler(x))))
ce25 = Graph('OX??ZHEDxLvId_rgaC@SA')
ce25.name(new = "ce25")
add_to_lists(ce25, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= maximum(card_periphery(x), radius(x)*welsh_powell(x))
ce26 = Graph("NF?_?o@?Oa?BC_?OOaO")
ce26.name(new = "ce26")
add_to_lists(ce26, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= floor(average_distance(x)) + maximum(max_even_minus_even_horizontal(x), brinkmann_steffen(x))
ce27 = Graph("K_GBXS`ysCE_")
ce27.name(new = "ce27")
add_to_lists(ce27, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= minimum(annihilation_number(x), 2*e^sum_temperatures(x))
ce28 = Graph("g??O?C_?`?@?O??A?A????????C?????G?????????A@aA??_???G??GA?@????????_???GHC???CG?_???@??_??OB?C?_??????_???G???C?O?????O??A??????G??")
ce28.name(new = "ce28")
add_to_lists(ce28, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= maximum(2*welsh_powell(x), maximum(max_even_minus_even_horizontal(x), laplacian_energy(x)))
ce29 = Graph("P@g??BSCcIA???COcSO@@O@c")
ce29.name(new = "ce29")
add_to_lists(ce29, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= maximum(order_automorphism_group(x), 2*cvetkovic(x) - matching_number(x))
ce30 = Graph("G~q|{W")
ce30.name(new = "ce30")
add_to_lists(ce30, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= max_even_minus_even_horizontal(x) + min_degree(x) + welsh_powell(x)
ce31 = Graph("VP??oq_?PDOGhAwS??bSS_nOo?OHBqPi?I@AGP?POAi?")
ce31.name(new = "ce31")
add_to_lists(ce31, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) >= order(x)/szekeres_wilf(x)
ce32 = Graph('H?`@Cbg')
ce32.name(new = "ce32")
add_to_lists(ce32, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= max_even_minus_even_horizontal(x) + minimum(card_positive_eigenvalues(x), card_center(x) + 1)
ce33 = Graph("O_aHgP_kVSGOCXAiODcA_")
ce33.name(new = "ce33")
add_to_lists(ce33, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= card_center(x) + maximum(diameter(x), card_periphery(x))
ce34 = Graph('H?PA_F_')
ce34.name(new = "ce34")
add_to_lists(ce34, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= card_center(x) + maximum(diameter(x), card_periphery(x))ce35 = Graph("")
ce35 = Graph("HD`cgGO")
ce35.name(new = "ce35")
add_to_lists(ce35, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) >= max_degree(x) - order_automorphism_group(x)
ce36 = Graph('ETzw')
ce36.name(new = "ce36")
add_to_lists(ce36, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= maximum(card_center(x), diameter(x)*max_degree(x))
ce37 = Graph("~?AA?G?????@@??@?A???????????O??????????G_?A???????????????A?AO?????????G???G?@???@???O?????????????C???????_???????C?_?W???C????????_??????????????????_???????_???O????????D??????????C????????GCC???A??G??????A@??A??@G???_?????@_??????_??G???K??????A????C??????????A???_?A????`??C_O????G????????????A?G???????????????????O?????C??????@???__?@O_G??C????????OA?????????????????????????GA_GA????O???_??O??O?G??G?_C???@?G???O???_?O???_??????C???????????????E_???????????????_@???O??????CC???O?????????OC_????_A????????_?G??????O??????_??????_?I?O??????A???????O?G?O???C@????????????_@????C?????@@???????C???O??A?????_??????A_??????????A?G????AB???A??C?G??????????G???A??@?A???????@???????D?_????B????????????????????g?C???C????G????????@??????@??A????????@????_??_???o?????????@????????????_???????A??????C????A?????C????O????@?@???@?A_????????CA????????????????H???????????????????O????_??OG??Ec?????O??A??_???_???O?C??`?_@??@??????O????G????????????A????@???_?????????_?A???AAG???O????????????????????C???_???@????????????_??H???A??W?O@????@_???O?_A??O????OG???????G?@??G?C?????G?????????@?????????G?O?????G???????_?????????@????@?????????G????????????C?G?????????_C?@?A????G??GA@????????????@?????C??G??????_?????????_@?????@???A?????@?????????????????CG??????_?????@???????@C???O????_`?????OA?G??????????????Q?A?????????????A????@C?????GO??_?C???????O???????@?G?A????O??G???_????_?????A?G_?C?????????C?")
ce37.name(new = "ce37")
add_to_lists(ce37, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= abs(-card_center(x) + min_degree(x)) + max_even_minus_even_horizontal(x)
ce38 = Graph('FVS_O')
ce38.name(new = "ce38")
add_to_lists(ce38, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= abs(-card_center(x) + max_degree(x)) + max_even_minus_even_horizontal(x)
ce39 = Graph("FBAuo")
ce39.name(new = "ce39")
add_to_lists(ce39, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= floor(inverse_degree(x)) + order_automorphism_group(x) + 1
ce40 = Graph('Htji~Ei')
ce40.name(new = "ce40")
add_to_lists(ce40, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= card_center(x) + maximum(residue(x), card_periphery(x))
ce42 = Graph('GP[KGC')
ce42.name(new = "ce42")
add_to_lists(ce42, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= maximum(girth(x), (barrus_bound(x) - order_automorphism_group(x))^2)
ce43 = Graph("Exi?")
ce43.name(new = "ce43")
add_to_lists(ce43, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= (brinkmann_steffen(x) - szekeres_wilf(x))^2 + max_even_minus_even_horizontal(x)
ce44 = Graph('GGDSsg')
ce44.name(new = "ce44")
add_to_lists(ce44, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= maximum(max_even_minus_even_horizontal(x), radius(x)*szekeres_wilf(x))
ce45 = Graph("FWKH?")
ce45.name(new = "ce45")
add_to_lists(ce45, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= maximum(card_periphery(x), radius(x)*szekeres_wilf(x))
ce46 = Graph('F`I`?')
ce46.name(new = "ce46")
add_to_lists(ce46, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= maximum(card_periphery(x), diameter(x) + inverse_degree(x))
ce47 = Graph("KVOzWAxewcaE")
ce47.name(new = "ce47")
add_to_lists(ce47, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= maximum(card_periphery(x), max_even_minus_even_horizontal(x) + min_degree(x))
ce48 = Graph('Iq]ED@_s?')
ce48.name(new = "ce48")
add_to_lists(ce48, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) >= sqrt(card_positive_eigenvalues(x))
ce49 = Graph("K^~lmrvv{~~Z")
ce49.name(new = "ce49")
add_to_lists(ce49, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= max_degree(x) + maximum(max_even_minus_even_horizontal(x), sigma_2(x))
ce50 = Graph('bCaJf?A_??GY_O?KEGA???OMP@PG???G?CO@OOWO@@m?a?WPWI?G_A_?C`OIG?EDAIQ?PG???A_A?C??CC@_G?GDI]CYG??GA_A??')
ce50.name(new = "ce50")
add_to_lists(ce50, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) >= matching_number(x) - order_automorphism_group(x) - 1
ce51 = Graph("Ivq~j^~vw")
ce51.name(new = "ce51")
add_to_lists(ce51, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) >= order(x)/szekeres_wilf(x)
ce52 = Graph('H?QaOiG')
ce52.name(new = "ce52")
add_to_lists(ce52, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) >= matching_number(x) - sigma_2(x) - 1
ce53 = Graph("]?GEPCGg]S?`@??_EM@OTp?@E_gm?GW_og?pWO?_??GQ?A?^HIRwH?Y?__BC?G?[PD@Gs[O?GW")
ce53.name(new = "ce53")
add_to_lists(ce53, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) >= -average_distance(x) + ceil(lovasz_theta(x))
ce54 = Graph('lckMIWzcWDsSQ_xTlFX?AoCbEC?f^xwGHOA_q?m`PDDvicEWP`qA@``?OEySJX_SQHPc_H@RMGiM}`CiG?HCsm_JO?QhI`?ARLAcdBAaOh_QMG?`D_o_FvQgHGHD?sKLEAR^ASOW~uAUQcA?SoD?_@wECSKEc?GCX@`DkC')
ce54.name(new = "ce54")
add_to_lists(ce54, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) >= -card_periphery(x) + matching_number(x)
ce55 = Graph("I~~~~~~zw")
ce55.name(new = "ce55")
add_to_lists(ce55, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) >= lovasz_theta(x)/edge_con(x)
ce56 = Graph('HsaGpOe')
ce56.name(new = "ce56")
add_to_lists(ce56, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) >= minimum(max_degree(x), floor(lovasz_theta(x)))
ce57 = Graph("^?H{BDHqHosG??OkHOhE??B[CInU?@j_A?CoA^azGPLcb_@GEYYRPgG?K@gdPAg?d@_?_sGcED`@``O")
ce57.name(new = "ce57")
add_to_lists(ce57, graph_objects, counter_examples, all_graphs)

# CE to independence_number>= barrus_bound(x) - max(card_center(x), card_positive_eigenvalues(x))
ce58 = Graph('Sj[{Eb~on~nls~NJWLVz~~^|{l]b\uFss')
ce58.name(new = "ce58")
add_to_lists(ce58, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) >= floor(tan(barrus_bound(x) - 1))
ce59 = Graph("RxCWGCB?G?_B?@??_?N??F??B_??w?")
ce59.name(new = "ce59")
add_to_lists(ce59, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) >= -1/2*diameter(x) + lovasz_theta(x)
ce60 = Graph('wSh[?GCfclJm?hmgA^We?Q_KIXbf\@SgDNxpwHTQIsIB?MIDZukArBAeXE`vqDLbHCwf{fD?bKSVLklQHspD`Lo@cQlEBFSheAH?yW\YOCeaqmOfsZ?rmOSM?}HwPCIAYLdFx?o[B?]ZYb~IK~Z`ol~Ux[B]tYUE`_gnVyHRQ?{cXG?k\BL?vVGGtCufY@JIQYjByg?Q?Qb`SKM`@[BVCKDcMxF|ADGGMBW`ANV_IKw??DRkY\KOCW??P_?ExJDSAg')
ce60.name(new = "ce60")
add_to_lists(ce60, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= maximum(card_negative_eigenvalues(x), max_common_neighbors(x) + max_even_minus_even_horizontal(x))
ce61 = Graph("KsaAA?OOC??C")
ce61.name(new = "ce61")
add_to_lists(ce61, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) >= minimum(floor(lovasz_theta(x)), tan(spanning_trees_count(x)))
ce62 = Graph("qWGh???BLQcAH`aBAGCScC@SoBAAFYAG?_T@@WOEBgRC`oSE`SG@IoRCK[_K@QaQq?c@?__G}ScHO{EcCa?K?o?E?@?C[F_@GpV?K_?_?CSW@D_OCr?b_XOag??C@gGOGh??QFoS?@OHDAKWIX_OBbHGOl??\Cb@?E`WehiP@IGAFC`GaCgC?JjQ???AGJgDJAGsdcqEA_a_q?")
ce62.name(new = "ce62")
add_to_lists(ce62, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) >= diameter(x)/different_degrees(x)
ce63 = Graph("KOGkYBOCOAi@")
ce63.name(new = "ce63")
add_to_lists(ce63, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) >= -max_common_neighbors(x) + min_degree(x)
ce64 = Graph('`szvym|h~RMQLTNNiZzsgQynDR\p~~rTZXi~n`kVvKolVJfP}TVEN}Thj~tv^KJ}D~VqqsNy|NY|ybklZLnz~TfyG')
ce64.name(new = "ce64")
add_to_lists(ce64, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) >= -10^different_degrees(x) + matching_number(x)
ce65 = Graph("W~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
ce65.name(new = "ce65")
add_to_lists(ce65, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) >= girth^max_degree+1
ce66 = Graph("~?@EG??????????@G????_???a???C????????@???A???????G??????C?GCG????????A???C@??????@????O??A??C?????_??O???CA???c??_?_?@????A????@??????C???C?G?O?C???G?????????O?_G?C????G??????_?????@??G???C??????O?GA?????O???@????????A?G?????????_C???????@??G??@??_??IA@???????G?@??????@??_?@????C??G???_????O???P???@???o??????O?????S?O???A???G?????c_?????D?????A???A?????G@???????O???H????O????@@????@K????????C??C?????G??")
ce66.name(new = "ce66")
add_to_lists(ce66, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= maximum(cycle_space_dimension(x), floor(lovasz_theta(x)))
ce67 = Graph("G??EDw")
ce67.name(new = "ce67")
add_to_lists(ce67, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) >= minimum(card_positive_eigenvalues(x), 2*card_zero_eigenvalues(x))
ce68 = Graph('HzzP|~]')
ce68.name(new = "ce68")
add_to_lists(ce68, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= maximum(max_degree(x), radius(x)^card_periphery(x))
ce69 = Graph("F?BvO")
ce69.name(new = "ce69")
add_to_lists(ce69, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) >= floor(lovasz_theta(x))/vertex_con(x)
ce70 = Graph('~?@Z??????O?M??`S??A?`?A?????@????`?????A?A?????A@????GO?@@??A_????????O_???I@_??G??A?`?C????????@???????????@??C?@?????O??@??CA??A?D??G?_?????_Q@G????C?_?A??@???O????G?O?G?_?????CoG?G???X??C???_CAG_C??????G?????@?Ao?????C???A??????_??SG??cOC??????????Ao????????_?????G???????D?????C??_?B?????a??_???????G?@?????C??????C?c?????G_?_??G??_Q????C????B?_CG????AGC???G?O??_I????@??????_??a??@?O_G??O??aA@@?????EA???@???????@???????O?O??@??`_G???????GCA?_GO????_?_????????????_??I?@?C???@????????G?aG??????W????@PO@???oC?CO???_??G?@@?CO??K???C@??O???@????D?????A?@G?G?O???_???????Ao??AC???G?_???G????????A??????_?p???W?A?Ao@?????_?????GA??????????????_?C??????@O????_@??O@Gc@??????????A_??????')
ce70.name(new = "ce70")
add_to_lists(ce70, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= maximum(matching_number(x), critical_independence_number(x))
ce71 = Graph('ECYW')
ce71.name(new = "ce71")
add_to_lists(ce71, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x)>=-1/2*x.diameter() + x.lovasz_theta()
ce72 = Graph('fdSYkICGVs_m_TPs`Fmj_|pGhC@@_[@xWawsgEDe_@g`TC{P@pqGoocqOw?HBDS[R?CdG\e@kMCcgqr?G`NHGXgYpVGCoJdOKBJQAsG|ICE_BeMQGOwKqSd\W?CRg')
ce72.name(new = "ce72")
add_to_lists(ce72, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) >= minimum(floor(lovasz_theta(x)), max_even_minus_even_horizontal(x) + 1)
ce73 = Graph('h???_?CA?A?@AA????OPGoC@????A@?A?_C?C?C_A_???_??_G????HG????c?G_?G??HC??A@GO?G?A@A???_@G_?_G_GC_??E?O?O`??@C?@???O@?AOC?G?H??O?P??C_?O_@??')
ce73.name(new = "ce73")
add_to_lists(ce73, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) >= minimum(diameter(x), lovasz_theta(x))
ce74 = Graph("FCQb_")
ce74.name(new = "ce74")
add_to_lists(ce74, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) >= minimum(girth(x), floor(lovasz_theta(x)))
ce75 = Graph('E?Bw')
ce75.name(new = "ce75")
add_to_lists(ce75, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= maximum(average_distance(x), max_even_minus_even_horizontal(x))*sum_temperatures(x)
ce76 = Graph("~?@DS?G???G_?A_?OA?GC??oa?A@?@?K???L?_?S_??CCSA_g???@D?????_?A??EO??GAOO_@C`???O?_CK_???_o_?@O??XA???AS???oE`?A?@?CAa?????C?G??i???C@qo?G?Og?_O?_?@???_G????o?A_@_?O?@??EcA???__?@GgO?O@oG?C?@??CIO?_??G??S?A?@oG_K?@C??@??QOA?C????AOo?p?G???oACAOAC@???OG??qC???C??AC_G?@??GCHG?AC@?_@O?CK?@?B???AI??OO_S_a_O??????AO?OHG?@?????_???EGOG??@?EF@?C?Pc?????C?W_PA?O@?_?@A@??OD_C?@?@?A??CC?_?i@?K?_O_CG??A?")
ce76.name(new = "ce76")
add_to_lists(ce76, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= maximum(matching_number(x), critical_independence_number(x))
ce77 = Graph("iF\ZccMAoW`Po_E_?qCP?Ag?OGGOGOS?GOH??oAAS??@CG?AA?@@_??_P??G?SO?AGA??M????SA????I?G?I???Oe?????OO???_S?A??A????ECA??C?A@??O??S?@????_@?_??S???O??")
ce77.name(new = "ce77")
add_to_lists(ce77, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= maximum(max_degree(x), radius(x)^card_periphery(x))
ce78 = Graph("G_aCp[")
ce78.name(new = "ce78")
add_to_lists(ce78, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= residue(x)^2
ce79 = Graph('J?B|~fpwsw_')
ce79.name(new = "ce79")
add_to_lists(ce79, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= 10^(card_center(x)*log(10)/log(sigma_2(x)))
ce80 = Graph('T?????????????????F~~~v}~|zn}ztn}zt^')
ce80.name(new = "ce80")
add_to_lists(ce80, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= diameter(x)^card_periphery(x)
ce81 = Graph('P?????????^~v~V~rzyZ~du{')
ce81.name(new = "ce81")
add_to_lists(ce81, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= radius(x)*residue(x) + girth(x)
ce82 = Graph('O????B~~^Zx^wnc~ENqxY')
ce82.name(new = "ce82")
add_to_lists(ce82, graph_objects, counter_examples, all_graphs)

"""
CE to independence_number(x) <= minimum(lovasz_theta(x), residue(x)^2)
    and
    <= minimum(annihilation_number(x), residue(x)^2)
    and
    <= minimum(fractional_alpha(x), residue(x)^2)
    and
    <= minimum(cvetkovic(x), residue(x)^2)
    and
    <= minimum(residue(x)^2, floor(lovasz_theta(x)))
    and
    <= minimum(size(x), residue(x)^2)
"""
ce83 = Graph('LEYSrG|mrQ[ppi')
ce83.name(new = "ce83")
add_to_lists(ce83, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= maximum(laplacian_energy(x), brinkmann_steffen(x)^2)
ce84 = Graph('~?@r?A??OA?C??????@?A????CC?_?A@????A?@???@?S?O????AO??????G???????C????????C?C???G?????_??????_?G?????O?A?_?O?O@??O???T@@??????O????C_???C?CO???@??@?@???_???O??O??A??O???O?A?OB?C?AD???C`?B?__?_????????Q?C??????????????_???C??_???A?gO??@C???C?EC?O??GG`?O?_?_??O????_?@?GA?_????????????G????????????????????AO_?C?????????P?IO??I??OC???O????A??AC@AO?o????????o@??O?aI?????????_A??O??G??o?????????_??@?????A?O?O?????G?????H???_????????A??a?O@O?_?D???????O@?????G???GG?CA??@?A@?A????GA?@???G??O??A??????AA???????O??_c??@???A?????_????@CG????????????A???A???????A?W???B????@?????HGO???????_@_?????C??????????_a??????_???????@G?@O?@@_??G@???????GG?O??A??????@????_??O_?_??CC?B???O??@????W??`AA????O??_?????????????????_???A??????@G??????I@C?G????????A@?@@?????C???p???????????????????G?_G????Z?A????_??????G????Q????@????????_@O????@???_QC?A??@???o???G???@???????O???CC??O?D?O?@C????@O?G?????A??@C???@????O?????????_??C??????_?@????O??????O?Y?C???_?????A??@OoG???????A???G??????CC??A?A?????????????????GA_???o???G??O??C???_@@??????@?????G??????????O???@O???????????A????S??_o????????A??B??????_??C????C?')
ce84.name(new = "ce84")
add_to_lists(ce84, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= girth(x)^ceil(laplacian_energy(x))
ce85 = Graph('bd_OPG_J_G?apBB?CPk@`X?hB_?QKEo_op`C?|Gc?K_?P@GCoGPTcGCh?CBIlqf_GQ]C_?@jlFP?KSEALWGi?bIS?PjO@?CCA?OG?')
ce85.name(new = "ce85")
add_to_lists(ce85, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= diameter(x)*residue(x) + different_degrees(x)
ce86 = Graph('SK|KWYc|^BJKlaCnMH^ECUoSC[{LHxfMG')
ce86.name(new = "ce86")
add_to_lists(ce86, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= maximum(max_common_neighbors(x), girth(x)^laplacian_energy(x))
ce87 = Graph('~?@iA?B@@b?[a??oHh_gC?@AGD?Wa???_E@_@o_AkGA@_o?_h??GG??cO??g?_?SD?d@IW?s?P_@@cSG?_B??d?CSI?OCQOH_?bo?CAaC???pGC@DO@?PHQOpO??A?_A@K[PC@?__???@OSCOGLO?oOAAA?IOX@??GC?O?P??oA_?KPIK?Q@A?sQC???LA???aQOC_AeG?Q?K_Oo?AB?OU?COD?VoQ?@D????A?_D?CAa?@@G?C??CGHcCA_cB?@c@_O?H??_@?@OWGGCo??AGC??AQ?QOc???Ow_?C[?O@@G_QH?H?O???_I@@PO????FAGk??C?ka@D@I?P?CooC@_O@?agAE??CpG?AA_`OO??_?Q?AiOQEK?GhB@CAOG?G?CC??C@O@GdC__?OIBKO?aOD_?OG???GACH@?b?@?B_???WPA?@_?o?XQQ?ZI_@?O_o_?@O??EDGOBEA??_aOSQsCO@?_DD`O??D?JaoP?G?AOQOCAS?k??S?c@?XW?QCO??_OAGOWc__G?_??G??L@OP?b?O?GCCMAH????????@@?A?C@oDaGG?Wk@H@OM?_A?IOu`SG?E@??W?I@EQA@@_@Wa?@?_??C??AAAiGQG@@?`@oA?_??OgC?K_G??G`?@S@B?A?HWc?HG??`gO???A?W?A?O?MpS??D?GS?GDC_??I@??IPAOdk`?CG??A?pPAgIDlCYCTSDgg?@FW?DI?O_OW?_S??AAQB_OOCF????XS_?@l_kAw__Ea?O?C_CGO??EG??WLb@_H??OCaAET@S?@?I???_??LaO_HCYG@G_G?_?_C???os?_G?OO@s_??_?_GGE`Os??_GCa?DWO?A@?@_CB`MOBCGIC???GKA_c?@BSh??@?RC[?eg?@hOC?_?BeGOaC?AWOSCm@G?A??A?G?Ga_')
ce87.name(new = "ce87")
add_to_lists(ce87, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= maximum(radius(x), max_degree(x))^2
ce88 = Graph('h@`CA???GH?AAG?OW@@????E???O?O???PO?O?_?G??`?O_???@??E?E??O??A?S@???S???????U?GAI???A?DA??C?C@??PA?A???_C_?H?AA??_C??DCO?C???_?AAG??@O?_?G')
ce88.name(new = "ce88")
add_to_lists(ce88, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= max_degree(x) + maximum(max_even_minus_even_horizontal(x), geometric_length_of_degree_sequence(x))
ce89 = Graph("_qH?S@??`??GG??O?_?C?_??@??@??G??C??_??C????O??G???@????O???A???@????C???C?????G????")
ce89.name(new = "ce89")
add_to_lists(ce89, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) >= floor(arccosh(lovasz_theta(x)))^2
ce90 = Graph("~?@Td|wi\\fbna~}wepkkbXcrW}\\~NvtLKpY\\J_Ub^~yM~^tHnM}jPffKkqnijvxD@xa{UOzzvr?L^PFi||yt@OQ\YU{Vh]tWzwzpj\\n|kR]`Y}RpCvxk{rEMRP\\}|}dNdNtbO~yrkgMxlOXr|FvQ{tvfKKnHrp^}jV\\B^n\\LvLZeyX}QSKN^sm~yl\\[NJZXqdk]O|^zHl~vC{w`Nsn}x]utqrJozKXV|eIUUPv~ydc}]xJNWZjW|lpYm}{Jf~JWMixb^t]e|S~B[vKc{K[Kjut~}Kj~iAl\\tVNgyZadvoA}rdTlr\\\\wNr^^kJzrp|qlVy]siKncI~`oNm|ul\\PxDRyzddDzrjUn~ciOgbR}p~Cz|~MlxYoEVnVuZkxJgvmtE]]}~PRp[He]oBQz]PVJ~gVnvSUR|QF|`lomFh[j|jIaS~vh~_rYiiK}FnEW}ovnntxtRFBakzvwn[biJhNvf|VDV?m~Y]ndmfJQ|M@QvnNf~MCyn~{HSU~fvEv~@}u|spOXzTVNY\\kjDNt\\zRMXxU|g|XrzFzDYiVvho}bQbyfI{{w[_~nrm}J~LhwH}TNmfM^}jqajl_ChY]M}unRK\\~ku")
ce90.name(new = "ce90")
add_to_lists(ce90, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= maximum(2*welsh_powell(x), max_even_minus_even_horizontal(x)^2)
ce91 = Graph("q?}x{k\FGNCRacDO`_gWKAq?ED?Qc?IS?Da?@_E?WO_@GOG@B@?Cc?@@OW???qO?@CC@?CA@C?E@?O?KK???E??GC?CO?CGGI??@?cGO??HG??@??G?SC???AGCO?KAG???@O_O???K?GG????WCG??C?C??_C????q??@D??AO???S????CA?a??A?G??IOO????B?A???_??")
ce91.name(new = "ce91")
add_to_lists(ce91, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) >= -max_common_neighbors(x) + min_degree(x) - 1
ce92 = Graph("qR}fexr{J\\innanomndYrzmy^p~Ri]c]lA{~jVurv]n~reCed~|j{TtvnMtB~nZFrz{wUnV^fzV\\rUlt|qvJubnFwWSxxzfZ}Btj`yV~rv\\nknwl~Z?T]{qwn~bFzh^\\{Ezv}p~I^RV|oXe~knL~x^nNtvYlrezLX^tj{S^Rflqqv]e|S^}vpbe~Ni]m]}zfbZolnPl{N~}]X?")
ce92.name(new = "ce92")
add_to_lists(ce92, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) >= diameter for regular graphs
# Ryan Pepper, June 2017
ce93 = Graph('_t???CA???_B?E?@??WAB?G??_GA?W?????@???W??B???A???BA??@__???G??@A???LA???AW???@_???G')
ce93.name(new = "ce93")
add_to_lists(ce93, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) >= diameter for regular graphs
# Ryan Pepper, June 2017
ce94 = Graph('Yv?GW?@?WB?A?A?@_?oGA?_KG????G??K??A???W??@???AO??BG??B?')
ce94.name(new = "ce94")
add_to_lists(ce94, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) >= -average_distance(x) + ceil(lovasz_theta(x))
ce95 = Graph('epCpih@K}gSGIZfc?Rkf{EWtVKJTmJtWYl_IoDOKOikwDSKtbH\\fi_g`affO\\|Agq`WcLoakSLNPaWZ@PQhCh?ylTR\\tIR?WfoJNYJ@B{GiOWMUZX_puFP?')
ce95.name(new = "ce95")
add_to_lists(ce95, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) >= ceil(1/2*cvetkovic(x))
# CE to independence_number(x) >= 1/2*cvetkovic(x)
ce96 = Graph('Gvz~r{')
ce96.name(new = "ce96")
add_to_lists(ce96, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) >= -max_common_neighbors(x) + min_degree(x)
ce97 = Graph('eLvnv~yv{yJ~rlB^Mn|v^nz~V]mwVji}^vZf{\\\\nqZLfVBze}y[ym|jevvt~NNucret|~ejj~rz}Q\\~^an}XzTV~t]a]v}nx\\u]n{}~ffqjn`~e}lZvV]t_')
ce97.name(new = "ce97")
add_to_lists(ce97, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) >= lovasz_theta(x)/radius(x)
ce100 = Graph('~?@A~~~~~~z~~~~~~~~^~~~~~z~~~~~~~~~~~~~~~v~~~v~~~~~~~~~~z~~~~~~~~^~~~~~~~~~}\~~}v~^~~}~~^~~~~~~~~~~~~^~~~~~~~~V~~~n~~n~~~~~~}~~|~}~~~~~~~~~~~~~~~~~~~~~vv~|~~~~~~~~~~~~~~~~~z~~w~~~~~~~~~~~~~~~~n~~~|~~~~~~~v~|~~~~~~~~~~}~|~r~V~~~n~~~~~~~~z~~}}~}~~~~vz~~z~~~z}~~~n~~~~~~~~~~~~n~~~~~~~z~~~~~~~~~~~~~~^~~~~~~~~~n~~]~~~~~n~~~}~~~~~~~~~~^~^~~~~}~~~~~~~~~~~z~~~~^~~~~~~w')
ce100.name(new = "ce100")
add_to_lists(ce100, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) >= matching_number(x) - order_automorphism_group(x) - 1
ce101 = Graph('I~~Lt~\Nw')
ce101.name(new = "ce101")
add_to_lists(ce101, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) >= card_positive_eigenvalues(x) - lovasz_theta(x)
ce102 = Graph('N^nN~~}Z~|}~~\~]zzw')
ce102.name(new = "ce102")
add_to_lists(ce102, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) >= card_negative_eigenvalues(x) - sigma_2(x)
ce103 = Graph('IOilnemjG')
ce103.name(new = "ce103")
add_to_lists(ce103, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) >= minimum(min_degree(x), floor(lovasz_theta(x)))
ce105 =  Graph('z@M@E?OYOSCPBTp?mOWaP_?W[OG[abE_?P[@?@REt?ggAAOH?N@?CATE\oE?WO@GOKu?LJ_??SDP@CIA?AFHCC?kZQMo@CkOGoiJSs`?g?oDJqC?S?qJSqA?GN]?OPd?cGHE?AOpE_c_O@kC_?DF@HGgJ?ygAACdcCMPA[d`SHE@?PqRE?CO_?CWO?H_b_EBoOKI@CWAadQ?eOO?oT_@STAWCCCMOK?A@?TsBoJa@?PGQ_CiKPC_@_iE_hL@ACAIJQDgOSG?G[cG_D[A_CbDKO[goBH_?S?')
ce105.name(new = "ce105")
add_to_lists(ce105, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= minimum(floor(lovasz_theta(x)), 10^laplacian_energy(x))
ce106 = Graph("k??????????????????????????????????????F~~~z~~~\~~~~{~^|nvny\^~~Njj~~zo}^yB|z~}h|Vv\Jft]~RlhV~tZMC^~fpvBynshVa~yw~@Tv\IVaJ}tvA\erD|Xx_rijkiIxLx}GE\pZ{yIwW?vV}K")
ce106.name(new = "ce106")
add_to_lists(ce106, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= max_even_minus_even_horizontal(x) + sinh(max_degree(x) - 1)
ce107 = Graph("~?@S???CO@_??C??????O?D????A?@?O?_????????????_???__?_O???????G??????A@????C????????????H???G??????_?G???????@????????OG?G_?C????C??K????A?_O???O?O????O?O????OOG?G??????C???????g???????C@???????A?OG_???CO?_?_??CC????@?G?????OO?@?C??????_?????O???A???????O?i??????????O?C???O????G??@??A?G??????A???_??A????OC?G???C?@?O???G??A?O???G?A??????C??C???@??????A??@????O???@??G?_?_???_???@G@???@???G??_?C????A??A?@AA????E????G?????????@O??@A????A????@?G?????????@@??_C???????G??@?@_???????@?????K?_???GO????_???O?C?C?@G??????????CG??@?G??????CQ??????G???????C__??C???_A?????????A_??o????_?@????")
ce107.name(new = "ce107")
add_to_lists(ce107, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= 10^cosh(log(residue(x)) - 1)
ce108 = Graph("U??????g}yRozOzw\wBn?zoBv_FN?Bn?B|??vO??")
ce108.name(new = "ce108")
add_to_lists(ce108, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) >= floor(caro_wei(x))^min_common_neighbors(x)
ce109 = Graph("Z???????????????????????????????B~~~v~~z~~|~nvnnjz~~xfntuV~w")
ce109.name(new = "ce109")
add_to_lists(ce109, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) >= min(card_positive_eigenvalues(x), 2*card_zero_eigenvalues(x))
ce110 = Graph("GUxvuw")
ce110.name(new = "ce110")
add_to_lists(ce110, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) >= minimum(lovasz_theta(x), max_degree(x)/card_center(x))
ce112 = Graph("flthktLhme|L|L]efhg|LbstFhhFhgbssG|L`FhiC]ecG|LCG|LaC]ew`FhnCG|L{Obstw`Fhlw`Fhi{ObssnCG|Ldw`FhmVaC]eknCG|LknCG|L??????_?????G")
ce112.name(new = "ce112")
add_to_lists(ce112, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= maximum(e^average_distance(x), max_degree(x)^2)
ce114 = Graph("~?@W?g??????????????????_???O??A???@c???W?S?P????O?@?@??G??????A????????O?????????????G??_?O?@G??????a??????GO???O????????C????O?A?@?AO???G_?AAG_?????o???????????????I??DG????????G??A_????C??O?G??_?_?????_G???????`??@??????A??C????OA?BG???????C?_???????a?@??O??_G?????????@@???O??????C@???A?A??G????????????@???_?????G?@O??????E???????????????O???@?E?O????G????C????A@????????A?????A_?Q???????AOO?@_?@??????????OG????g?????_?CG?????AA???O?G?????????__???????@??A??Q?????GOG??C????@??CC???????@???????G?C@???A??_?????A?????_????AC??C?@??????_?C??A???G???A???OGA????????A????_????K??O??????????@?????a???_??O??`??????A?????G???@?OOC??C???_?????")
ce114.name(new = "ce114")
add_to_lists(ce114, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= 2*wilf(x)^2 + radius(x)
ce115 = Graph("z??C????????A????AA?_??G????????????C?????_??@?????????????G?@??C?????A??C?????C??????C?????????C??????????O?_?????????????G@??A?C???O?????AC?????????O???@???_?S??????I???C???O?G????????G??C?_????A_??CG?A?????????_????A????????????????G???C???????????CC?G???C?????_????????o?I???_???????")
ce115.name(new = "ce115")
add_to_lists(ce115, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= card_periphery(x) + floor(e^laplacian_energy(x))
ce116 = Graph("J?Bzvrw}Fo?")
ce116.name(new = "ce116")
add_to_lists(ce116, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= residue(x)^2/density(x)
ce117 = Graph("N???F~}~f{^_z_~_^o?")
ce117.name(new = "ce117")
add_to_lists(ce117, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= maximum(caro_wei(x), e^different_degrees(x))^2
ce118 = Graph("O????B~~v}^w~o~o^wF}?")
ce118.name(new = "ce118")
add_to_lists(ce118, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= 10^sinh(average_vertex_temperature(x) + card_center(x))
ce119 = Graph("\?@CoO_CDOC?G?_?O?CO?_?BO?C???O?Bw??k??D_??O???_???C???A???@????g????")
ce119.name(new = "ce119")
add_to_lists(ce119, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= maximum(girth(x), min_degree(x))^laplacian_energy(x)
ce120 = Graph("^????????????????????X~~Umf|mezrBcjoezwF^{_Un}?|w{?kQ[?@Z}?FNl??Etw??Oz??F~Z_??")
ce120.name(new = "ce120")
add_to_lists(ce120, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= (2*girth(x))^card_periphery(x)
ce121 = Graph("~?@HhCGGC@?G?_@?@??_?G?@??C??G??G??C??@???G???_??@???@????_???G???@????C????G????G????C????@?????G?????_????@?????@??????_?????G?????@??????C??????G??????G??????C??????@???????G???????_??????@???????@????????_???????G???????@????????C????????G????????G????????C????????@?????????G?????????_????????@?????????@??????????_?????????G?????????@??????????C??????????G??????????G??????????C??????????@???????????G???????????_??????????@???????????@")
ce121.name(new = "ce121")
add_to_lists(ce121, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= ceil(laplacian_energy(x)*max_degree(x))
ce122 = Graph("~?@B??????????????????????????????????????????????????????????????????????????????????B\jeRlsk[rya~Sdr[HeojcW}xwcX_TgTVa?UhcYC?BTaaSo?}YoCRo@ov@_b?EysEu}?Do`uhm?@Ebfkm??~AHGh_?Awyl{w??TrUZDg??AuxBGO??FJ{LUo??O_IF]g??EVmFSY???`YcP_????l|v`}???Bg_[ge???ERCdO[???APuTc^????NQUWCO???B[QrBO????BMP^A_????eo{v`_????NgB\A_????O[Q@HO????@V_oTC?????FCDLPo????AECLPI?????DqZqCE??????")
ce122.name(new = "ce122")
add_to_lists(ce122, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= minimum(cvetkovic(x), 10^laplacian_energy(x))
ce123 = Graph("`??????????????????????????????????????????^^}nx~~[~~x}^k~~F~v^o^~vm@^~~wF~v}_F~~^_B~~~W?")
ce123.name(new = "ce123")
add_to_lists(ce123, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= minimum(fractional_alpha(x), 10^laplacian_energy(x))
ce124 = Graph("Z????????????????????~~~~~v~vf|~b~~o~~sF~~_^~}?~|k?z~w?^~}??")
ce124.name(new = "ce124")
add_to_lists(ce124, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= ceil(e^caro_wei(x))
ce125 = Graph("L??F~z{~FwN_~?")
ce125.name(new = "ce125")
add_to_lists(ce125, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= ceil(e^inverse_degree(x))
ce126 = Graph("Ov~~~}Lvs]^~~~~t~~}yJ")
ce126.name(new = "ce126")
add_to_lists(ce126, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= minimum(annihilation_number(x), 10^laplacian_energy(x))
ce127 = Graph("[??????~~~^{~w~w^{F~?~WB~_F~?F~?B~_?~w?F~??^s??~w??~w??^{??F~???")
ce127.name(new = "ce127")
add_to_lists(ce127, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= (diameter(x) + 1)*residue(x)
ce128 = Graph("M????B~~v}^w~o~o?")
ce128.name(new = "ce128")
add_to_lists(ce128, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= floor(10^sqrt(average_distance(x)))
ce129 = Graph("X????????????????????????????F~~~~z~~~Z~~{n}|u~utn~")
ce129.name(new = "ce129")
add_to_lists(ce129, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= minimum(lovasz_theta(x), 10^laplacian_energy(x))
ce130 = Graph("]?????????????????F|~nzz~~f~~F|~B~~_~~WF~~?^~{?nng?~~w?^^{?F~n??~~w?@~~_??")
ce130.name(new = "ce130")
add_to_lists(ce130, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= (residue(x) + 1)*girth(x)
ce131 = Graph("W?????????^~~}~}^~F~o~}B~wF}oF~oB~w?~}?F~o?^~??")
ce131.name(new = "ce131")
add_to_lists(ce131, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) <= minimum(size(x), 10^laplacian_energy(x))
ce132 = Graph("b????????????????????~~|~~|}~x~n~vp~{Nz{^e~svN}Xc~zFKZ~uyX[z[J\\ti@M~mT@]}X{SzQZQQ^posGNlZg?R}lp`LX{L?")
ce132.name(new = "ce132")
add_to_lists(ce132, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) >= -brinkmann_steffen(x) - girth(x) + matching_number(x)
ce133 = Graph("~?@VNtyzkW^v~_?UMVuzfYa_~}|tWZ}v~~v}WgheAI[pTRjr\\Wv}y~VeypzF\\^zmq}C\\UFoD^XJFQwvU^AovUND~XnF]STE_EF~v^~~JF\\X{X\\VFoeMvmy~xzZv\\^{|n~v~~~^v}y~xz]Wihy_Lzmq]CxSawvUNFZr]zj~fl|vMy~wz^Z}q}MxvsZv\\^{|nnuMy~wr^\\vmy~xz^^z~\\^}|nny{qUMHRpZSSE_EC_A@VXJAShyGe|vV~NZz~i}vUNnZr~n]zj~fl|vr^u~p~~}~~z}v}N~~v~~nzNwzf^\\n~TuQodI]AHvuMy^wr^\\wkNz^w~|~^}~y}vUNNZr~n~v}y~|z^^|~~^zn~~n~~v~y}q]MxVtZn|y{qUMHRpZfx|WyhIbL|vap[~}~~~~~~~~~}|vV~NZz~fz~jzHWydNDmzvz~~~~~~~~~~~~c?AGgA]Kw_PC_?_???O?????V~v~~~^~~n~~}\\uRofIca@FHcd~z~N~~~~~~~}Zaw}?oscOOx?d}ptR|EZznDbxsj~z~~~~~~~~~}^cAAYgA^KwgPCc]~v^~~^~~n~~}znz\\w~|~^}~~~r|v}y~|z^^|~~~t^Vuyp|z]^|~~~dZz~\\^{|nn}~~~ynw")
ce133.name(new = "ce133")
add_to_lists(ce133, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) >= -max_common_neighbors(x) + min_degree(x) - 1
ce134 = Graph('rQjxS{d~tvtrkz~pIJZ{gflAUdYPfybdzswqqS~i`~EfCmswgpu[zfAxLl\TtJEFlilHnZicmo}ZYJjAluT]d|scS\LgJ[|cs~}TBXxNQnJxSm]}oSMt{\kxUl|UhPZHlz`smizxCPTiNL[Mv|kbKUI}^r}oiAdMLE\^rga{v]z@U]Hb}wupjLh`Yg|Rn|`b[iLNp}Oudo~r_`oFEjTzvw')
ce134.name(new = "ce134")
add_to_lists(ce134, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) >= floor(tan(brooks(x)))
ce135 = Graph('Nzx~VT}yzxNd^J^Jn~w')
ce135.name(new = "ce135")
add_to_lists(ce135, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) >= floor(lovasz_theta(x))*sin(gutman_energy(x))
paley_17 = graphs.PaleyGraph(17)
paley_17.name(new = "paley_17")
add_to_lists(paley_17, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) >= ceil(lovasz_theta(x)) - diameter(x)
# CE to independence_number(x) >= ceil(lovasz_theta(x)) - radius(x)
paley_37 = graphs.PaleyGraph(37)
paley_37.name(new = "paley_37")
add_to_lists(paley_37, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) >= minimum(floor(lovasz_theta(x)), tan(randic(x)))
paley_53e = graphs.PaleyGraph(53)
paley_53e.add_edge(52,53)
paley_53e.name(new = "paley_53e")
add_to_lists(paley_53e, graph_objects, counter_examples, all_graphs)

# CE to independence_number(x) >= cos(alon_spencer(x))*floor(lovasz_theta(x))
paley_149 = graphs.PaleyGraph(149)
paley_149.name(new = "paley_149")
add_to_lists(paley_149, graph_objects, counter_examples, all_graphs)

#a K5 with a pendant
# CE to dirac => regular or planar
k5pendant = Graph('E~}?')
k5pendant.name(new="k5pendant")
add_to_lists(k5pendant, graph_objects, counter_examples, all_graphs)

#alon_seymour graph:
# CE to the rank-coloring conjecture,
# 56-regular, vertex_trans, alpha=2, omega=22, chi=chi'=edge_connect=56
alon_seymour=Graph([[0..63], lambda x,y : operator.xor(x,y) not in (0,1,2,4,8,16,32,63)])
alon_seymour.name(new="alon_seymour")
add_to_lists(alon_seymour, problem_graphs, counter_examples, all_graphs)

edge_critical_5=graphs.CycleGraph(5)
edge_critical_5.add_edge(0,3)
edge_critical_5.add_edge(1,4)
edge_critical_5.name(new="edge_critical_5")
add_to_lists(edge_critical_5, graph_objects, chromatic_index_critical, all_graphs)

# CE to independence_number(x) >= min(e - n + 1, diameter(x))
heather = graphs.CompleteGraph(4)
heather.add_vertex()
heather.add_vertex()
heather.add_edge(0,4)
heather.add_edge(5,4)
heather.name(new="heather")
add_to_lists(heather, graph_objects, counter_examples, all_graphs)

#residue = alpha = 3
# CE to residue = alpha => is_ore
ryan3=graphs.CycleGraph(15)
for i in range(15):
    for j in [1,2,3]:
        ryan3.add_edge(i,(i+j)%15)
        ryan3.add_edge(i,(i-j)%15)
ryan3.name(new="ryan3")
add_to_lists(ryan3, graph_objects, counter_examples, all_graphs)

#sylvester graph: 3-reg, 3 bridges, no perfect matching (why Petersen theorem requires no more than 2 bridges)
sylvester = Graph('Olw?GCD@o??@?@?A_@o`A')
sylvester.name(new="sylvester")
add_to_lists(sylvester, graph_objects, all_graphs)

fork = graphs.PathGraph(4)
fork.add_vertex()
fork.add_edge(1,4)
fork.name(new="fork")
add_to_lists(fork, graph_objects, all_graphs)

# one of the 2 order 11 chromatic edge-critical graphs discovered by brinkmann and steffen
edge_critical_11_1 = graphs.CycleGraph(11)
edge_critical_11_1.add_edge(0,2)
edge_critical_11_1.add_edge(1,6)
edge_critical_11_1.add_edge(3,8)
edge_critical_11_1.add_edge(5,9)
edge_critical_11_1.name(new="edge_critical_11_1")
add_to_lists(edge_critical_11_1, graph_objects, chromatic_index_critical, all_graphs)

#one of the 2 order 11 chromatic edge-critical graphs discovered by brinkmann and steffen
edge_critical_11_2 = graphs.CycleGraph(11)
edge_critical_11_2.add_edge(0,2)
edge_critical_11_2.add_edge(3,7)
edge_critical_11_2.add_edge(6,10)
edge_critical_11_2.add_edge(4,9)
edge_critical_11_2.name(new="edge_critical_11_2")
add_to_lists(edge_critical_11_2, graph_objects, chromatic_index_critical, all_graphs)

# chromatic_index_critical but not overfull
pete_minus=graphs.PetersenGraph()
pete_minus.delete_vertex(9)
pete_minus.name(new="pete_minus")
add_to_lists(pete_minus, graph_objects, chromatic_index_critical, all_graphs)

"""
The Haemers graph was considered by Haemers who showed that alpha(G)=theta(G)<vartheta(G).
The graph is a 108-regular graph on 220 vertices. The vertices correspond to the 3-element
subsets of {1,...,12} and two such vertices are adjacent whenever the subsets
intersect in exactly one element.

    sage: haemers
    haemers: Graph on 220 vertices
    sage: haemers.is_regular()
    True
    sage: max(haemers.degree())
    108
"""
haemers = Graph([Subsets(12,3), lambda s1,s2: len(s1.intersection(s2))==1])
haemers.relabel()
haemers.name(new="haemers")
add_to_lists(haemers, problem_graphs, all_graphs)

"""
The Pepper residue graph was described by Ryan Pepper in personal communication.
It is a graph which demonstrates that the residue is not monotone. The graph is
constructed by taking the complete graph on 3 vertices and attaching a pendant
vertex to each of its vertices, then taking two copies of this graph, adding a
vertex and connecting it to all the pendant vertices. This vertex has degree
sequence [6, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2] which gives residue equal to 4.
By removing the central vertex with degree 6, you get a graph with degree
sequence [3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1] which has residue equal to 5.

    sage: pepper_residue_graph
    pepper_residue_graph: Graph on 13 vertices
    sage: sorted(pepper_residue_graph.degree(), reverse=True)
    [6, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2]
    sage: residue(pepper_residue_graph)
    4
    sage: residue(pepper_residue_graph.subgraph(vertex_property=lambda v:pepper_residue_graph.degree(v)<6))
    5
"""
pepper_residue_graph = graphs.CompleteGraph(3)
pepper_residue_graph.add_edges([(i,i+3) for i in range(3)])
pepper_residue_graph = pepper_residue_graph.disjoint_union(pepper_residue_graph)
pepper_residue_graph.add_edges([(0,v) for v in pepper_residue_graph.vertices() if pepper_residue_graph.degree(v)==1])
pepper_residue_graph.relabel()
pepper_residue_graph.name(new="pepper_residue_graph")
add_to_lists(pepper_residue_graph, graph_objects, all_graphs)

"""
The Barrus graph was suggested by Mike Barrus in "Havel-Hakimi residues of Unigraphs" (2012) as an example of a graph whose residue (2) is
less than the independence number of any realization of the degree sequence. The degree sequence is [4^8,2].
The realization is the one given by reversing the Havel-Hakimi process.

    sage: barrus_graph
    barrus_graph: Graph on 9 vertices
    sage: residue(barrus_graph)
    2
    sage: independence_number(barrus_graph)
    3
"""
barrus_graph = Graph('HxNEG{W')
barrus_graph.name(new = "barrus_graph")
add_to_lists(barrus_graph, graph_objects, all_graphs)

# CE to (is_split)->((is_eulerian)->(is_regular))
# split graph from k4 and e2 that is eulerian but not regular
# isomorphic to c6 with a k4 subgraph
# eulerain, diameter = 3, radius=2, hamiltonian

k4e2split = graphs.CompleteGraph(4)
k4e2split.add_vertices([4,5])
k4e2split.add_edge(4,0)
k4e2split.add_edge(4,1)
k4e2split.add_edge(5,2)
k4e2split.add_edge(5,3)
k4e2split.name(new = "k4e2split")
add_to_lists(k4e2split, graph_objects, counter_examples, all_graphs)

# CE to (has_residue_equals_alpha)->((is_eulerian)->(alpha_leq_order_over_two))
triangle_star = Graph("H}qdB@_")
triangle_star.name(new = "triangle_star")
add_to_lists(triangle_star, graph_objects, counter_examples, all_graphs)

#flower with n petals
def flower(n):
    g = graphs.StarGraph(2*n)
    for x in range(n):
        v = 2*x+1
        g.add_edge(v,v+1)
    return g

flower_with_3_petals = flower(3)
flower_with_3_petals.name(new = "flower_with_3_petals")
add_to_lists(flower_with_3_petals, graph_objects, all_graphs)

flower_with_4_petals = flower(4)
flower_with_4_petals.name(new = "flower_with_4_petals")
add_to_lists(flower_with_4_petals, graph_objects, all_graphs)

# Gallai Tree graph
gallai_tree = Graph("`hCKGC@?G@?K?@?@_?w?@??C??G??G??c??o???G??@_??F???N????_???G???B????C????W????G????G????C")
gallai_tree.name(new = "gallai_tree")
add_to_lists(gallai_tree, graph_objects, all_graphs)

# Trigonal Antiprism w/ capped top face
trig_antiprism_capped = Graph("Iw?EthkF?")
trig_antiprism_capped.name(new = "trig_antiprism_capped")
add_to_lists(trig_antiprism_capped, graph_objects, all_graphs)

"""
From Willis's thesis, page 4
Alpha = Fractional Alpha = 4

    sage: independence_number(willis_page4)
    4
    sage: fractional_alpha(willis_page4)
    4
"""
willis_page4 = Graph("GlCKIS")
willis_page4.name(new = "willis_page4")
add_to_lists(willis_page4, graph_objects, all_graphs)

"""
From Willis's thesis, page 7

    sage: willis_page7.radius()
    2
    sage: average_distance(willis_page7)
    1.467
"""
willis_page7 = Graph("ELRW")
willis_page7.name(new = "willis_page7")
add_to_lists(willis_page7, graph_objects, all_graphs)

"""
From Willis's thesis, page 13, Fig. 2.7

    sage: independence_number(willis_page13_fig27)
    4
    sage: willis_page13_fig27.order()
    7
    sage: willis_page13_fig27.size()
    15
"""
willis_page13_fig27 = Graph("Fs\zw")
willis_page13_fig27.name(new = "willis_page13_fig27")
add_to_lists(willis_page13_fig27, graph_objects, all_graphs)

"""
From Willis's thesis, page 10, Figure 2.2
Graph for which the Cvetkovic bound is the best upper bound present in the thesis

    sage: independence_number(willis_page10_fig23)
    4
    sage: willis_page10_fig23.order()
    10
    sage: willis_page10_fig23.size()
    15
    sage: max_degree(willis_page10_fig23)
    3
    sage: min_degree(willis_page10_fig23)
    3
"""
willis_page10_fig23 = Graph("G|eKHw")
willis_page10_fig23.name(new = "willis_page10_fig23")
add_to_lists(willis_page10_fig23, graph_objects, all_graphs)

"""
From Willis's thesis, page 10, Figure 2.4
Graph for which the Cvetkovic bound is the best upper bound present in the thesis

    sage: independence_number(willis_page10_fig24)
    9
    sage: willis_page10_fig24.order()
    24
    sage: willis_page10_fig24.size()
    36
    sage: max_degree(willis_page10_fig24)
    3
    sage: min_degree(willis_page10_fig24)
    3
"""
willis_page10_fig24 = Graph("WvOGWK@?G@_B???@_?O?F?????G??W?@K_?????G??@_?@B")
willis_page10_fig24.name(new = "willis_page10_fig24")
add_to_lists(willis_page10_fig24, graph_objects, all_graphs)

"""
From Willis's thesis, page 13, Figure 2.6
Graph for which the fractional independence bound is the best upper bound present in the thesis

    sage: independence_number(willis_page13_fig26)
    3
    sage: willis_page13_fig26.order()
    7
    sage: willis_page13_fig26.size()
    12
    sage: max_degree(willis_page13_fig26)
    4
    sage: min_degree(willis_page13_fig26)
    3
"""
willis_page13_fig26 = Graph("FstpW")
willis_page13_fig26.name(new = "willis_page13_fig26")
add_to_lists(willis_page13_fig26, graph_objects, all_graphs)

"""
From Willis's thesis, page 21, Figure 3.1
Graph for which n/chi is the best lower bound present in the thesis

    sage: independence_number(willis_page21)
    4
    sage: willis_page21.order()
    12
    sage: willis_page21.size()
    20
    sage: max_degree(willis_page21)
    4
    sage: chromatic_num(willis_page21)
    3
"""
willis_page21 = Graph("KoD?Xb?@HBBB")
willis_page21.name(new = "willis_page21")
add_to_lists(willis_page21, graph_objects, all_graphs)

"""
From Willis's thesis, page 25, Figure 3.2
Graph for which residue is the best lower bound present in the thesis

    sage: independence_number(willis_page25_fig32)
    3
    sage: willis_page25_fig32.order()
    8
    sage: willis_page25_fig32.size()
    15
    sage: max_degree(willis_page25_fig32)
    6
    sage: chromatic_num(willis_page25_fig32)
    4
"""
willis_page25_fig32 = Graph("G@N@~w")
willis_page25_fig32.name(new = "willis_page25_fig32")
add_to_lists(willis_page25_fig32, graph_objects, all_graphs)

"""
From Willis's thesis, page 25, Figure 3.3
Graph for which residue is the best lower bound present in the thesis

    sage: independence_number(willis_page25_fig33)
    4
    sage: willis_page25_fig33.order()
    14
    sage: willis_page25_fig33.size()
    28
    sage: max_degree(willis_page25_fig33)
    4
    sage: chromatic_num(willis_page25_fig33)
    4
"""
willis_page25_fig33 = Graph("Mts?GKE@QDCIQIKD?")
willis_page25_fig33.name(new = "willis_page25_fig33")
add_to_lists(willis_page25_fig33, graph_objects, all_graphs)

# The Lemke Graph
lemke = Graph("G_?ztw")
lemke.name(new = "Lemke")
add_to_lists(lemke, graph_objects, all_graphs)

"""
From Willis's thesis, page 29, Figure 3.6
Graph for which the Harant Bound is the best lower bound present in the thesis

    sage: independence_number(willis_page29)
    4
    sage: willis_page29.order()
    14
    sage: willis_page29.size()
    28
    sage: max_degree(willis_page29)
    4
    sage: chromatic_num(willis_page29)
    4
"""
willis_page29 = Graph("[HCGGC@?G?_@?@_?_?M?@o??_?G_?GO?CC?@?_?GA??_C?@?C?@?A??_?_?G?D?@")
willis_page29.name(new = "willis_page29")
add_to_lists(willis_page29, graph_objects, all_graphs)

"""
From Willis's thesis, page 35, Figure 5.1
A graph where none of the upper bounds in the thesis give the exact value for alpha

    sage: independence_number(willis_page35_fig51)
    2
    sage: willis_page35_fig51.order()
    10
"""
willis_page35_fig51 = Graph("I~rH`cNBw")
willis_page35_fig51.name(new = "willis_page35_fig51")
add_to_lists(willis_page35_fig51, graph_objects, all_graphs)

"""
From Willis's thesis, page 35, Figure 5.2
A graph where none of the upper bounds in the thesis give the exact value for alpha

    sage: independence_number(willis_page35_fig52)
    2
    sage: willis_page35_fig52.order()
    10
"""
willis_page35_fig52 = Graph("I~zLa[vFw")
willis_page35_fig52.name(new = "willis_page35_fig52")
add_to_lists(willis_page35_fig52, graph_objects, all_graphs)

"""
From Willis's thesis, page 36, Figure 5.3
A graph where none of the upper bounds in the thesis give the exact value for alpha

    sage: independence_number(willis_page36_fig53)
    4
    sage: willis_page36_fig53.order()
    11
"""
willis_page36_fig53 = Graph("JscOXHbWqw?")
willis_page36_fig53.name(new = "willis_page36_fig53")
add_to_lists(willis_page36_fig53, graph_objects, all_graphs)

"""
From Willis's thesis, page 36, Figure 5.4
A graph where none of the upper bounds in the thesis give the exact value for alpha

    sage: independence_number(willis_page36_fig54)
    2
    sage: willis_page36_fig54.order()
    9
    sage: willis_page36_fig54.size()
    24
"""
willis_page36_fig54 = Graph("H~`HW~~")
willis_page36_fig54.name(new = "willis_page36_fig54")
add_to_lists(willis_page36_fig54, graph_objects, all_graphs)

"""
From Willis's thesis, page 36, Figure 5.5
A graph where none of the lower bounds in the thesis give the exact value for alpha

    sage: independence_number(willis_page36_fig55)
    3
    sage: willis_page36_fig54.order()
    7
    sage: willis_page36_fig54.size()
    13
"""
willis_page36_fig55 = Graph("F@^vo")
willis_page36_fig55.name(new = "willis_page36_fig55")
add_to_lists(willis_page36_fig55, graph_objects, all_graphs)

"""
From Willis's thesis, page 37, Figure 5.6
A graph where none of the lower bounds in the thesis give the exact value for alpha

    sage: independence_number(willis_page37_fig56)
    3
    sage: willis_page37_fig56.order()
    7
    sage: willis_page37_fig56.size()
    15
"""
willis_page37_fig56 = Graph("Fimzw")
willis_page37_fig56.name(new = "willis_page37_fig56")
add_to_lists(willis_page37_fig56, graph_objects, all_graphs)

"""
From Willis's thesis, page 37, Figure 5.8
A graph where none of the lower bounds in the thesis give the exact value for alpha

    sage: independence_number(willis_page37_fig58)
    3
    sage: willis_page37_fig58.order()
    9
    sage: willis_page37_fig58.size()
    16
"""
willis_page37_fig58 = Graph("H?iYbC~")
willis_page37_fig58.name(new = "willis_page37_fig58")
add_to_lists(willis_page37_fig58, graph_objects, all_graphs)

"""
From Willis's thesis, page 39, Figure 5.10
A graph where none of the upper or lower bounds in the thesis give the exact value for alpha

    sage: independence_number(willis_page39_fig510)
    5
    sage: willis_page39_fig510.order()
    12
    sage: willis_page39_fig510.size()
    18
"""
willis_page39_fig510 = Graph("Kt?GOKEOGal?")
willis_page39_fig510.name(new = "willis_page39_fig510")
add_to_lists(willis_page39_fig510, graph_objects, all_graphs)

"""
From Willis's thesis, page 40, Figure 5.12
A graph where none of the upper or lower bounds in the thesis give the exact value for alpha

    sage: independence_number(willis_page40_fig512)
    6
    sage: willis_page40_fig512.order()
    14
    sage: willis_page40_fig512.size()
    21
"""
willis_page40_fig512 = Graph("Ms???\?OGdAQJ?J??")
willis_page40_fig512.name(new = "willis_page40_fig512")
add_to_lists(willis_page40_fig512, graph_objects, all_graphs)

"""
From Willis's thesis, page 41, Figure 5.14
A graph where none of the upper or lower bounds in the thesis give the exact value for alpha

    sage: independence_number(willis_page41_fig514)
    5
    sage: willis_page41_fig514.order()
    12
    sage: willis_page41_fig514.size()
    18
"""
willis_page41_fig514 = Graph("Kt?GGGBQGeL?")
willis_page41_fig514.name(new = "willis_page41_fig514")
add_to_lists(willis_page41_fig514, graph_objects, all_graphs)

"""
From Willis's thesis, page 41, Figure 5.15
A graph where none of the upper or lower bounds in the thesis give the exact value for alpha

    sage: independence_number(willis_page41_fig515)
    4
    sage: willis_page41_fig515.order()
    11
    sage: willis_page41_fig515.size()
    22
"""
willis_page41_fig515 = Graph("JskIIDBLPh?")
willis_page41_fig515.name(new = "willis_page41_fig515")
add_to_lists(willis_page41_fig515, graph_objects, all_graphs)

"""
From Elphick-Wocjan page 8
"""
elphick_wocjan_page8 = Graph("F?Azw")
elphick_wocjan_page8.name(new = "Elphick-Wocjan p.8")
add_to_lists(elphick_wocjan_page8, graph_objects, all_graphs)

"""
From Elphick-Wocjan page 9
"""
elphick_wocjan_page9 = Graph("FqhXw")
elphick_wocjan_page9.name(new = "Elphick-Wocjan p.9")
add_to_lists(elphick_wocjan_page9, graph_objects, all_graphs)

"""
An odd wheel with 8 vertices
p.175
Rebennack, Steffen, Gerhard Reinelt, and Panos M. Pardalos. "A tutorial on branch and cut algorithms for the maximum stable set problem." International Transactions in Operational Research 19.1-2 (2012): 161-199.

    sage: odd_wheel_8.order()
    8
    sage: odd_wheel_8.size()
    14
"""
odd_wheel_8 = Graph("G|eKMC")
odd_wheel_8.name(new = "odd_wheel_8")
add_to_lists(odd_wheel_8, graph_objects, all_graphs)

"""
A facet-inducing graph
p.176
Rebennack, Steffen, Gerhard Reinelt, and Panos M. Pardalos. "A tutorial on branch and cut algorithms for the maximum stable set problem." International Transactions in Operational Research 19.1-2 (2012): 161-199.

    sage: facet_inducing.order()
    8
    sage: facet_inducing.size()
    11
"""
facet_inducing = Graph("G@hicc")
facet_inducing.name(new = "facet_inducing")
add_to_lists(facet_inducing, graph_objects, all_graphs)

"""
Double Fork
p.185
Rebennack, Steffen, Gerhard Reinelt, and Panos M. Pardalos. "A tutorial on branch and cut algorithms for the maximum stable set problem." International Transactions in Operational Research 19.1-2 (2012): 161-199.

    sage: double_fork.order()
    6
    sage: double_fork.size()
    5
"""
double_fork = Graph("E?dg")
double_fork.name(new = "double_fork")
add_to_lists(double_fork, graph_objects, all_graphs)

"""
Golomb Graph
Appears in THE FRACTIONAL CHROMATIC NUMBER OF THE PLANE by Cranston and Rabern
"""
golomb = Graph("I?C]dPcww")
golomb.name(new = "Golomb Graph")
add_to_lists(golomb, graph_objects, all_graphs)

"""
Fig. 1, G1 p. 454 of
Steinberg’s Conjecture is false by
 Vincent Cohen-Addad, Michael Hebdige, Daniel Král,
 Zhentao Li, Esteban Salgado
"""
steinberg_ce_g1 = Graph("N?CWGOOOH@OO_POdCHO")
steinberg_ce_g1.name(new = "steinberg_ce_g1")
add_to_lists(steinberg_ce_g1, graph_objects, counter_examples, all_graphs)

"""
4-pan from p. 1691 of
Graphs with the Strong Havel–Hakimi Property by Michael D. Barrus and Grant Molnar
"""
four_pan = Graph("DBw")
four_pan.name(new = "4-pan")
add_to_lists(four_pan, graph_objects, all_graphs)

"""
kite from p. 1691 of
Graphs with the Strong Havel–Hakimi Property by Michael D. Barrus and Grant Molnar
NOTE: Called kite in the paper, but will be called kite_with_tail here because we already have a kite
"""
kite_with_tail = Graph("DJk")
kite_with_tail.name(new = "kite with tail")
add_to_lists(kite_with_tail, graph_objects, all_graphs)

"""
Chartrand Fig 1.1

    sage: chartrand_11.order()
    8
    sage: chartrand_11.size()
    15
"""
chartrand_11 = Graph("G`RHx{")
chartrand_11.name(new = "chartrand fig 1.1")
add_to_lists(chartrand_11, graph_objects, all_graphs)

"""
Chartrand Fig 1.2

    sage: chartrand_12.order()
    8
    sage: chartrand_12.size()
    9
"""
chartrand_12 = Graph("G??|Qo")
chartrand_12.name(new = "chartrand fig 1.2")
add_to_lists(chartrand_12, graph_objects, all_graphs)

"""
Chartrand Fig 1.3

    sage: chartrand_13.order()
    8
    sage: chartrand_13.size()
    10
"""
chartrand_13 = Graph("G`o_g[")
chartrand_13.name(new = "chartrand fig 1.3")
add_to_lists(chartrand_13, graph_objects, all_graphs)

"""
Chartrand Fig 1.8 - G

    sage: chartrand_18_g.order()
    7
    sage: chartrand_18_g.size()
    8
"""
chartrand_18_g = Graph("Fo@Xo")
chartrand_18_g.name(new = "chartrand fig 1.8 - G")
add_to_lists(chartrand_18_g, graph_objects, all_graphs)

"""
Chartrand Fig 1.8 - F1

    sage: chartrand_18_f1.order()
    7
    sage: chartrand_18_f1.size()
    10
"""
chartrand_18_f1 = Graph("F@J]o")
chartrand_18_f1.name(new = "chartrand fig 1.8 - F1")
add_to_lists(chartrand_18_f1, graph_objects, all_graphs)

"""
Chartrand Fig 1.8 - F2

    sage: chartrand_18_f2.order()
    7
    sage: chartrand_18_f2.size()
    10
"""
chartrand_18_f2 = Graph("F?NVo")
chartrand_18_f2.name(new = "chartrand fig 1.8 - F2")
add_to_lists(chartrand_18_f2, graph_objects, all_graphs)

"""
Chartrand Fig 1.10 - H1
CE to independence_number(x) <= maximum(girth(x), card_center(x) + card_periphery(x))

    sage: chartrand_110_h1.order()
    7
    sage: chartrand_110_h1.size()
    7
"""
chartrand_110_h1 = Graph("F@@Kw")
chartrand_110_h1.name(new = "chartrand fig 1.10 - H1")
add_to_lists(chartrand_110_h1, graph_objects, counter_examples, all_graphs)

"""
Chartrand Fig 1.10 - H2

    sage: chartrand_110_h2.order()
    7
    sage: chartrand_110_h2.size()
    7
"""
chartrand_110_h2 = Graph("F?C^G")
chartrand_110_h2.name(new = "chartrand fig 1.10 - H2")
add_to_lists(chartrand_110_h2, graph_objects, all_graphs)

"""
Chartrand Fig 1.10 - H3

    sage: chartrand_110_h3.order()
    9
    sage: chartrand_110_h3.size()
    8
"""
chartrand_110_h3 = Graph("H???]Os")
chartrand_110_h3.name(new = "chartrand fig 1.10 - H3")
add_to_lists(chartrand_110_h3, graph_objects, all_graphs)

"""
Chartrand Fig 1.10 - H4

    sage: chartrand_110_h4.order()
    9
    sage: chartrand_110_h4.size()
    8
"""
chartrand_110_h4 = Graph("H?C?JEK")
chartrand_110_h4.name(new = "chartrand fig 1.10 - H4")
add_to_lists(chartrand_110_h4, graph_objects, all_graphs)

# From their Mathing Theory book
lovasz_plummer = Graph('iOQBC__???G_?OCG@??C???_C?G?@_?__??????_??@???E?C?C?A?A??CC????O???@??G??_????o?????????A?????O????B?????A????CO?????C?????@??????C??????W?????CO')
lovasz_plummer.name(new = "lovasz_plummer graph")
add_to_lists(lovasz_plummer, graph_objects, all_graphs)

# Jorgenson graphs from: Jørgensen, Leif K. "Diameters of cubic graphs." Discrete applied mathematics 37 (1992): 347-351.
jorgenson_1 = Graph('Ss??GOG?OC?_?I?DGCa?oDO?i??_C@?AC')
jorgenson_1.name(new = "jorgenson_1")
add_to_lists(jorgenson_1, graph_objects, all_graphs)

jorgenson_2 = Graph('GtP@Ww')
jorgenson_2.name(new = "jorgenson_2")
add_to_lists(jorgenson_2, graph_objects, all_graphs)

# CGT graphs come from
#     Examples and Counter Examples in Graph Theory by Michael Capobianco and John C. Molluzzo

cgt_5 = Graph("J??GkPPgbG?")
cgt_5.name(new = "CGT Page 5")
add_to_lists(cgt_5, graph_objects, all_graphs)

cgt_10_left = Graph("E`~o")
cgt_10_left.name(new = "CGT Page 10, left")
add_to_lists(cgt_10_left, graph_objects, all_graphs)

cgt_10_right = Graph("EFxw")
cgt_10_right.name(new = "CGT Page 10, right")
add_to_lists(cgt_10_right, graph_objects, all_graphs)

cgt_11_1221 = Graph("H`_YPN~")
cgt_11_1221.name(new = "CGT Page 11, Figure 1.22.1")
add_to_lists(cgt_11_1221, graph_objects, all_graphs)

cgt_13_bottom = Graph("Iv?GOKFY?")
cgt_13_bottom.name(new = "CGT Page 13, Bottom")
add_to_lists(cgt_13_bottom, graph_objects, all_graphs)

cgt_15_left = Graph("FD^vW")
cgt_15_left.name(new = "CGT Page 15, left")
add_to_lists(cgt_15_left, graph_objects, all_graphs)

# Sabidussi Graphs from Sabidussi, Gert. "The centrality index of a graph." Psychometrika 31.4 (1966): 581-603.

sabidussi_2 = Graph('Ls_aGcKSpiDCQH')
sabidussi_2.name(new = "sabidussi_2")
add_to_lists(sabidussi_2, graph_objects, all_graphs)

sabidussi_3 = Graph('H`?NeW{')
sabidussi_3.name(new = "sabidussi_3")
add_to_lists(sabidussi_3, graph_objects, all_graphs)

sabidussi_4 = Graph('L?O_?C@?G_oOF?')
sabidussi_4.name(new = "sabidussi_4")
add_to_lists(sabidussi_4, graph_objects, all_graphs)

sabidussi_5 = Graph('L??GH?GOC@AWKC')
sabidussi_5.name(new = "sabidussi_5")
add_to_lists(sabidussi_5, graph_objects, all_graphs)

sabidussi_6 = Graph('L??G?COGQCOCN?')
sabidussi_6.name(new = "sabidussi_6")
add_to_lists(sabidussi_6, graph_objects, all_graphs)

barnette = Graph("Ss?GOCDA?@_I@??_C?q?QC?_O@@??I??S")
barnette.name(new = "Barnette-Bosak-Lederberg Graph")
add_to_lists(barnette, graph_objects, all_graphs)

# Maximally Irregular Graphs
mir_6 = MIR(6)
mir_6.name(new = "max_irregular_6")
add_to_lists(mir_6, graph_objects, all_graphs)

mir_7 = MIR(7)
mir_7.name(new = "max_irregular_7")
add_to_lists(mir_7, graph_objects, all_graphs)

# Radius Critical Ciliates
c4_1 = Ciliate(2, 3)
c4_1.name(new = "Ciliate 4, 1")
add_to_lists(c4_1, graph_objects, all_graphs)

c4_2 = Ciliate(2, 4)
c4_2.name(new = "Ciliate 4, 2")
add_to_lists(c4_2, graph_objects, all_graphs)

c6_1 = Ciliate(3, 4)
c6_1.name(new = "Ciliate 6, 1")
add_to_lists(c6_1, graph_objects, all_graphs)

# Antiholes
"""
An odd antihole with 7 vertices
p.175
Rebennack, Steffen, Gerhard Reinelt, and Panos M. Pardalos. "A tutorial on branch and cut algorithms for the maximum stable set problem." International Transactions in Operational Research 19.1-2 (2012): 161-199.

    sage: odd_antihole_7.order()
    7
    sage: odd_antihole_7.size()
    14
"""
antihole_7 = Antihole(7)
antihole_7.name(new = "Antihole 7")
add_to_lists(antihole_7, graph_objects, all_graphs)

antihole_8 = Antihole(8)
antihole_8.name(new = "Antihole 8")
add_to_lists(antihole_8, graph_objects, all_graphs)

"""
p.10
Barrus, Michael D. "On fractional realizations of graph degree sequences." arXiv preprint arXiv:1310.1112 (2013).
"""
fish = Graph("E@ro")
fish.name(new = "fish")
add_to_lists(fish, graph_objects, all_graphs)

"""
p.10
Barrus, Michael D. "On fractional realizations of graph degree sequences." arXiv preprint arXiv:1310.1112 (2013).
"""
fish_mod = Graph("EiKw")
fish_mod.name(new = "fish_mod")
add_to_lists(fish_mod, graph_objects, all_graphs)

"""
All of the graphs with the name barrus_[0-9]{6} comes from the appendix of
Barrus, Michael D. "On fractional realizations of graph degree sequences." arXiv preprint arXiv:1310.1112 (2013).
"""

barrus_322111a = Graph("EAIW")
barrus_322111a.name(new = "barrus_322111a")
add_to_lists(barrus_322111a, graph_objects, all_graphs)

barrus_322111b = Graph("E?NO")
barrus_322111b.name(new = "barrus_322111b")
add_to_lists(barrus_322111b, graph_objects, all_graphs)

barrus_322221b = Graph("ECXo")
barrus_322221b.name(new = "barrus_322221b")
add_to_lists(barrus_322221b, graph_objects, all_graphs)

barrus_322221c = Graph("EAN_")
barrus_322221c.name(new = "barrus_322221c")
add_to_lists(barrus_322221c, graph_objects, all_graphs)

barrus_332211c = Graph("E@hW")
barrus_332211c.name(new = "barrus_332211c")
add_to_lists(barrus_332211c, graph_objects, all_graphs)

barrus_332211d = Graph("E?lo")
barrus_332211d.name(new = "barrus_332211d")
add_to_lists(barrus_332211d, graph_objects, all_graphs)

barrus_332222a = Graph("E`ow")
barrus_332222a.name(new = "barrus_332222a")
add_to_lists(barrus_332222a, graph_objects, all_graphs)

barrus_332222b = Graph("E`dg")
barrus_332222b.name(new = "barrus_332222b")
add_to_lists(barrus_332222b, graph_objects, all_graphs)

barrus_332222c = Graph("EoSw")
barrus_332222c.name(new = "barrus_332222c")
add_to_lists(barrus_332222c, graph_objects, all_graphs)

barrus_332222d = Graph("E_lo")
barrus_332222d.name(new = "barrus_332222d")
add_to_lists(barrus_332222d, graph_objects, all_graphs)

barrus_333221a = Graph("E`LW")
barrus_333221a.name(new = "barrus_333221a")
add_to_lists(barrus_333221a, graph_objects, all_graphs)

barrus_333221b = Graph("EKSw")
barrus_333221b.name(new = "barrus_333221b")
add_to_lists(barrus_333221b, graph_objects, all_graphs)

barrus_333221c = Graph("EELg")
barrus_333221c.name(new = "barrus_333221c")
add_to_lists(barrus_333221c, graph_objects, all_graphs)

barrus_333221d = Graph("EC\o")
barrus_333221d.name(new = "barrus_333221d")
add_to_lists(barrus_333221d, graph_objects, all_graphs)

barrus_333311 = Graph("E@lo")
barrus_333311.name(new = "barrus_333311")
add_to_lists(barrus_333311, graph_objects, all_graphs)

barrus_333322a = Graph("ES\o")
barrus_333322a.name(new = "barrus_333322a")
add_to_lists(barrus_333322a, graph_objects, all_graphs)

barrus_333322c = Graph("ED^_")
barrus_333322c.name(new = "barrus_333322c")
add_to_lists(barrus_333322c, graph_objects, all_graphs)

barrus_422211a = Graph("EGEw")
barrus_422211a.name(new = "barrus_422211a")
add_to_lists(barrus_422211a, graph_objects, all_graphs)

barrus_422211b = Graph("E?No")
barrus_422211b.name(new = "barrus_422211b")
add_to_lists(barrus_422211b, graph_objects, all_graphs)

barrus_432221a = Graph("E@pw")
barrus_432221a.name(new = "barrus_432221a")
add_to_lists(barrus_432221a, graph_objects, all_graphs)

barrus_432221b = Graph("E_Lw")
barrus_432221b.name(new = "barrus_432221b")
add_to_lists(barrus_432221b, graph_objects, all_graphs)

barrus_432221c = Graph("EANg")
barrus_432221c.name(new = "barrus_432221c")
add_to_lists(barrus_432221c, graph_objects, all_graphs)

barrus_432221d = Graph("E?^o")
barrus_432221d.name(new = "barrus_432221d")
add_to_lists(barrus_432221d, graph_objects, all_graphs)

barrus_433222a = Graph("E@vo")
barrus_433222a.name(new = "barrus_433222a")
add_to_lists(barrus_433222a, graph_objects, all_graphs)

barrus_433222c = Graph("EPVW")
barrus_433222c.name(new = "barrus_433222c")
add_to_lists(barrus_433222c, graph_objects, all_graphs)

barrus_433222d = Graph("E`NW")
barrus_433222d.name(new = "barrus_433222d")
add_to_lists(barrus_433222d, graph_objects, all_graphs)

barrus_433321a = Graph("EIMw")
barrus_433321a.name(new = "barrus_433321a")
add_to_lists(barrus_433321a, graph_objects, all_graphs)

barrus_433321b = Graph("E@^o")
barrus_433321b.name(new = "barrus_433321b")
add_to_lists(barrus_433321b, graph_objects, all_graphs)

barrus_433321c = Graph("EPTw")
barrus_433321c.name(new = "barrus_433321c")
add_to_lists(barrus_433321c, graph_objects, all_graphs)

barrus_433332a = Graph("EBzo")
barrus_433332a.name(new = "barrus_433332a")
add_to_lists(barrus_433332a, graph_objects, all_graphs)

barrus_433332b = Graph("E`^o")
barrus_433332b.name(new = "barrus_433332b")
add_to_lists(barrus_433332b, graph_objects, all_graphs)

barrus_433332c = Graph("EqLw")
barrus_433332c.name(new = "barrus_433332c")
add_to_lists(barrus_433332c, graph_objects, all_graphs)

barrus_442222a = Graph("E?~o")
barrus_442222a.name(new = "barrus_442222a")
add_to_lists(barrus_442222a, graph_objects, all_graphs)

barrus_442222b = Graph("E_lw")
barrus_442222b.name(new = "barrus_442222b")
add_to_lists(barrus_442222b, graph_objects, all_graphs)

barrus_443322a = Graph("E@~o")
barrus_443322a.name(new = "barrus_443322a")
add_to_lists(barrus_443322a, graph_objects, all_graphs)

barrus_443322b = Graph("EHuw")
barrus_443322b.name(new = "barrus_443322b")
add_to_lists(barrus_443322b, graph_objects, all_graphs)

barrus_443322c = Graph("EImw")
barrus_443322c.name(new = "barrus_443322c")
add_to_lists(barrus_443322c, graph_objects, all_graphs)

barrus_443322d = Graph("EQlw")
barrus_443322d.name(new = "barrus_443322d")
add_to_lists(barrus_443322d, graph_objects, all_graphs)

barrus_443322e = Graph("E`lw")
barrus_443322e.name(new = "barrus_443322e")
add_to_lists(barrus_443322e, graph_objects, all_graphs)

barrus_443331a = Graph("EBxw")
barrus_443331a.name(new = "barrus_443331a")
add_to_lists(barrus_443331a, graph_objects, all_graphs)

barrus_443331b = Graph("E`\w")
barrus_443331b.name(new = "barrus_443331b")
add_to_lists(barrus_443331b, graph_objects, all_graphs)

barrus_443333a = Graph("Es\w")
barrus_443333a.name(new = "barrus_443333a")
add_to_lists(barrus_443333a, graph_objects, all_graphs)

barrus_443333c = Graph("Eqlw")
barrus_443333c.name(new = "barrus_443333c")
add_to_lists(barrus_443333c, graph_objects, all_graphs)

barrus_444332b = Graph("Ed\w")
barrus_444332b.name(new = "barrus_444332b")
add_to_lists(barrus_444332b, graph_objects, all_graphs)

barrus_444332c = Graph("EMlw")
barrus_444332c.name(new = "barrus_444332c")
add_to_lists(barrus_444332c, graph_objects, all_graphs)

barrus_444433a = Graph("ER~o")
barrus_444433a.name(new = "barrus_444433a")
add_to_lists(barrus_444433a, graph_objects, all_graphs)

barrus_444433b = Graph("Et\w")
barrus_444433b.name(new = "barrus_444433b")
add_to_lists(barrus_444433b, graph_objects, all_graphs)

"""
From:
Alcón, Liliana, Marisa Gutierrez, and Glenn Hurlbert. "Pebbling in split graphs." SIAM Journal on Discrete Mathematics 28.3 (2014): 1449-1466.
"""
pyramid = Graph("EElw")
pyramid.name(new = "pyramid")
add_to_lists(pyramid, graph_objects, all_graphs)

"""
From:
p.3
Dvořák, Zdeněk, and Jordan Venters. "Triangle-free planar graphs with small independence number." arXiv preprint arXiv:1702.02888 (2017).
"""
thomas_walls_3 = Graph("IRaIACbFG")
thomas_walls_3.name(new = "thomas_walls_3")
add_to_lists(thomas_walls_3, graph_objects, all_graphs)

"""
From:
p.3
Dvořák, Zdeněk, and Jordan Venters. "Triangle-free planar graphs with small independence number." arXiv preprint arXiv:1702.02888 (2017).
"""
thomas_walls_4 = Graph("LQCkCD?OGM@EKB")
thomas_walls_4.name(new = "thomas_walls_4")
add_to_lists(thomas_walls_4, graph_objects, all_graphs)

"""
P. 24, 1st figure
From a knuth paper, linked in issue #342
"""
knuth_24_1 = Graph("E`~w")
knuth_24_1.name(new = "knuth_24_1")
add_to_lists(knuth_24_1, graph_objects, all_graphs)

"""
P. 24, 2nd figure
From a knuth paper, linked in issue #343
"""
knuth_24_2 = Graph("Er~w")
knuth_24_2.name(new = "knuth_24_2")
add_to_lists(knuth_24_2, graph_objects, all_graphs)

"""
From Craig Larson, Critical Independence paper
Has non empty critical independence set
"""
larson = Graph("F@N~w")
larson.name(new = "larson")
add_to_lists(larson, graph_objects, all_graphs)

"""
From:
Vizing's independence number conjecture is true asymptotically
Eckhard Steffen
"""
steffen_1 = Graph("HqP@xw{")
steffen_1.name(new = "steffen_1")
add_to_lists(steffen_1, graph_objects, all_graphs)

"""
From:
Vizing's independence number conjecture is true asymptotically
Eckhard Steffen
"""
steffen_2 = Graph("N??xpow@o@?AoBoBW@_")
steffen_2.name(new = "steffen_2")
add_to_lists(steffen_2, graph_objects, all_graphs)

"""
From:
Vizing's independence number conjecture is true asymptotically
Eckhard Steffen
"""
steffen_3 = Graph("Z??xpow@o@?A?B?B?@_?C??_?@??@??A???__?Dc??kO?AoO?AWO?AWG?@K?")
steffen_3.name(new = "steffen_3")
add_to_lists(steffen_3, graph_objects, all_graphs)

"""
From:
p.231
Balinski, Michel L. "Notes—On a Selection Problem." Management Science 17.3 (1970): 230-231.
"""
balanski = Graph("H??xuRo")
balanski.name(new = "balanski")
add_to_lists(balanski, graph_objects, all_graphs)

"""
Instance of the Erdos-Faber-Lovasz conjecture
"""
efl_instance = Graph("J`?GECrB~z_")
efl_instance.name(new = "efl_instance")
add_to_lists(efl_instance, graph_objects, all_graphs)

"""
From Lipták, László, and László Lovász. "Critical facets of the stable set polytope." Combinatorica 21.1 (2001): 61-88.
"""
crit_facet = Graph("J?AAHGIC^o?")
crit_facet.name(new = "critical facet graph")
add_to_lists(crit_facet, graph_objects, all_graphs)





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

#could run this occasionally to check there are no duplicates
#graph_objects = remove_duplicates(union_objects, idfun=Graph.graph6_string)
