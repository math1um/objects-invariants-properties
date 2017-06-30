load("objects-invariants-properties/Invariants/invariants.sage")
print("loaded invariants")

load("objects-invariants-properties/Properties/properties.sage")
print("loaded properties")

load("objects-invariants-properties/Objects/graphs.sage")
print("loaded graphs")

load("objects-invariants-properties/Theorems/theorems.sage")
print("loaded theorems")

#TESTING

#check for invariant relation that separtates G from class defined by property
def find_separating_invariant_relation(g, objects, property, invariants):
    L = [x for x in objects if (property)(x)]
    for inv1 in invariants:
        for inv2 in invariants:
            if inv1(g) > inv2(g) and all(inv1(x) <= inv2(x) for x in L):
                return inv1.__name__, inv2.__name__
    print "no separating invariants"



#finds "difficult" graphs for necessary conditions, finds graphs which don't have property but which have all necessary conditions
def test_properties_upper_bound_theory(objects, property, theory):
     for g in objects:
         if not property(g) and all(f(g) for f in theory):
             print g.name()

#finds "difficult" graphs for sufficient conditions, finds graphs which dont have any sufficient but do have property
def test_properties_lower_bound_theory(objects, property, theory):
     for g in objects:
         if property(g) and not any(f(g) for f in theory):
             print g.name()

def find_coextensive_properties(objects, properties):
     for p1 in properties:
         for p2 in properties:
             if p1 != p2 and all(p1(g) == p2(g) for g in objects):
                 print p1.__name__, p2.__name__
     print "DONE!"


#load graph property data dictionary, if one exists
try:
    graph_property_data = load(os.environ['HOME'] +'/objects-invariants-properties/graph_property_data.sobj')
    print "loaded graph properties data file"
except IOError:
    print "can't load graph properties sobj file"
    graph_property_data = {}



#this version will open existing data file, and update as needed
def update_graph_property_data(new_objects,properties):
    global graph_property_data
    #try to open existing sobj dictionary file, else initialize empty one
    try:
        graph_property_data = load(os.environ['HOME'] +'/objects-invariants-properties/graph_property_data.sobj')
    except IOError:
        print "can't load properties sobj file"
        graph_property_data = {}

    #check for graph key, if it exists load the current dictionary, if not use empty prop_value_dict as *default*
    for g in new_objects:
        print g.name()
        if g.name not in graph_property_data.keys():
            graph_property_data[g.name()] = {}

        #check for property key, if it exists load the current dictionary, if not initialize an empty dictionary for property
        for prop in properties:
            try:
                graph_property_data[g.name()][prop.__name__]
            except KeyError:
                graph_property_data[g.name()][prop.__name__] = prop(g)

    save(graph_property_data, "graph_property_data.sobj")
    print "DONE"

#load graph property data dictionary, if one exists
try:
    graph_invariant_data = load(os.environ['HOME'] +'/objects-invariants-properties/graph_invariant_data.sobj')
    print "loaded graph invariants data file"
except IOError:
    print "can't load graph invariant sobj file"
    graph_invariant_data = {}


#this version will open existing data file, and update as needed
def update_graph_invariant_data(new_objects,invariants):
    #try to open existing sobj dictionary file, else initialize empty one
    global graph_invariant_data
    try:
        graph_invariant_data = load(os.environ['HOME'] +'/objects-invariants-properties/graph_invariant_data.sobj')
        print "loaded graph invariants data file"
    except IOError:
        print "can't load invariant sobj file"
        graph_invariant_data = {}

    #check for graph key, if it exists load the current dictionary, if not use empty prop_value_dict as *default*
    for g in new_objects:
        print g.name()
        if g.name not in graph_invariant_data.keys():
            graph_invariant_data[g.name()] = {}

        #check for property key, if it exists load the current dictionary, if not initialize an empty dictionary for property
        for inv in invariants:
            try:
                graph_invariant_data[g.name()][inv.__name__]
            except KeyError:
                graph_invariant_data[g.name()][inv.__name__] = inv(g)

    save(graph_invariant_data, "graph_invariant_data.sobj")
    print "DONE"
