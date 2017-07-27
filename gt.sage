load("objects-invariants-properties/gt_utilities.sage")
print("loaded utilities")

load("objects-invariants-properties/Invariants/invariants.sage")
print("loaded invariants")

load("objects-invariants-properties/Properties/properties.sage")
print("loaded properties")

load("objects-invariants-properties/Theorems/theorems.sage")
print("loaded theorems")

load("objects-invariants-properties/Objects/graphs.sage")
print("loaded graphs")

print("\nRemember to load DIMACS and Sloane graphs if you want them")

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
