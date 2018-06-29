#! /bin/sh
cd build
cat > db-test.sage <<SAGESCRIPT
load("gt.sage")
print "INVARIANT_START"
for inv in all_invariants:
    print inv.__name__
print "INVARIANT_END"
print "PROPERTY_START"
for prop in properties + removed_properties:
    print prop.__name__
print "PROPERTY_END"
SAGESCRIPT
cat > db-missing.py <<PYTHONSCRIPT
import sys, os
ls = [l.strip() for l in sys.stdin]
invariants = ls[ls.index("INVARIANT_START")+1:ls.index("INVARIANT_END")]
invariant_files = [(s[:-4] if s.endswith(".sql") else s) for s in os.listdir('../db/invariants')]
properties = ls[ls.index("PROPERTY_START")+1:ls.index("PROPERTY_END")]
property_files = [(s[:-4] if s.endswith(".sql") else s) for s in os.listdir('../db/properties')]
missing_invariants = [s for s in invariants if s not in invariant_files]
orphaned_invariants = [s for s in invariant_files if s not in invariants]
missing_properties = [s for s in properties if s not in property_files]
orphaned_properties = [s for s in property_files if s not in properties]
print "Invariants"
print "=========="
print "Missing"
print "-------"
if missing_invariants:
    for inv in missing_invariants:
        print "  {}".format(inv)
else:
    print "No missing invariants"
print "Orphaned"
print "--------"
if orphaned_invariants:
    for inv in orphaned_invariants:
        print "  {}".format(inv)
else:
    print "No orphaned invariants"
print
print "Properties"
print "=========="
print "Missing"
print "-------"
if missing_properties:
    for prop in missing_properties:
        print "  {}".format(prop)
else:
    print "No missing properties"
print "Orphaned"
print "--------"
if orphaned_properties:
    for prop in orphaned_properties:
        print "  {}".format(prop)
else:
    print "No orphaned properties"
if len(orphaned_invariants) + len(orphaned_properties):
    print "There are orphaned invariants and/or properties!"
    exit(1)
PYTHONSCRIPT
$HOME/SageMath/sage db-test.sage | python db-missing.py
