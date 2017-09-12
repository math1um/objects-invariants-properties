## Objects, Invariants, and Properties for Graph Theory

Designed for use with the Sage program [CONJECTURING](http://nvcleemp.github.io/conjecturing/) and maintained by Craig Larson, Nico Van Cleemput, and Reid Barden.

OIP-GT is a set of files that contain objects (graphs), invariants and properties of graphs, and theorems relating to those invariants and graphs. While designed for automated conjecturing, these files can be used generally as well.

### Loading the file

Download the [zip file](https://github.com/math1um/objects-invariants-properties/releases/download/2017-09-12/objects-invariants-properties.zip) and unpack it. Make sure that the files are located in the directory from which you will run Sage, and not in a subdirectory. Note that to use these files on [CoCalc](https://www.cocalc.com) within a Sagemath worksheet, this means that the files should be placed in the home directory of the project. Now you can load `gt.sage`. This file contains all coded graphs, invariants, properties, and theorems. It does not load the DIMACS graphs, Sloane Graphs, or the database utilities. Use the utility methods to load the DIMACS and Sloane graphs. This populates the `dimacs_graphs` list and the `sloane_graphs` list.

```sage
sage: load("gt.sage")
sage: load_dimacs_graphs()
sage: load_sloane_graphs()
```

If you want to use the database with precomputed invariant and property values, then you need to load another file.
```sage
sage: load("gt_precomputed_database.sage")
```

### Use
Assuming you have CONJECTURING setup in the same directory as well, you can easily get conjectures using these new objects.

```sage
sage: load("conjecturing.py")
sage: conjecture(graph_objects, efficiently_computable_invariants, 0)
```

If you have also loaded the database with precomputed invariant values, you can make use of them as shown below.

```sage
sage: precomputed_invs = precomputed_invariants_for_conjecture()
sage: conjecture(graph_objects, efficiently_computable_invariants, 0, precomputed = precomputed_invs)
```
