## OIP-GT

#### Objects, Invariants, and Properties for Graph Theory

Designed for use with the Sage program [CONJECTURING](http://nvcleemp.github.io/conjecturing/) and maintained by Craig Larson, Nico Van Cleemput, and Reid Barden.

OIP-GT is a set of files that contain objects (graphs), invariants and properties of graphs, and theorems relating to those invariants and graphs. While designed for automated conjecturing, these files can be used generally as well.

##### Setup
Currently, it is set up to be saved and loaded a specific way and will not function without it being this way. First cd to the root of your project, then clone the oip-gt repo from Github there.

```sh
$ cd path/to/project
$ git clone https://github.com/math1um/objects-invariants-properties
```

This should create a subdirectory called `objects-invariants-properties`. Then you can load `gt.sage`. GT loads the utilities file, as well as all coded graphs, invariants, properties, and theorems. It does not load the DIMACS graphs, Sloane Graphs, or the database utilities. Use the utility methods to load the DIMACS and Sloane graphs. This populates the `dimacs_graphs` list.

```sage
sage: load("objects-invariants-properties/gt.sage")
sage: load_dimacs_graphs()
sage: load_sloane_graphs()
```

##### Use

Assuming you have CONJECTURING setup here as well, you can easily get conjectures using these new objects.

```sage
sage: load("conjecturing.py")
sage: conjecture(graph_objects, efficiently_computable_invariants, 0)
```

##### Known Issues

The graphs McLaughlinGraph and LocalMcLaughlinGraph, that are built in to sage, cannot be created on a barebones installation of Sage. They require extra packages. This is not an issue on Cocalc/SageMathCloud. 
