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

This should create a subdirectory called `objects-invariants-properties`. Then you can load `gt.sage`. The utilities file is unnecessary in most cases.

```sage
sage: load("objects-invariants-properties/gt.sage")
sage: load("objects-invariants-properties/gt_utilities.sage")
```

`gt.sage` loads all of the other necessary files so you can now access all invariant methods, graph objects and lists, etc. 

##### Use

Assuming you have CONJECTURING setup here as well, you can easily get conjectures using these new objects.

```sage
sage: load("conjecturing.py")
sage: conjecture(graph_objects, efficiently_computable_invariants, 0)
```
