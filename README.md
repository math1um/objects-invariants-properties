## OIP-GT

#### Objects, Invariants, and Properties for Graph Theory

Designed for use with the Sage program [CONJECTURING](http://nvcleemp.github.io/conjecturing/) and maintained by Craig Larson, Nico Van Cleemput, and Reid Barden.

OIP-GT is a set of files that contain objects (graphs), invariants and properties of graphs, and theorems relating to those invariants and graphs. While designed for automated conjecturing, these files can be used generally as well.

During development the project is split over multiple file, but during the build process everything from the objects (graphs), invariants and properties part will be compiled into a single file.

##### Setup
Currently, it is set up to be saved and loaded a specific way and will not function without it being this way. Note that the development version CANNOT directly be loaded into Sage. Below we describe the process of to build and use the project. If you are not interested in contributing to the project, and just want to use the final result, then you can just skip over this part, and download the built files from our website.

First cd to the root of your project, then clone the oip-gt repo from Github there.

```sh
$ cd path/to/project
$ git clone https://github.com/math1um/objects-invariants-properties
```

NOTE ON BRANCHES: The master branch is known to load with no issues on Mac running Mac OSX v10.11.6 using a bare bones installation of SageMath version 7.6, Release Date: 2017-03-25. However, it is known to throw some errors on RHELS 7.3 with the gap_packages installed. The compile_server branch will be used to test this issue.

This should create a subdirectory called `objects-invariants-properties`. You can cd to this directory and run the make script.

```sh
$ cd objects-invariants-properties
$ make build
```

This should create a subdirectory called `build` containing some Sage files and a database file. Copy these files to the root of your project, or to the location from which you want to run Sage.

##### Loading the file

Either get the files using the way described in Setup, or download the zip file on our website. Make sure that the files are located in the directory from which you will run Sage, and not in a subdirectory. Note that to use these files on CoCalc within a Sagemath worksheet, this means that the files should be placed in the home directory of the project. Now you can load `gt.sage`. GT all coded graphs, invariants, properties, and theorems. It does not load the DIMACS graphs, Sloane Graphs, or the database utilities. Use the utility methods to load the DIMACS and Sloane graphs. This populates the `dimacs_graphs` list.

```sage
sage: load("gt.sage")
sage: load_dimacs_graphs()
sage: load_sloane_graphs()
```

If you want to use the database with precomputed invariant and property values, then you need to load another file.
```sage
sage: load("gt_precomputed_database.sage")
```

##### Use

Assuming you have CONJECTURING setup here as well, you can easily get conjectures using these new objects.

```sage
sage: load("conjecturing.py")
sage: conjecture(graph_objects, efficiently_computable_invariants, 0)
```

If you have also loaded the database with precomputed invariant values, you can make use of them as shown below.

```sage
sage: precomputed_invs = precomputed_invariants_for_conjecture()
sage: conjecture(graph_objects, efficiently_computable_invariants, 0, precomputed = precomputed_invs)
```

##### Known Issues

The graphs McLaughlinGraph and LocalMcLaughlinGraph, that are built in to sage, cannot be created on a barebones installation of Sage. They require extra packages. This is not an issue on Cocalc/SageMathCloud.
