# Objects-Invariants-Properties
Objects, Invariants, and Properties for Graph Theory (OIP-GT)

OIP-GT is a set of files that contain:
 - Objects (graphs)
 - Functions which compute invariants and properties of graphs
 - Theorems relating to those invariants and graphs
 - Precomputed values of those functions for those graphs

The ostensible goal of this project is to encode all known graph theory in a usable, accessible way. Why?:
 1. To act as a library, where researchers can find past results in a standard format.
 2. To aid students, educators, and researchers in combining modern computational resources with their study of graph theory.
 3. **For use in conjecturing new knowledge and testing those conjectures.**

The final point above is the primary concern of this project's maintainers. Combining a growing repository of graph theory knowledge with automated conjecturing software (in particular, [CONJECTURING](http://nvcleemp.github.io/conjecturing/), described later) has proven effective in generating interesting conjectures.

Conjecturing also creates a feedback loop, since once proved/disproved they can themselves be added to OIP as theorems/counterexamples!

###### More information (especially for researchers):

We note that GitHub has proven a valuable tool for this project. For one, it enables us to share this repository with a growing community of researchers. Second, it enables us to recruit students (many with interests in Computer Science, Math, or both) to contribute to OIP at a much faster rate than we could alone. These contributions are often organized as part of summer workshops.

See https://arxiv.org/abs/1801.01814 for a more extensive introduction to the motivations for this project, past workshops, some of the results produced with OIP, and a history of efforts in automated conjecturing.

## Getting started

The below instructions should enable anyone to start using OIP-GT. If you are unable to get started, please submit an Issue if there's a specific problem, or contact us (see "Maintained by" below) otherwise.

The primary language/software for this project is [Sage](http://www.sagemath.org/). For those familiar with Python, Sage is simply an extension.

Thus, after finishing initial setup, users will need to have some basic knowledge of Python to use OIP-GT. There are links to tutorials are at the end of this section.

(For those wondering, Sage is built on [Python 2](https://www.python.org/). Sage is working towards a long-term goal of transitioning to Python 3, but we have no control over when this happens!).

### Installing Sage and CONJECTURING

For users less-experienced with setting up programming environements, we suggest considering use of [CoCalc](https://cocalc.com/), an online computing environment. CoCalc comes preinstalled with Sage, making setup easier.

However, CoCalc is far from perfect. For example, it may not be the best choice for large-scale computations. Also, CoCalc's free tier has limited resources. In particular, CoCalc's free tier does not allow remote internet resources, so you won't be able to pull from GitHub to CoCalc, which would make contributing to OIP-GT difficult. But, these shouldn't affect you if you're just getting started for the first time.

At this point, you should either:
 - Create an account on [CoCalc](https://cocalc.com/), create a Sage worksheet, and then possibly work through a [Sage tour](http://doc.sagemath.org/html/en/a_tour_of_sage/)/[tutorial](http://doc.sagemath.org/html/en/tutorial/).
 - Or, [install Sage](), and then possibly work through a [Sage tour](http://doc.sagemath.org/html/en/a_tour_of_sage/)/[tutorial](http://doc.sagemath.org/html/en/tutorial/). For Windows users, be sure to fully read the [additional instructions](https://wiki.sagemath.org/SageWindows).


program [CONJECTURING (available on GitHub)](http://nvcleemp.github.io/conjecturing/).

### Install OIP-GT (without git)

### Tutorials for Python / Sage / git

## More documentation

## How to use / Examples

## Contributing

Instructions for novices...

Guidelines for not...

## Maintained by:
 - Craig Larson (@math1um). Email: clarson@vcu.edu  Web: http://www.people.vcu.edu/~clarson/
 - Nick Van Cleemput (@nvcleemp).

Other significant contributors include:
 - Reid Barden (@rbarden), Summery 2017. Web: https://reidbarden.com/
 - Justin Yirka (@yirkajk), Summer 2018. Web: https://www.justinyirka.com/



#########################
# Old stuff below.



During development the project is split over multiple file, but during the build process everything from the objects (graphs), invariants and properties part will be compiled into a single file.

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


##### Setup

```
##### Loading the file

Either get the files using the way described in Setup, or download the zip file on our website. Make sure that the files are located in the directory from which you will run Sage, and not in a subdirectory. Note that to use these files on CoCalc within a Sagemath worksheet, this means that the files should be placed in the home directory of the project. Now you can load `gt.sage`. GT all coded graphs, invariants, properties, and theorems. It does not load the DIMACS graphs, Sloane Graphs, or the database utilities. Use the utility methods to load the DIMACS and Sloane graphs. This populates the `dimacs_graphs` list.

```sage
sage: load("gt.sage")
sage: load_dimacs_graphs()
sage: load_sloane_graphs()
```

If you want to use the database with precomputed invariant and property values, then you need to load another file.
sage
sage: load("gt_precomputed_database.sage")
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
