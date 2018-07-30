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

📚 See https://arxiv.org/abs/1801.01814 for a more extensive introduction to the motivations for this project, past workshops, some of the results produced with OIP, and a history of efforts in automated conjecturing.

## Getting started :bowtie: 🔰

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

## Examples / how to use

During development the project is split over multiple file, but during the build process everything from the objects (graphs), invariants and properties part will be compiled into a single file.

Currently, it is set up to be saved and loaded a specific way and will not function without it being this way. Note that the development version CANNOT directly be loaded into Sage. Below we describe the process of to build and use the project. If you are not interested in contributing to the project, and just want to use the final result, then you can just skip over this part, and download the built files from our website.

First cd to the root of your project, then clone the oip-gt repo from Github there.

```sh
$ cd path/to/project
$ git clone https://github.com/math1um/objects-invariants-properties
```

This should create a subdirectory called `objects-invariants-properties`. You can cd to this directory and run the make script.

```sh
$ cd objects-invariants-properties
$ make build
```

This should create a subdirectory called `build` containing some Sage files and a database file. Copy these files to the root of your project, or to the location from which you want to run Sage.
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

## Where to find more documentation 📜

## Contributing

Note: Please don't file an issue to ask a question. See "Maintained by" below for contact information.

### Reporting bugs 🐞🐛 and Suggesting improvements 👍

Have you found a bug / problem with OIP-GT 🐞🐛? Do you have a suggestion for an improvement? We track both as [GitHub issues](https://github.com/math1um/objects-invariants-properties/issues).

Note that improvements / enhancements may include requests for new functions, requests for additional graphs, suggestions for improving usability, documentation, or reliability, and more.

 1. Before submitting a report, please make sure you're using the [latest release of OIP-GT](https://github.com/math1um/objects-invariants-properties/releases). Maybe your bug/suggestion has already been resolved.

 2. Next, search the list of issues for related reports to see if the problem has already been reported / feature has already been suggested. If it has and the issue is still open, you can add any new information as a comment to the existing issue instead of opening a new one.

 3. If you've reached this step, then you should open an issue! See our [Contributing Guidelines](CONTRIBUTING.md) for details on what information to provide. In general, the more information, the better.

Keep in mind that we have a small team, so be sure to describe the impact of your bug (is it critical, or just bothersome?) / why your suggestion is important (how many Fields Medals will this result in?). If you'd like things done sooner than later, see below to see how YOU can contribute the code.

### Submitting code (via pull requests) 🎁💘

Contributions can include resolving any open [issue](https://github.com/math1um/objects-invariants-properties/issues), such as programming a new property, adding some precomputed values to the database, or improving documentation.

Anybody is welcome to contribute! If you're not sure where to start, please, please contact us. We want to get more researchers/developers involved in contributing to OIP-GT. There are plenty of "beginner" issues available.

To contribute, you'll need to be familiar with GitHub, Pull requests, and probably with programming sage. You can find some Sage tutorials linked above. Here are some guides to help with GitHub:
 - [GitHub Hello World](https://guides.github.com/activities/hello-world/)
 - [Forking Projects and Pull Requests](https://guides.github.com/activities/forking/)
 - [Mastering issues and working with other devs](https://guides.github.com/features/issues/)

The summarized version of the process includes:
 1. Find an issue you'd like to help with. If you have a bug or feature request you'd like to resolve, then you should still begin by following the steps above to create an issue (that way we understand what bug or feature you're resolving!). Otherwise, you should check out the list of issues to find something that interests you.
 2. Clone OIP-GT so that you can edit the source code.
 3. Make changes. See our [Contributing Guidelines](CONTRIBUTING.md) for code requirements and expectations!
 4. Submit a pull request.
 5. Work with us to answer any questions and make any improvements as we review your pull request.
 6. Party! 🎉

See our [Contributing Guidelines](CONTRIBUTING.md) for code requirements and expectations, and for a more detailed description of our review process.

## Maintained by 😎
Contact:
- Craig Larson (@math1um). Email: clarson@vcu.edu  Web: http://www.people.vcu.edu/~clarson/

Current maintainers:
 - Craig Larson (@math1um). Email: clarson@vcu.edu  Web: http://www.people.vcu.edu/~clarson/
 - Nico Van Cleemput (@nvcleemp).
 - Justin Yirka (@yirkajk), Summer 2018. Web: https://www.justinyirka.com/

Past significant contributions by:
 - Reid Barden (@rbarden), Summery 2017. Web: https://reidbarden.com/
