# Objects-Invariants-Properties
**Objects, Invariants, and Properties for Graph Theory (OIP-GT)**

OIP-GT is a set of files that contain:
 - Objects (graphs)
 - Functions which compute invariants and properties of graphs
 - Theorems relating to those invariants and graphs
 - Precomputed values of those functions for those graphs

The ostensible goal of this project is to encode all of graph theory in an accessible repository. Why?:
 1. To act as a library, where researchers can easily find past results.
 2. To help researchers, educators, and students to combine computational resources with their study of graph theory.
 3. **For use in conjecturing new knowledge and testing those conjectures.**

The final point above is the most exciting one! Combining our growing repository of graph theory knowledge with automated conjecturing software has shown promise as a tool to develop potential new theorems which might not otherwise be found (see below section for examples and references).

##### More information (especially for researchers):

We note that GitHub has proven a valuable tool for this project. For one, it enables us to share this repository with a growing community of researchers. Ideally, researchers can use this repository to generate and test new conjectures. Then, researchers will contribute their findings either as counterexamples (graphs not previously in OIP-GT), or as theorems (with citations, when a conjecture is proven).

Second, it enables us to recruit students. Some of these students have interests in computer science, rather than math, and this project is a tool to bridge these fields. We have organized several summer workshops, each with a particular focus in graph theory, where students program, conjecture, and prove/disprove as a group. These efforts spur growth in OIP-GT much faster than we could produce alone.

📚 See https://arxiv.org/abs/1801.01814 for a more extensive introduction to the motivations for this project, past workshops, some of the results produced with OIP, and a history of efforts in automated conjecturing.

## Getting started :bowtie: 🔰

The below instructions should enable anyone to start using OIP-GT. If you are unable to get started, please submit an Issue if there's a specific problem, or contact us (see "Maintained by" below) otherwise.

The primary language/software for this project is [Sage](http://www.sagemath.org/). For those familiar with Python, Sage is simply an extension.

Thus, after finishing initial setup, users will need to have some basic knowledge of Python to use OIP-GT. There are links to tutorials are at the end of this section.

(For those wondering, Sage is built on [Python 2](https://www.python.org/). Sage is working towards a long-term goal of transitioning to Python 3, but we have no control over when this happens!).

### Installing Sage and CONJECTURING

For users less experienced with setting up programming environments, we suggest considering use of [CoCalc](https://cocalc.com/), an online computing environment. CoCalc comes preinstalled with Sage, making setup easier.

However, CoCalc is far from perfect. For example, it may not be the best choice for large-scale computations. Also, CoCalc's free tier has limited resources. In particular, CoCalc's free tier does not allow remote internet resources, so you won't be able to pull from GitHub to CoCalc, which would make contributing to OIP-GT difficult. But, these shouldn't affect you if you're just getting started for the first time.

At this point, you should either:
 - Create an account on [CoCalc](https://cocalc.com/), create a Sage worksheet, and possibly work through a [Sage tour](http://doc.sagemath.org/html/en/a_tour_of_sage/)/[tutorial](http://doc.sagemath.org/html/en/tutorial/).
 - Or, [install Sage](), and possibly work through a [Sage tour](http://doc.sagemath.org/html/en/a_tour_of_sage/)/[tutorial](http://doc.sagemath.org/html/en/tutorial/). For Windows users, be sure to fully read the [additional instructions](https://wiki.sagemath.org/SageWindows).

As mentioned above, a primary purpose for OIP-GT is automated conjecturing. We use the program [CONJECTURING (available on GitHub)](https://github.com/nvcleemp/conjecturing) by Nico Van Cleemput. Of course, you're free to use the data in OIP-GT in other ways too.

To install CONJECTURING, you can follow their [instructions](http://nvcleemp.github.io/conjecturing/). They describe how to install CONJECTURING as a Sage package, callable from any working directory.

However, for first-time users (and in fact for all users on CoCalc, where admin rights are limited), you can setup CONJECTURING locally by following these simplified steps:

######### CAN you make conjecturing on windows? Assuming you have Sage installed.

 1. Download the [latest release](https://github.com/nvcleemp/conjecturing/archive/0.12-CoCalc.zip) (or, you can use tools like `git` or `wget`).
 2. Extract / unzip the package. If you're using CoCalc, you can either unzip before you upload to CoCalc, or afterwards. To unzip after uploading to CoCalc, just click on the file and select "Extract Files".
 3. Copy `conjecturing.py` out of the directory `sage`. Put the copy into whatever directory you plan to work in later.
 4. `make` the contents of the directory `c`. For more help:
   1. This step requires a terminal window. On CoCalc, just select "New" and "Terminal".
   2. Use the command `ls` to list files in your current folder. Use the command `cd someFolderName` to change into `someFolder`. Repeat this until you're in the `c` folder.
   3. Now, run `make`.
 5. This should create a new directory `build` inside of `c`. Copy the file `expressions` from `build` into whatever directory you plan to work in later.

And that's it! You've installed Sage and CONJECTURING. You're now ready to install OIP-GT.

### Install OIP-GT

 1. Download and unzip the [latest release](https://github.com/math1um/objects-invariants-properties/releases).
 2. Copy the files out of the folder and into the directory you plan to work in.
 3. You're done. 🎉

#### Note for users which install OIP-GT by cloning the repository:

To build the source files, open a terminal and `cd` to the root OIP directory. Then, run `make`. This should create a new directory named `build`. Copy all of the files from `build` into the directory you plan to work in. You can do this by running `cp build/* someDirectory`, where `someDirectory` is wherever you plan to work.

### Tutorials for Python / Sage

Sage:
 - [The Sage Tutorial](http://doc.sagemath.org/html/en/tutorial/)

Python:
 - [The Python Tutorial](https://docs.python.org/3/tutorial/)
 - [Code Academy](https://www.codecademy.com/learn/learn-python)

## Examples / how to use

After completing the installation instructions above, you are ready to use OIP-GT. Note that all of the files (ex. `conjecturing.py`, `gt.sage`, ...) are assumed to be in your current working directory. In CoCalc, this is by default whatever folder your current Sage worksheet is saved in.

If you need to change your current working directory, you can run
```sh
os.chdir("someDirectory")
```
from within Sage or Python.

To start, load the different components:
```sage
load("conjecturing.py")
load("gt.sage")
load("gt_precomputed_database.sage")
```

Note that the OIP-GT GitHub repository contains lists of graphs that are not by default included in the release download. You can download some additional lists (ex. a list of all maximal triangle-free graphs up to order 16) from the [Objects directory on GitHub](https://github.com/math1um/objects-invariants-properties/tree/master/src/Objects). These are either loaded by
```sage
load("dimacsgraphs.sage")
```
or by following the instructions in the file, as in the case of `mtf_graphs.sage`.

Once you've loaded these modules, you are ready to work.

The graphs included by default are all in the list `all_graphs`. So, you can find certain graphs which meet some criteria by running something like
```sage
myGraphs = [g for g in all_graphs if g.order() < 20 and g.is_hamiltonian()]
myGraphs2 = [g for g in all_graphs if is_two_connected(g)]
```

If you have in mind a particular graph, ...

The properties...

The invariants...

Theorems...

Precomputed...

Conjecturing...

## Where to find more documentation 📜

From the command prompt:
 - In Sage, typing `??` after any command will display the included docstring. For example, typing `has_star_center??` will display documentation and examples for the functions `has_star_center(g)`.

Sage, online:
 - Built-in Sage graphs, accessible as `graphs.SomeGraph()`: http://doc.sagemath.org/html/en/reference/graphs/sage/graphs/graph_generators.html.
 - Sage `Graph` package, with built-in properties and invariants: http://doc.sagemath.org/html/en/reference/graphs/sage/graphs/generic_graph.html
 - Even more built-in Sage properties and invariants, just for undirected graphs: http://doc.sagemath.org/html/en/reference/graphs/sage/graphs/graph.html

OIP-GT, online:
 - Using the search bar on [our GitHub](https://github.com/math1um/objects-invariants-properties) will search for any mentions of a phrase in the source code, Issues, commit messages, and more.
 - You can view and download the source code from [our GitHub](https://github.com/math1um/objects-invariants-properties).

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
