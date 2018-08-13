# Objects-Invariants-Properties

## Contents
1. [Intro](#contents)
2. [Getting Started 🔰](#getting-started-)
    1. [Required Knowledge 💭❓](#required-knowledge-)
    2. [Tutorials 🐍🐧](#tutorials)
    3. [What computing environment are you using? 💻](#what-computing-environment-are-you-using-)
        - [CoCalc](#cocalc)
    4. [Install Sage 1️⃣](#install-sage-1️⃣)
    5. [Install CONJECTURING 2️⃣](#install-conjecturing-2️⃣)
    6. [Install OIP-GT 3️⃣](#install-oip-gt-3️⃣)
3. [Examples / Tutorial 🎓](#examples--tutorial-)
    1. [Load OIP-GT](#load-oip-gt)
    2. [Graphs](#graphs)
    3. [Properties](#properties)
    4. [Invariants](#invariants)
    5. [Conjecturing](#conjecturing)
        - [Add theorems to conjecturing](#add-theorems-to-conjecturing)
        - [Add precomputed values to conjecturing](#add-precomputed-values-to-conjecturing)
    6. [Precompute additional values](#precompute-additional-values)
4. [Where to find more examples and documentation 📜](#where-to-find-more-examples-and-documentation-)
5. [Contributing](#contributing)
    1. [Reporting bugs 🐞🐛 and suggesting improvements ✨](#reporting-bugs--and-suggesting-improvements-)
    2. [Submitting code (via pull requests) ❤️🎁](#submitting-code-via-pull-requests-️)
6. [Maintained by 😎](#maintained-by-)

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

##### More information (especially for researchers) 📚📚:

GitHub has proven a valuable tool for this project.

For one, it enables us to share this repository with a growing community of researchers. Ideally, researchers can use this repository to generate and test new conjectures. Then, researchers will contribute their findings either as counterexamples (graphs not previously in OIP-GT) or as theorems (with citations, when a conjecture is proven).

Second, it enables us to recruit students. Some of these students have interests in computer science, rather than math, and this project is a tool to bridge these fields. We have organized several summer workshops, each with a particular focus in graph theory, where students program, conjecture, and prove/disprove as a group. These efforts spur growth in OIP-GT much faster than we could produce alone.

📔 See [https://arxiv.org/abs/1801.01814](https://arxiv.org/abs/1801.01814) for a more extensive introduction to the motivations for this project, past workshops, some of the results produced with OIP, and a history of efforts in automated conjecturing.

📔 See [https://www.sciencedirect.com/science/article/pii/S0004370215001575](https://www.sciencedirect.com/science/article/pii/S0004370215001575) for a description of the CONJECTURING program's algorithm.

## Getting started 🔰

We've written the below instructions so that **anybody** can use OIP-GT. This is a math project. We hope to recruit mathematicians, no matter how programming-phobic they are!

If you have trouble getting started, please contact us (see [Maintained by](#maintained-by-) at the end). If you find a specific problem that means OIP-GT doesn't work on your system, please submit an Issue (see [Reporting bugs](#maintained-by-) below).

### Required knowledge 💭❓

The primary language used to interact with OIP-GT is [Sage](http://www.sagemath.org/). For those familiar with Python, Sage is Python 2 with many built-in libraries and some syntactic sugar.

Even if you don't know Python, the below instructions should be detailed enough to get you started! There are links to Sage and Python tutorials below.

Users may also find it helpful to know about git, GitHub, and working in the shell/terminal. This isn't required for most usage, but it helps when problem-solving and if you'd like to contribute to the repository.

#### Tutorials

- Sage:
    - [The Sage Tutorial](http://doc.sagemath.org/html/en/tutorial/)
- Python 🐍:
    - [The Python Tutorial](https://docs.python.org/3/tutorial/)
    - [Code Academy](https://www.codecademy.com/learn/learn-python)
- GitHub and git:
    - [GitHub Hello World](https://guides.github.com/activities/hello-world/)
    - [Forking Projects and Pull Requests](https://guides.github.com/activities/forking/)
    - [Mastering issues and working with other developers](https://guides.github.com/features/issues/)
    - [Code Academy](https://www.codecademy.com/learn/learn-git)
- Terminal / shell / Unix 🐧:
    - [Ryan's Tutorials](https://ryanstutorials.net/linuxtutorial/)
    - [Code Academy](https://www.codecademy.com/learn/learn-the-command-line)

### What computing environment are you using? 💻

You can run Sage and OIP-GT on any machine and in any environment you'd like! - with the following caveats and advice:

#### CoCalc

For users less experienced with setting up and managing programming environments, you might consider using the online computing environment [CoCalc](https://cocalc.com/), which comes preinstalled with Sage and many other features. CoCalc is run by the developers of Sage.

However, CoCalc is far from perfect. Resources are limited for users on the free tier. In particular, CoCalc's free tier does not allow remote internet resources, so you'll have to add additional files by first downloading them to your machine and then uploading them to CoCalc. The paid tiers do offer more features, but even with those features, CoCalc will not be ideal for advanced users who want to run large-scale computations.

#### Windows

Windows users may find things difficult locally. Some software, including git, just works more easily on Unix systems. And, CONJECTURING currently requires users to build using `make`. This is possible to do on Windows, but you'll have to do some googling (maybe you  can use [Linux on Windows](https://docs.microsoft.com/en-us/windows/wsl/install-win10)). If you'd prefer not to deal with these challenges, then you might want to find a Unix environment to use (virtualize Linux with [VirtualBox](https://www.virtualbox.org/), or `ssh` into a remote server provided by your university), or using CoCalc might be a good option.

#### Remote server 🐧

Note that if you use a remote server, such as a shared computing environment at your university, you may need to contact your sysadmin for help installing some software. Also, when running Sage remotely using only a terminal/CLI, you may lose the ability to use the awesome function
```sage
someGraph.show()
```
which draws a picture of given graph object. In this case, a combination of running Sage locally and remotely, or remotely and on CoCalc, may be a good option.

### Install Sage 1️⃣

To set up Sage, either:
- Create an account on [CoCalc](https://cocalc.com/), create a Sage worksheet, and possibly work through a [Sage tour](http://doc.sagemath.org/html/en/a_tour_of_sage/)/[tutorial](http://doc.sagemath.org/html/en/tutorial/).
- Or, [download and install Sage](http://www.sagemath.org/index.html), and possibly work through a [Sage tour](http://doc.sagemath.org/html/en/a_tour_of_sage/)/[tutorial](http://doc.sagemath.org/html/en/tutorial/). For Windows users, be sure to skim the [additional instructions](https://wiki.sagemath.org/SageWindows).

### Install CONJECTURING 2️⃣

As mentioned above, the primary purpose for OIP-GT is automated conjecturing. Of course, you're free to use the data in OIP-GT in other ways too.

We designed OIP-GT with the program [CONJECTURING (available on GitHub)](https://github.com/nvcleemp/conjecturing) by Nico Van Cleemput in mind.

To install CONJECTURING, you can follow their [instructions](https://github.com/nvcleemp/conjecturing/tree/master/spkg) for installing CONJECTURING as a Sage package. This installs the program as a package for all users on the machine, accessible from any working directory, and will probably require admin privileges (you cannot get admin privileges on CoCalc).

For CoCalc users and for first-time users who would prefer a simpler process, you can set up CONJECTURING by following these steps:
1. Download the [latest release of CONJECTURING](https://github.com/nvcleemp/conjecturing/releases) (choose the "CoCalc" version whether you are using CoCalc or not).
2. Extract / unzip the package. If you're using CoCalc, upload the zipped file to CoCalc first, then unzip after uploading by clicking on the file and selecting "Extract Files".
3. In the new folder / directory, there should be a directory named `sage`. Copy `conjecturing.py` out of `sage` into whatever directory you plan to work in later.
4. Now, build the contents of the directory `c`. As described above, this may not be simple to do on Windows. For Unix (including CoCalc) users:
    1. This step requires a terminal window. On CoCalc, select "New" and "Terminal".
    2. Use the command `ls` to list files in your current folder. Use the command `cd someFolderName` to change into `someFolder`. Repeat this until you're in the `c` folder.
    3. Now, run the command `make`.
5. This should create a new directory `build` inside of `c`. Copy the file `expressions` from `build` into whatever directory you plan to work in later.

### Install OIP-GT 3️⃣

1. Download and unzip the [latest release](https://github.com/math1um/objects-invariants-properties/releases).
2. Copy the files each of the files into the directory you plan to work in.
3. You're done. 🎉

#### Note for users who install OIP-GT by cloning the repository:

To build the source files, open a terminal and `cd` to the root OIP directory. Then, run `make`. This should create a new directory named `build`. Copy all of the files from `build` into the directory you plan to work in. You can do this by running `cp build/* someDirectory`, where `someDirectory` is wherever you plan to work.

The above process takes the individual files containing graphs / Objects, Properties, Invariants, etc. and builds a single file `gt.sage`. To use OIP-GT, load `gt.sage`. To contribute to OIP-GT, edit the individual files.

## Examples / Tutorial 🎓

See the links to [additional examples and documentation](#where-to-find-more-examples-and-documentation-) after this section for more help.

The below examples assume that `gt.sage` and the other files you downloaded above are located in your current working directory. If not, then either copy the files, use `cd`, or use `os.chdir("dirName")`.

### Load OIP-GT

To start, load the different components:
```sage
load("conjecturing.py")
load("gt.sage")
load("gt_precomputed_database.sage")
```

Note that the OIP-GT GitHub repository contains lists of graphs that are not by default included in the release download. You can download some additional lists (ex. a list of all maximal triangle-free graphs up to order 16) from the [Objects directory on GitHub](https://github.com/math1um/objects-invariants-properties/tree/master/src/Objects). These are either loaded with a command like
```sage
load("dimacsgraphs.sage")
```
or by following the instructions in the file, as in the case of `mtf_graphs.sage`.

### Graphs

The graphs in `gt.sage` are all in the list `all_graphs`.

You can find graphs which meet some criteria by running something like
```sage
myGraphs = [g for g in all_graphs if g.order() < 20 and g.is_hamiltonian()] # All Hamiltonian graphs of order less than 20.
myGraphs2 = [g for g in all_graphs if is_two_connected(g)] # All the 2-connected graphs
```
Note that when we say "All the 2-connected graphs", we do not mean "all of the 2-connected graphs in the universe"; we only mean a subset of the graphs that we have programmed into `gt.sage`.

To check whether a graph isomorphic to a particular graph is in some list, use the function
```sage
does_graph_exist(someGraphObject, someList)
```
which will print the names of any graphs isomorphic to `someGraphObject` and return a Boolean.

You can find many graph generators built-in to the Sage `graphs` package. These work like
```sage
myGraph = graphs.BullGraph()
myGraph2 = graphs.CompleteGraph(5)
```

### Properties

Graph properties are functions which take a graph as input and return `True` or `False`.

Some properties are built-in to Sage. These are part of the `Graph` class and are called like
```sage
myGraph.is_hamiltonian()
myGraph.is_planar()
```

We have many more properties built-in to `gt.sage`. These functions are called like
```sage
has_star_center(myGraph)
is_two_connected(myGraph)
```

Here are some lists of properties built-in to `gt.sage`:
```sage
properties
efficiently_computable_properties
intractable_properties
sage_properties
```
The list `properties` contains all the built-in properties, and these are the properties we have precomputed values for. The list `efficiently_computable_properties` are properties we have identified as "efficient", usually meaning polynomial-time complexity; `intractable_properties` are all other properties. The list `sage_properties` is the subset of `properties` which are functions from the `Graph` class.

### Invariants

Graph invariants are functions which take a graph as input and return a number.

Some invariants are in to Sage. These are part of the `Graph` class and are called like
```sage
myGraph.order()
myGraph.size()
```

We have many more invariants built-in to `gt.sage`. These functions are called like
```sage
max_degree(myGraph)
independence_number(myGraph)
```

Here are some lists of invariants in to `gt.sage`:
```sage
all_invariants
efficient_invariants
intractable_invariants
sage_efficient_invariants
sage_intractable_invariants
```
The list `all_invariants` contains all the built-in invariants, and these are the invariants we have precomputed values for. The list `efficient_invariants` are the invariants we have identified as "efficient", usually meaning polynomial-time complexity; `intractable_invariants` are all other invariants. The lists `sage_efficient_invariants` and `sage_intractable_invariants` are the subsets which are functions from the Sage `Graph` class.

### Conjecturing

Running the conjecturing program is simple. Just make sure that you have loaded `conjecturing.py` and then run
```sage
myGraphs = [g for g in all_graphs[0:20] if g.order() < 40] # Picks a list of small graphs from the beginning of all_graphs
conjs = propertyBasedConjecture(myGraphs, efficiently_computable_properties[0:10], 0, sufficient=True)
for c in conjs:
    print c
```

Any conjecture printed out will be true for all input graphs and properties. *But*, whether it's in general is left for you to prove or disprove.

If you were conjecturing on invariants, the command would be
```sage
conjs = conjecture(someGraphs, someInvariants, mainInvariant, upperBound=True)
```

In the first command, `mainProperty` is set to 0. In the second command, `mainInvariant` is not yet set. You should set these values to the index of the property/invariant you are want conditions or bounds on. For example, the first command finds sufficient conditions for the first property in the list `efficiently_computable_properties[0:10]` (which was `is_regular` when I ran it). If you'd like conditions related to the second property in the list, whatever it happens to be, you would pass `1`.

Note that in the above two commands, you can set the `sufficient` and `upperBound` parameters to `True` or `False`. They both default to `True`. If set to `False`, then conjectures will be made for necessary conditions or for lower bounds, respectively.

Besides setting the list of functions to analyze and the function of interest, there is one other way to introduce a condition into your conjectures. For example, if I want a bound on the independence number of Hamiltonian graphs, it may not be immediately obvious how to make CONJECTURING consider both independence number (invariant) and Hamiltonicity (property). Here, if the list of `someGraphs` you pass contains *only* Hamiltonian graphs, then any conjecturing bounds will have the implicit qualifier "If the graph is Hamiltonian, then...".

The process of conjecturing is relatively fast, with the bottleneck usually being the computation of each function on each graphs. See [Add precompute values to conjecturing](#add-precomputed-values-to-conjecturing) below for one way to speed things up. Another way to reduce the time to conjecture is to reduce the size of the inputs. One heuristic we have found useful is to conjecture on a relatively small list of graphs, select which conjectures we find interesting or viable, and then (usually, quickly) search for counterexamples in the full list of graphs.

One final note concerning pedantic cases: we encourage you not to include the empty graph, `Graph(0)`, `Graph(1)`, or any other possibly frustrating graph in your conjecture input. We have attempted to consider these cases when defining the functions in `gt.sage`. However, since there is often a lot of disagreement in these cases (is the empty graph complete? Hamiltonian? 2-regular?), passing them as input may prevent CONJECTURING from making some otherwise viable conjectures.

For more help with CONJECTURING, we encourage you to run the commands
```sage
conjecture?
propertyBasedConjecture?
```
so you can see the full list of parameters to each function.

#### Add theorems to conjecturing

Conjectures are most interesting when they **improve** on already known results.

The definition of "improve" is important here. First consider properties. If we are looking for sufficient conditions C that imply property P, the CONJECTURING program will only output a conjecture if the number of graphs implied to be P increases from what is already implied by other theory. Similarly, if we are looking for necessary conditions C that are implied by property P, a conjecture will be made only if the number of graphs with conditions necessary for P would decrease.

In the case of invariants, the definition of "improve" is more obvious, as any inequality which is tighter/closer to the observed/known results than current theory is.

So, as you add theorems to your conjecturing, be careful of what we call "bingos". If your property theorems already characterize all of the graphs, or your invariant theorems reach equality, then the program will not make any new conjectures. In these cases, you could add more graphs. You might also remove functions which are already well-described by theory, and focus on other relationships; for example, if looking for invariant conjectures with complete graphs, I would not include both `size` and `order` in my list of invariants.

For properties, set the `theory` parameter to a list of properties which return a Boolean which imply the main property when `True` (are implied to be `True` when the main property is `True`) when when for sufficient (necessary):
```sage
propertyBasedConjecture(objects, properties, mainProperty, theory=listofThoerems)
```

For invariants, set the `theory` parameter to a list of functions which return known bounds on the main invariants:
```sage
conjecture(objects, invariants, mainInvariant, theory=listOfTheorems)
```

There are some theorems built-in to `gt.sage`. Again, these are just various functions, some sorted into various lists. Find a list of these theorems [here](src/Theorems/theorems.sage).

#### Add precomputed values to conjecturing

Make sure that you have loaded `gt_precomputed_database.sage` and that `gt_precomputed_database.db` is in the current working directory.

For properties, run the conjecture as you normally would, but setting the `precomputed` parameter as:
```sage
precomputedDictionary = precomputed_properties_for_conjecture()
propertyBasedConjecture(objects, properties, mainProperty, precomputed=precomputedDictionary)
```

For invariants, run the conjecture as you normally would, but setting the `precomputed` parameter as:
```sage
precomputedDictionary = precomputed_invariants_for_conjecture()
conjecture(objects, invariants, mainInvariant, precomputed=precomputedDictionary)
```

If the values for the given graphs for the given functions exist, then this will help massively speed up the conjectures.

Not all values for all graphs and functions have been precomputed, especially for larger graphs and slower functions. To check or filter by what has been computed, use commands like
```sage
precomputed = properties_as_dict()
g_key = myGraph.canonical_label(algorithm='sage').graph6_string() # Values are stored according to graph's **canonical** graph6 string.
print g_key in precomputed # Check if the graph exists at all in the precomputed database
print someProperty.__name__ in precomputed[g_key] # Check if the value for some property has been computed for the graph.
```
The same commands will work for invariants, if `properties_as_dict` is replaced by `invariants_as_dict`.

### Precompute additional values

If you would like to computed additional values, either for graphs and functions combinations we have neglected, or for graphs and functions you have added yourself, then use the following functions.

In the below functions, you specify the list of functions and graphs to compute, and the number of seconds to spend on each combination before timing out and moving on.
```sage
update_invariant_database(someinvariants, graphs, timeout=60)
update_property_database(someProperties, graphs, timeout=60)
```

Keep in mind that the graphs and functions passed to the update methods must be unique. In particular, the no graphs can be isomorphic to each other.

If you would to contribute your computations to the OIP-GT repository, you will need to use the below command to dump the entire `gt_precomputed_database.db` file into a collection of individual `.sql` files. For further instructions on contributing, see [Contributing](#contributing) below.
```sage
dump_database("filePathWhereToDump")
```

## Where to find more examples and documentation 📜

Sage and OIP-GT, from the command prompt:
- In Sage, typing `?` or `??` after any command will display documentation and examples. For example,
```sage
has_star_center?
```
or
```sage
Graph.is_hamiltonian?
```

Python, from the command prompt (see links to tutorials above for other questions):
- Run the command `help(object)` to get information on any Python object (the single `?` above is just a shorthand version of `help`). For example,
```python
help(print)
help([1,2,3])
```

Sage, online documentation:
- List of built-in Sage graphs, called as `graphs.SomeGraph()`: http://doc.sagemath.org/html/en/reference/graphs/sage/graphs/graph_generators.html.
- The Sage `Graph` package, with a list of built-in properties and invariants called as `someGraph.is_some_property()`: http://doc.sagemath.org/html/en/reference/graphs/sage/graphs/generic_graph.html
- Even more built-in Sage properties and invariants for undirected graphs, called the same as above: http://doc.sagemath.org/html/en/reference/graphs/sage/graphs/graph.html

OIP-GT, online:
- Using the search bar on [our GitHub](https://github.com/math1um/objects-invariants-properties) will search for any mentions of a phrase in the source code, issues, commit messages, and more.
- You can view and download the source code from [our GitHub](https://github.com/math1um/objects-invariants-properties). **This can be useful if you're wondering what lists or graphs or functions are already built-in to OIP-GT.**

## Contributing

Note: Please don't file an issue to ask a question. See [Maintained by](#maintained-by-) below for contact information.

### Reporting bugs 🐞🐛 and Suggesting improvements ✨

Have you found a bug / problem with OIP-GT 🐞🐛? Do you have a suggestion for an improvement ✨? We track both as [GitHub issues](https://github.com/math1um/objects-invariants-properties/issues).

Note that improvements / enhancements may include requests for new functions, requests to add graphs, suggestions for improving usability, documentation, or reliability, and more.

1. Before submitting a report, please make sure you're using the [latest release of OIP-GT](https://github.com/math1um/objects-invariants-properties/releases). Maybe your bug/suggestion has already been resolved.
2. Next, search the list of issues for related reports to see if the problem has already been reported / feature has already been suggested. If it has and the issue is still open, you can add any new information as a comment to the existing issue instead of opening a new one.
3. If you've reached this step, then you should open an issue! See our [Contributing Guidelines](CONTRIBUTING.md) for details on what information to provide. In general, the more information, the better.

Keep in mind that we have a small team, so be sure to describe the impact of your bug (is it critical, or just bothersome?) / why your suggestion is important (how many Fields Medals will this result in?). If you'd like things done more quickly, see below to see how YOU can contribute the code.

### Submitting code (via pull requests) ❤️🎁

Contributions can include resolving any open [issue](https://github.com/math1um/objects-invariants-properties/issues), such as programming a new property, adding precomputed values to the database, improving documentation, and more.

Everybody is welcome to contribute! If you're not sure where to start, please contact us. We want to get more researchers / developers involved in contributing to OIP-GT. There are plenty of "beginner" issues available.

To contribute, you'll need to be familiar with GitHub pull requests and with programming - although the amount and type of programming may be minimal, depending on the issue. You can use the tutorials linked to above in [Getting Started](#getting-started-bowtie-).

The basic process is (more details in our [Contributing Guidelines](CONTRIBUTING.md)):
1. Find an issue you'd like to help with. If you have a bug or feature request you'd like to resolve, then you should still begin by following the steps above to create an issue (that way we understand what bug or feature you're resolving!). Otherwise, you should check out the list of issues to find something that interests you.
2. Clone OIP-GT so that you can edit the source code.
3. Make changes to the code / database. See our [Contributing Guidelines](CONTRIBUTING.md) for code requirements and expectations, and maybe for some helpful tips.
4. Submit a pull request.
5. Work with us to answer any questions and make any improvements as we review your pull request.
6. Your contribution gets approved and added to OIP-GT! 🎉

## Maintained by 😎
Contact ✉️:
- Craig Larson (@math1um). Email: clarson@vcu.edu  Web: [http://www.people.vcu.edu/~clarson/]

Current maintainers 🔨🔧🔩:
- Craig Larson (@math1um). Email: clarson@vcu.edu  Web: [http://www.people.vcu.edu/~clarson/](http://www.people.vcu.edu/~clarson/)
- Nico Van Cleemput (@nvcleemp). Email: nico.vancleemput@gmail.com Web: [http://nvcleemp.be/academic/index.html](http://nvcleemp.be/academic/index.html)

Past significant contributions by 👻:
- Justin Yirka (@yirkajk), Summer 2018. Web: [https://www.justinyirka.com/](https://www.justinyirka.com/)
- Reid Barden (@rbarden), Summery 2017. Web: [https://reidbarden.com/](https://reidbarden.com/)
