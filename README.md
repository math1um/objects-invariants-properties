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

##### More information (especially for researchers) 📚📚:

GitHub has proven a valuable tool for this project.

For one, it enables us to share this repository with a growing community of researchers. Ideally, researchers can use this repository to generate and test new conjectures. Then, researchers will contribute their findings either as counterexamples (graphs not previously in OIP-GT) or as theorems (with citations, when a conjecture is proven).

Second, it enables us to recruit students. Some of these students have interests in computer science, rather than math, and this project is a tool to bridge these fields. We have organized several summer workshops, each with a particular focus in graph theory, where students program, conjecture, and prove/disprove as a group. These efforts spur growth in OIP-GT much faster than we could produce alone.

📔 See https://arxiv.org/abs/1801.01814 for a more extensive introduction to the motivations for this project, past workshops, some of the results produced with OIP, and a history of efforts in automated conjecturing.

📔 See https://www.sciencedirect.com/science/article/pii/S0004370215001575 for a description of the CONJECTURING program's algorithm.

## Getting started :bowtie: 🔰

We've written the below instructions so that **anybody** can use OIP-GT. This is a math project. We hope to recruit mathematicians, no matter how programming-phobic they are!

If you have trouble getting started, please contact us (see "Maintained by" at the end). If you find a specific problem that means OIP-GT doesn't work on your system, please submit an Issue (see "Reporting bugs" below).

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
 - GitHub and git :octocat::
   - [GitHub Hello World](https://guides.github.com/activities/hello-world/)
   - [Forking Projects and Pull Requests](https://guides.github.com/activities/forking/)
   - [Mastering issues and working with other developers](https://guides.github.com/features/issues/)
   - [Code Academy](https://www.codecademy.com/learn/learn-git)
 - Terminal / shell / Unix 🐧:
   - [Ryan's Tutorials](https://ryanstutorials.net/linuxtutorial/)
   - [Code Academy](https://www.codecademy.com/learn/learn-the-command-line)

### What computing environment / type of machine are you using? 💻

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
 - Or, [download and install Sage](), and possibly work through a [Sage tour](http://doc.sagemath.org/html/en/a_tour_of_sage/)/[tutorial](http://doc.sagemath.org/html/en/tutorial/). For Windows users, be sure to skim the [additional instructions](https://wiki.sagemath.org/SageWindows).

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

#### Note for users who install OIP-GT by cloning the repository :octocat::

To build the source files, open a terminal and `cd` to the root OIP directory. Then, run `make`. This should create a new directory named `build`. Copy all of the files from `build` into the directory you plan to work in. You can do this by running `cp build/* someDirectory`, where `someDirectory` is wherever you plan to work.

The above process takes the individual files containing graphs / Objects, Properties, Invariants, etc. and builds a single file `gt.sage`. To use OIP-GT, load `gt.sage`. To contribute to OIP-GT, edit the individual files.

## Examples / Tutorial 🎓

The below examples assume that `gt.sage` and the other files you downloaded above are located in your current working directory. If not, then either copy the files, use `cd`, or use `os.chdir("dirName")`.

### Load OIP-GT

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

### Graphs

The graphs in `gt.sage` are all in the list `all_graphs`.

You can find graphs which meet some criteria by running something like
```sage
myGraphs = [g for g in all_graphs if g.order() < 20 and g.is_hamiltonian()] # All Hamiltonian graphs of order less than 20.
myGraphs2 = [g for g in all_graphs if is_two_connected(g)] # All the 2-connected graphs
```
Note, when we say "All the 2-connected graphs", we do not mean "all of the 2-connected graphs in the universe"; we only mean a subset of the graphs that we have programmed into `gt.sage`.

does_graph_exist(g, L)
graph 6 string, and show
Other lists

TO DO

### Properties

TO DO

### Invariants

TO DO

### Conjecturing

Bingos

TO DO

#### Add theorems to conjecturing

TO DO

#### Add precomputed values to conjecturing

TO DO

### Precompute additional values

TO DO

### Where to find more examples and documentation 📜

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

Note: Please don't file an issue to ask a question. See "Maintained by" below for contact information.

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

To contribute, you'll need to be familiar with GitHub pull requests and with programming - although the amount and type of programming may be minimal, depending on the issue. You can use the tutorials linked to above in "Getting Started".

The basic process is (more details in our [Contributing Guidelines](CONTRIBUTING.md)):
 1. Find an issue you'd like to help with. If you have a bug or feature request you'd like to resolve, then you should still begin by following the steps above to create an issue (that way we understand what bug or feature you're resolving!). Otherwise, you should check out the list of issues to find something that interests you.
 2. Clone OIP-GT so that you can edit the source code.
 3. Make changes to the code / database. See our [Contributing Guidelines](CONTRIBUTING.md) for code requirements and expectations, and maybe for some helpful tips.
 4. Submit a pull request.
 5. Work with us to answer any questions and make any improvements as we review your pull request.
 6. Your contribution gets approved and added to OIP-GT! 🎉

## Maintained by 😎
Contact ✉️:
- Craig Larson (@math1um). Email: clarson@vcu.edu  Web: http://www.people.vcu.edu/~clarson/

Current maintainers 🔨🔧🔩:
 - Craig Larson (@math1um). Email: clarson@vcu.edu  Web: http://www.people.vcu.edu/~clarson/
 - Nico Van Cleemput (@nvcleemp).
 - Justin Yirka (@yirkajk), Summer 2018. Web: https://www.justinyirka.com/

Past significant contributions by 👻:
 - Reid Barden (@rbarden), Summery 2017. Web: https://reidbarden.com/
