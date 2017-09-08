## Contributing to [OIP-GT](https://github.com/math1um/objects-invariants-properties)

#### Bugs
Bugs are incorrect results from **THIS** set of files. They could be incorrect codings of graphs, or invariant methods that return incorrect values for one or more graphs. If there is a problem with CONJECTURING, visit their [issues page](https://github.com/nvcleemp/conjecturing/issues) to report it.
* If you find a bug, check the [issues](https://github.com/math1um/objects-invariants-properties/issues) on Github to make sure that no one else has already reported it. If it is already there, go ahead and confirm the bug so that we can verify multiple people have the same issue.
* To report a bug, open a [new issue](https://github.com/math1um/objects-invariants-properties/issues/new). Include a descriptive title and message with as much information as you can:
     * If the bug references a graph, please add some easy invariants/properties to show it doesn't equal what we think it should, and feel free to add a new, accurate coding of the graph following the contributing guidelines. **Never delete a graph's code!** Just because it isn't what we thought it was, doesn't mean we can't use it. A graph is a graph and therefore useful to the program. Just change the comment to make it accurate.
     * If the bug references an invariant, property, or theorem, add references to (a) graph(s) where the returned value is incorrect and show the calculation, if possible, for the correct value. If you can code the correct invariant, please feel free to do so and follow the contributing guidelines.
* Inefficiency of code (naive solutions to invariants, etc.) is not a bug! It's not good, but if it returns the correct value, it is not a bug.

#### Enhancements

* It is very possible that some method or graph construction is not as efficient as it could be. If you have an enhancement to contribute, open a pull request with your enhanced version. Please use the title and description to make very clear the change you made and how it is more efficient code. It may take time for the enhancement to be verified and the PR accepted.

#### Contributing Objects
* Note that during development the project is split over multiple file, but during the build process everything will be compiled into a single file.
* Ensure that the graph doesn't already exist in the `graph.sage` file. A quick `ctrl-f` search should, with luck, give you a go ahead or not. But don't worry if you miss it and code it anyways, you'll catch this in a later step. 
* Check the [issues](https://github.com/math1um/objects-invariants-properties/issues) on Github to make sure someone isn't already contributing your graph. Let people know with a comment on an open issue or by opening a new issue if you are going to add a specific graph. The goal is to not duplicate effort.
* Code the graph, either by using its Graph6 string or by coding it manually. Give it a name. Add it to the appropriate lists using the `add_to_lists` function. Be sure to add a comment above the graph giving a short description of the graph if it is not trivial.
   ```sage
   # CE to independence_number(x) <= residue(x)^(degree_sum(x)^density(x))
   c102 = graphs.CycleGraph(102)
   c102.name(new = "c102")
   add_to_lists(c102, graph_objects, counter_examples)
   ```
* For **ALL OBJECTS**, use the utility function called `does_graph_exist` to make sure the graph, or an isomorphic graph doesn't already exist in the graphs.sage file already. This example returns true.
   ```sage
   graph = graphs.PetersenGraph()
   does_graph_exist(graph, all_graphs)
   ```
* From here you have two options:
   * Open a pull request with the newly coded graph. Make sure the PR description clearly describes what you've done and, if appropriate, includes issue number references.
   * **Preferred** Comment your code in the issue so that someone else may confirm you have it correct. Please use the code blocks using three back ticks before and after your code for all code in an issue comment. From there, follow the first option, noting that it has been confirmed already in the issue.

#### Contributing Invariants, Theorems, and Properties

* Ensure that the method doesn't already exist in the appropriate file. A quick `ctrl-f` search should, with luck, give you a go ahead or not.
* Check the [issues](https://github.com/math1um/objects-invariants-properties/issues) on Github to make sure someone isn't already contributing this method. Let people know with a comment on an open issue or by opening a new issue if you are going to tackle it.
* Code the invariant, theorem, or property. Add a method comment so that others can know what the method does and how to use it. Doctests are very helpful to ensure the method returns the correct value. **REQUIRED** for theorems, and highly suggested for invariants and properties, is a citation to a proof. Unproven theorems will not be accepted.
   * Theorems can and should also be added as invariants.
   ```sage
   def distinct_degrees(g):
    """
    returns the number of distinct degrees of a graph
        sage: distinct_degrees(p4)
        2
        sage: distinct_degrees(k4)
        1
    """
    return len(set(g.degree()))
   ```
   * Like contributing an object, from here you have two options:
   * Open a pull request with the newly coded method. Make sure the PR description clearly describes what you've done and, if appropriate, includes issue number references.
   * **Preferred** Comment your code in the issue so that someone else may confirm you have it correct. Please use the code blocks using three back ticks before andafter your code for all code in an issue comment. From there, follow the first option, noting that it has been confirmed already in the issue.
