# aisier
`aisier` (from the words AI and easier) is a command line tool that makes machine learning (ML) projects management that use [TensorFlow](https://www.tensorflow.org/) easier. Every time we need to start a new ML project, we repeat the same operations and we re-create approximately the same folder-tree and the same code structure. These operations can be automatized: this is why **aisier**.

**aisier is useful to:**
* have a standard structure across all your ML projects;
* create a project, create a CSV dataset, optimize it and train a model in a fast and easy way;
* analyze the features in the dataset determining which features affect more the accuracy of the model;
* plot training history and performance of the model;

### Install

    git clone https://github.com/pagiux/aisier.git
    cd aisier
    python setup.py build
    sudo python setup.py install
  
### Usage

    aisier <cmd> -h
To get a clue about command parameters.

    aisier init <name>
To create a <name> project structure. Inside the file `model.py`, generated by the _init_ command, there is the definition of the model and its training, using [TensorFlow Functional API](https://www.tensorflow.org/guide/keras/functional). Inside the file `prepare.py`, generated by the _init_ command, there is the definition of the dataset pre-processing procedure.

    aisier prepare <name> <path>
To process the files in <path> using the procedure defined inside the file `prepare.py`.
  
    aisier optimize-dataset <name>
To remove duplicates in the dataset. Other kind of optimizations could be implemented in future.

    aisier train <name>
To train the model as defined inside the file `model.py`

    aisier analyze <name>
To analyze the features in the dataset, using PCA, Chi2, Correlation matrix (Pearson correlation), Lasso feature selection, recursive feature elimination and so on.

    aisier view <name>
To plot model performance and training history.

### Example

- A [behavioral ransomware detector](https://github.com/pagiux/aisier-ransomware-detector) from [IRP logs](https://github.com/pagiux/IRPLogger).

### Credits

I would really like to thank [evilsocket](https://github.com/evilsocket) and its [ergo](https://github.com/evilsocket/ergo) for the inspiration and the ideas that made the development of this tool possible. 

### License

`aisier` was made with ♥ and it is released under the GPL 3 license.
