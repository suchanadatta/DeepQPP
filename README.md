# DeepQPP
This is a pairwise interaction based Deep Learning Model for supervised query performance prediction. The entire model is comprised of two modules - 1. **InteractionMatrix** (developed with Java) and 2. **DeepQPP** (written in Python) 

## Requirements
To run the DeepQPP model, just check if your conda environment is good with the following packages. For more details please go through **requirements.txt**
|                |
|----------------|
|Numpy 1.19.4|
|Keras 2.3.0|
|Tensorflow 2.2.0|
|Scikit-learn 0.23.2|

## Guide to use
**Step-1:** Create a [conda environment](https://phoenixnap.com/kb/how-to-install-anaconda-ubuntu-18-04-or-20-04) and activate it by - 
> conda activate <environment_name>

**Step-2:** Check all the packages listed above using correct version of your pip -
> pip list

In case any package is missing, install the right version in your current conda environment.

There is a top level bash script **main.sh**. Firstly, it runs the InteractionMatrix module to generate matching histograms of pseudo-relevant documents as proposed in the paper : [A Deep Relevance Matching Model for Ad-hoc Retrieval](https://dl.acm.org/doi/10.1145/2983323.2983769). It computes Log-IDF based histograms for a document with respect to a given query. This is built on top of LCH(Log-Count-based Histogram); LCH(with IDF) performs the best as reported in the paper. Given a query, interaction matrices computed for the set of respective relevant documents are stored in a single file with the name **query_id.hist**.

**Step-3:** Provide all arguments in order to run the bash script **interaction.sh** in main.sh. Following arguments should be given :
``````````````````````````````````````````````````````````````````````````````````````````
> Query file path (.xml file)
> Path of the Lucene index
> Stopwords file path
> Initial retrieved documents file (pseudo-relevant documents)
> Word vector file path
> Name of the field used for searching
> Interaction matrix path (where matrices will be stored)
``````````````````````````````````````````````````````````````````````````````````````````

Next, supervised deepQPP module is trained by a set of query pairs' relative specificity computed through **query_pair_judgement.py**. We train the model with paired data and tested with both paired and point test set. K-fold cross validation is used to test model's efficiency. 

**Step-4:** Following arguments to be given in order to run the bash script **qppeval.sh** through main.sh. Check if arguments below are set in main.sh -
``````````````````````````````````````````````````````````````````````````````````````````
> Path of the AP file
> Path of the interaction matrix (separate file for each qid)
> Training batch size
> No. of epochs
> No. of cross validation folds
> Type of evaluation : pair/point
> Want to save predicted values? -- yes/no
``````````````````````````````````````````````````````````````````````````````````````````



