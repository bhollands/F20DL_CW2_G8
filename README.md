F20DL_CW2_G8

Part 1 - Decision trees: Perry & Calum
Part 2 - Neural Networks: Bernard & Ram

What to do:
Before you start: Choose the software in which to conduct the project: we recommend Python for the
part concerning Neural Networks. Create folders on your computer to store classifiers, screenshots
and results of all your experiments, as explained below.

Your experiments will consist of two parts – in Part-1 you will work with Decision trees and in Part -2 –
with Linear Classifiers and Neural Networks.
For each of the two parts, you will do the following:

1. Using the provided training data sets, and the 10-fold cross validation, run the classifier, and note its
accuracy for varying learning parameters. Record all your findings and explain them. Make sure you
understand and can explain logically the meaning of the confusion matrix, as well as the information
contained in the “Detailed Accuracy” field: TP Rate, FP Rate, Precision, Recall, F Measure, ROC
Area.

2. Use Visualization tools to analyze and understand the results.

3. Repeat steps 1 and 2, this time using training and testing data sets instead of the cross validation.
That is, build the classifier using the training data set, and test the classifier using the provided
test data set. Note the accuracy.

4. Make new training and testing sets, by moving 4000 of the instances from the original training set
into the testing set. Then, repeat step 3.

5. Make new training and testing sets again, this time removing 9000 instances from the
original training set and placing them into the testing set again repeat step 3.

6. Analyze your results from the point of view of the problem of classifier over-fitting.
_________________________________________________________________________________

Detailed technical instructions:
Part 1. Decision tree learning.

In this part, you are asked to explore decision tree algorithms:
1. J48 Algorithm
2. One other Decision tree algorithm of your choice (e.g. random forest).

You should compare their relative performance on the given data set. For this:
 Experiment with various decision tree parameters: binary splits or multiple branching, pruning,
confidence threshold for pruning, and the minimal number of instances permissible per leaf.

 Experiment with their relative performance based on the output of confusion matrices as well as
other metrics (TP Rate, FP Rate, Precision, Recall, F Measure, ROC Area). Note that different
algorithms can perform differently on various metrics. Does it happen in your experiments? –
Discuss.

 Record all the above results by going through the steps 1-6.
_________________________________________________________________________________

Part 2. Neural Networks.

In this part, you will work with Neural Networks:
 Run a Linear classifier on the data set. This will be your base for comparison.

 Run a Multilayer Perceptron, experiment with various Neural Network parameters: add or remove
layers, change their sizes, vary the learning rate, epochs and momentum, and validation threshold.

 Experiment with relative performance of Neural Networks and changing parameters. Base your
comparative study on the output of confusion matrices as well as other metrics (TP Rate, FP Rate,
Precision, Recall, F Measure, ROC Area).

 Record all the above results by going through the steps 1-6.

 For higher marks, try running Convolutional Neural Networks, and repeat all of the above steps for
them. 