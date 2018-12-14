# CS229
CS229 Project: Analysis of Code Submissions in Competitive Programming Contests

by Wenli Looi

## Description of files

* dl.py: Custom scraper to download contest submissions from Codeforces for 10 contests. Maintains a cache so that the downloading can be easily resumed.
* clang.jl: Processes the downloaded submissions using Clang to generate features.
* clang.sh: Shell script to run 4 parallel instances of clang.jl to process the data faster. (Clang seems single-threaded.)
* tokensproc.jl: Processes the output of clang.jl to extract bigrams and perform normalization/scaling.
* learnlib.jl: Library functions related to reading data, printing data stats, processing Clang tokens, training models with ScikitLearn.jl, converting ratings to ranks, computing weights, evaluating models.
* learnlibtf.jl: Library functions related to training and evaluating models (logistic regression and neural network) with TensorFlow.
* learn.jl: Learns TensorFlow models and writes models to a file.
* learntopbigrams.jl: Interprets GDA model to find the most common unigrams/bigrams for a specific class.
* learncv.jl: Generates stats for 10-fold cross validation. Uses TensorFlow models written to the file. The linear regression and GDA models are trained on-demand (as they are fast to train).
