# Natural-Language-
Assignment: Question classification - in the course Natural Language from IST - University of Lisboa

Implementation of three classifiers: Support Vector Machine, Naive Bayes and Random forrest. The classifiers are used in a question classification problem. 
qc.py loads the optimal models created and trained in training.py and makes predictions on a test set. The same preprocessing is used in both scripts, i.e. on both the training and on the dev/test sets. Use the command python qc.py TESTFILE TRAINFILE, where TESTFILE and TRAINFILE is e.g., test.txt and trainWithoutDev.txt respectively, to produce test_results.csv.![image](https://user-images.githubusercontent.com/48654042/155713488-3d7e7a13-28bc-49c9-883f-566e25ac6bd6.png)
