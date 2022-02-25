# Natural-Language-
Assignment: Question classification - in the course Natural Language from IST - University of Lisboa

Implementation of three classifiers: Support Vector Machine, Naive Bayes and Random forrest. The classifiers are used in a question classification problem. The dataset 
qc.py loads the optimal models created and trained in training.py and makes predictions on a test set. The same preprocessing is used in both scripts, i.e. on both the training and on the dev/test sets. Use the command python qc.py TESTFILE TRAINFILE, where TESTFILE and TRAINFILE is e.g., test.txt and trainWithoutDev.txt respectively, to produce test_results.csv.

Models & Preprocessing:

Several techniques were performed in terms of pre-processing. First all symbols that are not letters were removed, and every letter is converted to lowercase. A check is also performed to remove all words that are not a real word. The next step was to remove all the stop words. SpaCy is a third-party library which contains the most common stop words in the English language. The process of tokenization was applied to convert each sentence to a list of letters. Afterwards, each word went through a lemmatization algorithm provided by SpaCy. Lemmatization refers to reducing each word into its lemma, e.g., “better” becomes “good”.

The algorithms tested are Naive Bayes, Random Forest (RF), and Support Vector Machine (SVM). We obtained the best results with the Naive Bayes, but the differences were marginal. 

Experimental Setup & Results:

Tuning of hyperparameters is a time consuming task - especially when working with three different classifiers. In addition to time consumption, changing the parameters manually also made it hard to keep track of the values already tried. A solution to this problem was using sklearn’s library function GridSearchCV. This function iterates through predefined hyperparameters and tries to fit the model to the training data - and in the end - produces the parameters best fit for the model with regards to our evaluation metric, accuracy. 

Results: 

Classification accuracy alone could be misleading if we have more than two classes in our dataset and an unequal number of instances of each class. The dataset used for training our models have an unequal distribution of class instances. Confusion matrices can be used in order to get a better understanding of which errors the model is making. Therefore, confusion matrices were calculated using the built-in function from the package sklearn.metrics. 

![image](https://user-images.githubusercontent.com/48654042/155714879-2ba002c9-6166-4516-bcd9-1d932b417fea.png)

![image](https://user-images.githubusercontent.com/48654042/155714919-19dc166f-25f2-4007-8769-41b8eb6c5a74.png)

![image](https://user-images.githubusercontent.com/48654042/155714944-f49c4342-9723-44c3-9736-a7f90b5ef995.png)

![image](https://user-images.githubusercontent.com/48654042/155714958-908a03c1-d820-421d-b8bf-c14d9558b688.png)

The result file from the given testing set on the 10th of November will be performed with the complement Naïve Bayes classifier. As we got the best accuracy with this algorithm (even though all three gave decent results), this is our algorithm of choice. The pre-processing steps will remain the same.

Error Analysis: 

As there is a great deal of overlap between the categories, it is not expected that a model can achieve upwards of 100% accuracy. A trend for all the methods is that history is often mis-predicted as geography. There are far more examples of history, and the differences between the classes may be too subtle for machine learning.  An example we have seen from the results is that e.g., sentences with label history mentions a city or direction, and is thus predicted as geography. An example is the sentence: “The Rosebuds of this Northwest city were the first pro hockey team in the US”. Here we see that both direction, country and city is mentioned, making it hard to differentiate the classes. Another example is “In 1770 Capt. James Cook became the first European to sight Australia’s fertile east coast, which he named “New this””. 

