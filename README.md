# Natural-Language-
Assignment: Question classification - in the course Natural Language from IST - University of Lisboa

Implementation of three classifiers: Support Vector Machine, Naive Bayes and Random forrest. The classifiers are used in a question classification problem. 
qc.py loads the optimal models created and trained in training.py and makes predictions on a test set. The same preprocessing is used in both scripts, i.e. on both the training and on the dev/test sets. Use the command python qc.py TESTFILE TRAINFILE, where TESTFILE and TRAINFILE is e.g., test.txt and trainWithoutDev.txt respectively, to produce test_results.csv.
Models & Preprocessing
Several techniques were performed in terms of pre-processing. The techniques will briefly be explained in the following section. First all symbols that are not letters were removed, and every letter is converted to lowercase. A check is also performed to remove all words that are not a real word. The next step was to remove all the stop words. SpaCy is a third-party library which contains the most common stop words in the English language. The process of tokenization was applied to convert each sentence to a list of letters. Afterwards, each word went through a lemmatization algorithm provided by SpaCy. Lemmatization refers to reducing each word into its lemma, e.g., “better” becomes “good”.
The algorithms tested are Naive Bayes, Random Forest (RF), and Support Vector Machine (SVM). We obtained the best results with the Naive Bayes, but the differences were marginal. A brief description of the method will be presented.

Naive Bayes is a probabilistic learning algorithm based upon the Bayes rule P(y│x)=(P(x│y)p(y))/(P(x)), where y denotes the class, and x∈[x_1,x_2,…x_n] is a feature vector consisting of a sample from the data set. The bayes rule can thus be written as p(y│x_1,x_2,…,x_n )=  (P(x_1│y)P(x_2│y)…p(x_n│y)p(y))/(P(x_1 )P(x_2 )…P(x_n ) ). The denominator does not change, hence we can rewrite the equation as p(y│x)∝p(y) Π_(i=1 )^n p(x_i│y). The most probable class is the one with the highest likelihood, i.e., Naïve Bayes can be written as 

y ̂=argmax┬y⁡〖p(y) 〖 Π〗_(i=1 )^n p(x_i│y)〗
The model we used is called complement Naïve Bayes, which flips the objective by minimizing the inverse argument instead, i.e., 
y ̂=argmin┬y⁡〖 p(y)  Π_(i=1)^n  1/(p(x_i│y) )〗
Instead of calculating the likelihood of a word occurring in a class, rather the likelihood of the word occurring in other classes is calculated. The method has found to perform better with imbalanced data sets (Rennie et al, 2003). For a description of Random Forrest or SVM, the interested reader is guided to introduction papers (Ho, T. K. ,1995) and (Cortes, C., & Vapnik, V, 1995)


