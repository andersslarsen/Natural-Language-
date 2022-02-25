import pandas as pd
import numpy as np
import pickle
import spacy
import sys
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

NAMEOFTESTFILE = sys.argv[1]
NAMEOFTHETRAINFILE = sys.argv[2]

nlp = spacy.load("en_core_web_sm")
pd.set_option('display.max_columns', None)

np.random.seed(500)
Corpus = pd.read_csv(NAMEOFTHETRAINFILE, delimiter = "\t", names=['Label', 'Q', 'A'])
Corpus_test = pd.read_csv(NAMEOFTESTFILE, delimiter = "\t", header=None)

#Fixing format of dev set, i.e. adding new column
#for every NaN element, and then dropping the column
Corpus_test.columns = ['Label', 'Q', 'A', 'NaN']
Corpus_test.drop('NaN', axis=1)
Corpus_test.reset_index(drop=True)
Corpus_test.to_csv("test4.csv")

Corpus['Q'] = [entry.lower() for entry in Corpus['Q']]
Corpus_test['Q'] = [entry.lower() for entry in Corpus_test['Q']]
stop = STOP_WORDS

def pre_processing(text):
    doc = nlp(text)
    proc = [token.lemma_ for token in doc if token.is_stop==False and token.text.isalpha()==True]
    return str(proc)

Corpus['Q_final'] = Corpus['Q'].map(pre_processing)
Corpus_test['Q_final'] = Corpus_test['Q'].map(pre_processing)

x_train = Corpus['Q_final']
y_train = Corpus['Label']

x_test = Corpus_test['Q_final']
y_test = Corpus_test['Label']

Encoder = LabelEncoder()
Encoder.fit(y_train)
y_train = Encoder.transform(y_train)
y_test = Encoder.transform(y_test)

Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(Corpus['Q_final'])

Train_X_Tfidf = Tfidf_vect.transform(x_train)
Test_X_Tfidf = Tfidf_vect.transform(x_test)

def main():
    # Classifier - Algorithm - SVM
    # fit the training dataset on the classifier
    SVM = svm.SVC(C=10, kernel='rbf', degree=3, gamma=1)
    SVM.fit(Train_X_Tfidf, y_train)
    # predict the labels on validation dataset
    predictions_SVM = SVM.predict(Test_X_Tfidf)
    decode_pred_svm = Encoder.inverse_transform(predictions_SVM)
    # Use accuracy_score function to get the accuracy
    print("SVM Accuracy Score -> ", accuracy_score(predictions_SVM, y_test) * 100)
    print(confusion_matrix(y_test, predictions_SVM))
    #np.savetxt("results_SVM.csv", decode_pred_svm, fmt='%s')

    # Classifier - Algorithm - Naive Bayes
    # fit the training dataset on the classifier
    Naive = naive_bayes.ComplementNB()
    Naive.fit(Train_X_Tfidf, y_train)
    # predict the labels on validation dataset
    predictions_NB = Naive.predict(Test_X_Tfidf)
    decode_pred_nb = Encoder.inverse_transform(predictions_NB)
    # Use accuracy_score function to get the accuracy
    print("Naive Bayes Accuracy Score -> ", accuracy_score(predictions_NB, y_test) * 100)
    print(confusion_matrix(y_test, predictions_NB))
    np.savetxt("dev_results.csv", decode_pred_nb, fmt='%s')

    # Classifier - Algorithm - Random Forest
    # fit the training dataset on the classifier
    rf = RandomForestClassifier(n_estimators=500, min_samples_split = 20, max_features = 5)
    rf.fit(Train_X_Tfidf, y_train)
    # predict the labels on validation dataset
    predictions_RF = rf.predict(Test_X_Tfidf)
    decode_pred_rf = Encoder.inverse_transform(predictions_RF)
    # Use accuracy_score function to get the accuracy
    print("Random Forest Accuracy Score -> ", accuracy_score(predictions_RF, y_test) * 100)
    print(confusion_matrix(y_test, predictions_RF))

    # saving the models to disk
    filename = 'svm_trained.sav'
    pickle.dump(SVM, open(filename, 'wb'))

    filename = 'nb_trained.sav'
    pickle.dump(Naive, open(filename, 'wb'))

    filename = 'rf_trained.sav'
    pickle.dump(rf, open(filename, 'wb'))

if __name__ == "__main__":
  main()
