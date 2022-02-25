import pandas as pd
import numpy as np
import pickle
import spacy
import sys
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
#from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

NAMEOFTESTFILE = sys.argv[1]
NAMEOFTHETRAINFILE = sys.argv[2]

nlp = spacy.load("en_core_web_sm")
pd.set_option('display.max_columns', None)

df_test = pd.read_csv(NAMEOFTESTFILE, delimiter = "\t", names=['Q', 'A'])
Corpus = pd.read_csv(NAMEOFTHETRAINFILE, delimiter = "\t", names=['Label', 'Q', 'A'])

Corpus['Q'] = [entry.lower() for entry in Corpus['Q']]
df_test['Q'] = [entry.lower() for entry in df_test['Q']]
stop = STOP_WORDS

def pre_processing(text):
    doc = nlp(text)
    proc = [token.lemma_ for token in doc if token.is_stop==False and token.text.isalpha()==True]
    return str(proc)

Corpus['Q_final'] = Corpus['Q'].map(pre_processing)
df_test['Q_final'] = df_test['Q'].map(pre_processing)

x_train = Corpus['Q_final']
y_train = Corpus['Label']

x_test = df_test['Q_final']

Encoder = LabelEncoder()
Encoder.fit(y_train)
y_train = Encoder.transform(y_train)

Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(Corpus['Q_final'])

Train_X_Tfidf = Tfidf_vect.transform(x_train)
Test_X_Tfidf = Tfidf_vect.transform(x_test)

def main():
    # Loading models
    SVM = pickle.load(open('svm_trained.sav', 'rb'))
    Naive = pickle.load(open('nb_trained.sav', 'rb'))
    rf = pickle.load(open('rf_trained.sav', 'rb'))

    prediction_SVM = SVM.predict(Test_X_Tfidf)
    decode_pred_svm = Encoder.inverse_transform(prediction_SVM)
    np.savetxt("resultsSVM.csv", decode_pred_svm, fmt='%s')

    prediction_Naive = Naive.predict(Test_X_Tfidf)
    decode_pred_nb = Encoder.inverse_transform(prediction_Naive)
    np.savetxt("resultsNB.csv", decode_pred_nb, fmt='%s')

    prediction_RF = rf.predict(Test_X_Tfidf)
    decode_pred_rf = Encoder.inverse_transform(prediction_RF)
    np.savetxt("resultsRF.csv", decode_pred_rf, fmt='%s')

if __name__ == "__main__":
  main()
