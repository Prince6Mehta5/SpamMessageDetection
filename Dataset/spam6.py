import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('Dataset\spam.tsv', sep="\t")

hamDF = df[df['label'] == "ham"]
spamDF = df[df['label'] == "spam"]
#print(hamDF.shape)
#print(spamDF.shape)
hamDF = hamDF.sample(spamDF.shape[0])

finalDF = pd.concat([hamDF, spamDF], ignore_index=True)
X_train, X_test, Y_train, Y_test = train_test_split(finalDF['message'], finalDF['label'], test_size = 0.2, random_state = 0, shuffle = True, stratify = finalDF['label'])

#pipeline

model = Pipeline([('tfidf', TfidfVectorizer()), ('model', SVC(C = 1000, gamma = 'auto'))])
#model = Pipeline([('tfidf', TfidfVectorizer()), ('model', RandomForestClassifier(n_estimators=100, n_jobs=-1))])
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
#print(confusion_matrix(Y_test, Y_pred))
#print(classification_report(Y_test, Y_pred))
#print(accuracy_score(Y_test, Y_pred))
print(accuracy_score(Y_test, Y_pred))
#print(model.predict(["You’ve been overcharged for your 2021 taxes. Get your IRS tax refund here: [Link]"]))

joblib.dump(model, "mySVC Model.pkl")