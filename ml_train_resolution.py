import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

from utils import preprocessing
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def train():
    df = pd.read_excel('datasets/raw_data.xlsx', engine='openpyxl')
    df['cleanText'] = df['err_msg'].map(lambda x: preprocessing(x))

    cv = CountVectorizer(max_features=1500, min_df=5, max_df=0.7)
    cv.fit(df['cleanText'])

    features = cv.transform(df['cleanText']).toarray()
    labels = df['label']

    models = [
        RandomForestClassifier(n_estimators=100, random_state=0),
        LinearSVC(C=1.0,random_state=2),
        MultinomialNB(),
        LogisticRegression(random_state=0),
        XGBClassifier(n_estimators=100, learning_rate=0.03333, random_state=0)
    ]
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=2)

    CV = 10
    cv_df = pd.DataFrame(index=range(CV * len(models)))
    entries = []
    for model in models:
        model_name = model.__class__.__name__
        for train_index, test_index in skf.split(features, labels):
            X_train, X_test = features[train_index], features[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            model.fit(X_train, y_train)
            predicted = model.predict(X_test)
            f1  = metrics.f1_score(y_test, predicted, average='weighted')
            pre = metrics.precision_score(y_test, predicted, average='weighted')
            re  = metrics.recall_score(y_test, predicted, average='weighted')
            acc = metrics.accuracy_score(y_test, predicted)
            entries.append((model_name, f1, pre, re, acc))
            
    cv_df = pd.DataFrame(entries, columns=['model_name', 'f1', 'pre', 'recall', 'acc'])
    f1_accuracy = cv_df.groupby('model_name').f1.mean()
    precision_accuracy = cv_df.groupby('model_name').pre.mean()
    recall_accuracy = cv_df.groupby('model_name').recall.mean()
    accuracy = cv_df.groupby('model_name').acc.mean()
    # std_accuracy = cv_df.groupby('model_name').accuracy.std()
    acc = pd.concat([f1_accuracy, precision_accuracy, recall_accuracy, accuracy], axis= 1, 
            ignore_index=True)
    acc.columns = ['Mean f1-score', 'Mean Precision', 'Mean Recall', 'Mean Accuracy']
    print(acc)

    plt.figure(figsize=(8,5))
    sns.boxplot(x='model_name', y='f1', 
                data=cv_df, 
                color='lightblue', 
                showmeans=True)
    plt.title("MEAN ACCURACY (cv = {})\n".format(CV), size=14)
    plt.show()


def main():
    train()

if __name__ == '__main__':
    main()