import re, string
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s):
    return re_tok.sub(r' \1 ', s).split()


class NBTfidfVectorizer(TfidfVectorizer):
    """Class for generating Naive Bayes features with tf-idf priors.
    Can also be used to generate tf-idf only.
    """
    def __init__(self):
        super().__init__(
            ngram_range=(1,2), tokenizer=tokenize,
            min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
            smooth_idf=1, sublinear_tf=1)
        # Nive Bayes parameter
        self._r = None

    def fit(self, X, y):
        """Calculate NB and tf-idf parameters """
        # fit and generate TF-IDF features
        X_tfidf = super().fit_transform(X)

        # get NB features
        p = (X_tfidf[y == 1].sum(0) + 1) / ((y == 1).sum() + 1)
        q = (X_tfidf[y == 0].sum(0) + 1) / ((y == 0).sum() + 1)
        self._r = np.log(p / q)

    def transform(self, X):
        X_tfidf = super().transform(X)
        return X_tfidf.multiply(self._r)

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


class NBLogisticRegression(LogisticRegression, NBTfidfVectorizer):
    def __init__(self):
        self.regressor = LogisticRegression(C=4, dual=True)
        self.vectorizer = NBTfidfVectorizer()

    def fit(self, X, y):
        print('Fitting NBTfidf')
        X_NBTfidf = self.vectorizer.fit_transform(X, y)

        print('Fitting LogisticRegression')
        self.regressor.fit(X_NBTfidf, y)

    def predict_proba(self, X):
        X_NBTfidf = self.vectorizer.transform(X)
        return self.regressor.predict_proba(X_NBTfidf)[:,1]

    def predict(self, X):
        X_NBTfidf = self.vectorizer.transform(X)
        return self.regressor.predict(X_NBTfidf)


if __name__ == '__main__':
    # Code from https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline
    data = pd.read_csv(r'C:\Users\Abdul\Workspace\Kaggle_Toxic\data\train.csv')
    # data = pd.read_csv(r'/u/a_saleh/projects/dialog-eq/datasets/toxic_comments/train.csv')
    data['toxic'] = data['toxic'] + data['insult'] + data['obscene'] + data['severe_toxic'] + data['identity_hate'] + data['threat']
    data['toxic'][data['toxic'] != 0] = 1

    train, test = train_test_split(data, test_size=0.25)

    train['none'] = 1-train['toxic']
    print('{} none labels out of {} comments'.format(train['none'].sum(), train.shape[0]))
    print('so {} of the comments are non toxic'.format(train['none'].sum() / train.shape[0]))

    COMMENT = 'comment_text'
    train[COMMENT].fillna("<unk>", inplace=True)
    test[COMMENT].fillna("<unk>", inplace=True)

    logistic = NBLogisticRegression()
    logistic.fit(train[COMMENT], train['toxic'].values)

    train_preds = logistic.predict(train[COMMENT])
    test_preds = logistic.predict(test[COMMENT])

    print('Train accuracy is: {:.3f}'.format(accuracy_score(train['toxic'], train_preds)))
    print('Train recall (True positive) is {:.3f}'.format(recall_score(train['toxic'], train_preds)))
    print('Train precision is {:.3f}'.format(precision_score(train['toxic'], train_preds)))
    print('Train F1 is {:3f}'.format(f1_score(train['toxic'], train_preds)))

    print('*' * 20)
    print('*' * 20)
    print('Test accuracy is: {:.3f}'.format(accuracy_score(test['toxic'], test_preds)))
    print('Test recall (True positive) is {:.3f}'.format(recall_score(test['toxic'], test_preds)))
    print('Test precision is {:.3f}'.format(precision_score(test['toxic'], test_preds)))
    print('Test F1 is {:3f}'.format(f1_score(test['toxic'], test_preds)))

    print('#' * 20)
    print('#' * 20)
    print('Training model on full data')
    logistic = NBLogisticRegression()
    logistic.fit(data[COMMENT], data['toxic'].values)

    print('Saving trained toxicity model')
    with open('toxicity_model.pkl', 'wb') as f:
        pickle.dump(logistic, f)
