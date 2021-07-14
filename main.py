import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle


def read_data(filename):
    data = pd.read_pickle(filename)
    return data

df = pd.read_csv('/content/data/train.csv')
df.to_pickle('/content/data/train.pkl')
df = pd.read_csv('/content/data/test.csv')
df.to_pickle('/content/data/test.pkl')
train = read_data('/content/data/train.pkl')
test = read_data('/content/data/test.pkl')
train.describe()
train.head()

from sklearn.model_selection import train_test_split
train, validation = train_test_split(train, random_state=42, test_size=0.1, shuffle=True)


def count_labels_per_category(df):
    df_toxic = df.drop(['id', 'comment_text'], axis=1)
    counts = []
    categories = list(df_toxic.columns.values)
    for i in categories:
        counts.append((i, df_toxic[i].sum()))
    df_stats = pd.DataFrame(counts, columns=['category', 'number_of_comments'])
    return df_stats

df_stats_train = count_labels_per_category(train)
df_stats_train

def plot_count_labels_per_category(df_stats):
    df_stats.plot(x='category', y='number_of_comments', kind='bar', legend=False, grid=True, figsize=(8, 5))
    plt.title("Number of comments per category")
    plt.ylabel('# of Occurrences', fontsize=12)
    plt.xlabel('category', fontsize=12)
    plot_count_labels_per_category(df_stats_train)


X_train, y_train = train['comment_text'].values, train.iloc[:,2:].values
X_val, y_val = validation['comment_text'].values, validation.iloc[:,2:].values
X_test, y_test = test['comment_text'].values, test.iloc[:,2:].values


def convertClass(tags, classes):
    result = []
    for i, tag in enumerate(tags):
        if tag > 0:
            result.append(classes[i])
    if len(result) == 0:
        result.append('safe')

    return result


y_train = np.array([convertClass(tag,classes) for tag in y_train])
y_val = np.array([convertClass(tag,classes) for tag in y_val])
y_test = np.array([convertClass(tag,classes) for tag in y_test])

import re

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
NEW_LINE = re.compile('\n')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))


def text_prepare(text):
    """
        text: a string

        return: modified initial string
    """
    text = text.lower()  # lowercase text
    text = NEW_LINE.sub(' ', text)  # replace NEW_LINE symbols in our texts by space
    text = REPLACE_BY_SPACE_RE.sub(' ', text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text)  # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join([word for word in text.split() if word not in STOPWORDS])
    return text

tags_counts = {}
for tags in y_train:
    for tag in tags:
        if tag in tags_counts:
            tags_counts[tag] += 1
        else:
            tags_counts[tag] = 1
# Dictionary of all words from train corpus with their counts.
words_counts = {}

for title in X_train:
    for word in title.split():
        if word in words_counts:
            words_counts[word] += 1
        else:
            words_counts[word] = 1



most_common_tags = sorted(tags_counts.items(), key=lambda x: x[1], reverse=True)[:5]
print('Common tags',most_common_tags)
print('-----------------')
most_common_words = sorted(words_counts.items(), key=lambda x: x[1], reverse=True)[:5]
print('common words',most_common_words)

from sklearn.feature_extraction.text import TfidfVectorizer


def tfidf_features(X_train, X_val, X_test):
    """
        X_train, X_val, X_test — samples
        return TF-IDF vectorized representation of each sample and vocabulary
    """
    # Create TF-IDF vectorizer with a proper parameters choice
    # Fit the vectorizer on the train set
    # Transform the train, test, and val sets and return the result

    tfidf_vectorizer = TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(1, 2), token_pattern='(\S+)')
    tfidf_vectorizer.fit(X_train)
    X_train = tfidf_vectorizer.transform(X_train)
    X_val = tfidf_vectorizer.transform(X_val)
    X_test = tfidf_vectorizer.transform(X_test)
    return X_train, X_val, X_test, tfidf_vectorizer.vocabulary_

X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf_vocab = tfidf_features(X_train, X_val, X_test)
tfidf_reversed_vocab = {i:word for word,i in tfidf_vocab.items()}


from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer(classes=sorted(tags_counts.keys()))
y_train = mlb.fit_transform(y_train)
y_val = mlb.fit_transform(y_val)
#y_test = mlb.fit_transform(y_test)

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier


def train_classifier(X_train, y_train):
    """
      X_train, y_train — training data

      return: trained classifier
    """

    # Create and fit LogisticRegression wraped into OneVsRestClassifier.

    lr = LogisticRegression(C=4.0, penalty='l2')  # use L2 to optimise
    ovr = OneVsRestClassifier(lr)
    ovr.fit(X_train, y_train)
    return ovr


classifier_tfidf = train_classifier(X_train_tfidf, y_train)
y_train_predicted_labels_tfidf = classifier_tfidf.predict(X_train_tfidf)
y_train_predicted_scores_tfidf = classifier_tfidf.decision_function(X_train_tfidf)
y_train_pred_inversed = mlb.inverse_transform(y_train_predicted_labels_tfidf)
y_train_inversed = mlb.inverse_transform(y_train)
for i, text in enumerate(X_train[20:26]):
    print('Title:\t{}\nTrue labels:\t{}\nPredicted labels:\t{}\n\n'.format(
        text,
        ','.join(y_train_inversed[20:26][i]),
        ','.join(y_train_pred_inversed[20:26][i])
    ))


y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)
y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(X_val_tfidf)
y_val_pred_inversed = mlb.inverse_transform(y_val_predicted_labels_tfidf)
y_val_inversed = mlb.inverse_transform(y_val)
for i,text in enumerate(X_val[84:90]):
    print('Title:\t{}\nTrue labels:\t{}\nPredicted labels:\t{}\n\n'.format(
        text,
        ','.join(y_val_inversed[84:90][i]),
        ','.join(y_val_pred_inversed[84:90][i])
    ))


from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
def print_evaluation_scores(y_val, predicted):

    print('Accuracy:',accuracy_score(y_val, predicted))
    print('F1 Score:',f1_score(y_val, predicted, average='weighted'))
    print('Precision:',average_precision_score(y_val, predicted,average='weighted'))
    print('Recall:',recall_score(y_val, predicted,average='weighted'))


def print_words_for_tag(classifier, tag, tags_classes, index_to_words, all_words):
    """
        classifier: trained classifier
        tag: particular tag
        tags_classes: a list of classes names from MultiLabelBinarizer
        index_to_words: index_to_words transformation
        all_words: all words in the dictionary

        return nothing, just print top 5 positive and top 5 negative words for current tag
    """
    print('Tag:\t{}'.format(tag))
    # Extract an estimator from the classifier for the given tag.
    # Extract feature coefficients from the estimator.
    est = classifier.estimators_[tags_classes.index(tag)]
    top_positive_words = [index_to_words[index] for index in est.coef_.argsort().tolist()[0][-5:]]
    top_negative_words = [index_to_words[index] for index in est.coef_.argsort().tolist()[0][:5]]
    print('Top positive words:\t{}'.format(', '.join(top_positive_words)))
    print('Top negative words:\t{}\n'.format(', '.join(top_negative_words)))


DICT_SIZE = len(words_counts)
WORDS_TO_INDEX = {word[0]:i for i,word in enumerate(sorted(words_counts.items(), key=lambda x: x[1], reverse=True)[:DICT_SIZE])}
INDEX_TO_WORDS = {WORDS_TO_INDEX[i]:i for i in WORDS_TO_INDEX}####### YOUR CODE HERE #######
####### YOUR CODE HERE #######
ALL_WORDS = WORDS_TO_INDEX.keys()


print_words_for_tag(classifier_tfidf, 'safe', mlb.classes, tfidf_reversed_vocab, ALL_WORDS)
print_words_for_tag(classifier_tfidf, 'identity_hate', mlb.classes, tfidf_reversed_vocab, ALL_WORDS)
print_words_for_tag(classifier_tfidf, 'insult', mlb.classes, tfidf_reversed_vocab, ALL_WORDS)
print_words_for_tag(classifier_tfidf, 'obscene', mlb.classes, tfidf_reversed_vocab, ALL_WORDS)
print_words_for_tag(classifier_tfidf, 'severe_toxic', mlb.classes, tfidf_reversed_vocab, ALL_WORDS)
print_words_for_tag(classifier_tfidf, 'threat', mlb.classes, tfidf_reversed_vocab, ALL_WORDS)
print_words_for_tag(classifier_tfidf, 'toxic', mlb.classes, tfidf_reversed_vocab, ALL_WORDS)



filename = 'finalmodel.sav'
pickle.dump(classifier_tfidf, open(filename, 'wb'))

