import os
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import re
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from sklearn import utils
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

nltk.download('punkt')
nltk.download('stopwords')

# Get crawled data
data = pd.read_csv(os.path.join(os.getcwd(), 'data/arxiv-feb-2020.csv'),
                   encoding='latin1')

titles = data['title']
texts = data['abstract']
labels = data['subject']

# Preprocess: cleaning & normalizing
preprocessed_texts = []
stop_words = set(stopwords.words('english'))

for text, title in zip(texts, titles):
    words_in_text = []

    sents = sent_tokenize(text)
    sents.append(title)
    
    norm_sents = []
    for sent in sents:
        sent = re.sub(r"[^a-z0-9-]+", " ", sent.lower())
        norm_sents.append(sent)
    for sent in norm_sents:
        words = word_tokenize(sent)
        for word in words:
            if (word not in stop_words) and (len(word) > 2):
                words_in_text.append(word)
    
    preprocessed_texts.append(words_in_text)

# Indexing cs-relavant subjects
subjects = ['cs.AI', 'cs.CL', 'cs.CC', 'cs.CE', 'cs.CG',
            'cs.GT', 'cs.CV', 'cs.CY', 'cs.CR', 'cs.DS',
            'cs.DB', 'cs.DL', 'cs.DM', 'cs.DC', 'cs.ET',
            'cs.FL', 'cs.GL', 'cs.GR', 'cs.AR', 'cs.HC',
            'cs.IR', 'cs.IT', 'cs.LO', 'cs.LG', 'cs.MS',
            'cs.MA', 'cs.MM', 'cs.NI', 'cs.NE', 'cs.NA',
            'cs.OS', 'cs.OH', 'cs.PF', 'cs.PL', 'cs.RO',
            'cs.SI', 'cs.SE', 'cs.SD', 'cs.SC', 'cs.SY']

subjects_index = {}
for i, subject in enumerate(subjects):
    subjects_index[subject] = i

# Make train/test data in TaggedDocument() format
train_data = []
test_data = []

for i, text in enumerate(preprocessed_texts):
    if i < 4500:
        train_data.append(
            TaggedDocument(words=text,
                           tags=[subjects_index.get(labels[i], 40)])
        )
    else:
        test_data.append(
            TaggedDocument(words=text,
                           tags=[subjects_index.get(labels[i], 40)])
        )

# Train and save Dov2Vec model
model = Doc2Vec(dm=1,
                vector_size=100,
                negative=5,
                hs=0,
                min_count=3,
                sample=0,
                workers=4,
                alpha=0.025,
                min_alpha=0.001)

model.build_vocab([x for x in train_data])

train_data = utils.shuffle(train_data)
model.train(train_data,
            total_examples=len(train_data),
            epochs=30)

model.save(os.path.join(os.getcwd(), 'model/model-feb-2020.d2v'))

# Get x_train, y_train, x_test, and y_test
def vector_for_learning(model, input_docs):
    sents = input_docs
    targets, feature_vectors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, feature_vectors

y_train, x_train = vector_for_learning(model, train_data)
y_test, x_test = vector_for_learning(model, test_data)

# 6 ML models (classifier)
logreg = LogisticRegression(n_jobs=1, C=1e5)
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)

print('LogReg accuracy: {}'.format(accuracy_score(y_test, y_pred)))
print('LogReg f1-score: {}'.format(f1_score(y_test, y_pred, average='weighted')))

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

print('KNN accuracy: {}'.format(accuracy_score(y_test, y_pred)))
print('KNN f1-score: {}'.format(f1_score(y_test, y_pred, average='weighted')))

dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)

print('Decision Tree accuracy: {}'.format(accuracy_score(y_test, y_pred)))
print('Decision Tree f1-score: {}'.format(f1_score(y_test, y_pred, average='weighted')))

rf = RandomForestClassifier(n_estimators=10)
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)

print('Random Forest accuracy: {}'.format(accuracy_score(y_test, y_pred)))
print('Random Forest f1-score: {}'.format(f1_score(y_test, y_pred, average='weighted')))

gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_pred = gnb.predict(x_test)

print('Naive Bayes accuracy: {}'.format(accuracy_score(y_test, y_pred)))
print('Naive Bayes f1-score: {}'.format(f1_score(y_test, y_pred, average='weighted')))

svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)

print('SVM accuracy: {}'.format(accuracy_score(y_test, y_pred)))
print('SVM f1-score: {}'.format(f1_score(y_test, y_pred, average='weighted')))