
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score

from scipy.stats import itemfreq
from scipy import interp

from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

'''
plt.style.use('seaborn-deep')
matplotlib.style.use('ggplot')

%matplotlib inline
'''

# Dataset
df = pd.read_csv('data/churn.csv')

# Preprocessing: object to boolean
df["Int'l Plan"] = df["Int'l Plan"] == 'yes'
df['VMail Plan'] = df['VMail Plan'] == 'yes'
df['Churn?'] = df['Churn?'] == 'True.'

# Feature selection: continues and booleans
df = df.drop(['State', 'Area Code', 'Phone'], axis=1)

# Classes
y = df.pop('Churn?').values

# Feature names
feature_names = df.columns

# Features
X = df.values

# Train and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y)

# model: Skelearn - RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_hat = model.predict(X_test)


# Accuracy score on Test dataset
print 'Accuracy score', model.score(X_test, y_test)

# Confusion matrix
print "Confusion matrix"
print 'Columns:  Predictions:  0 | 1'
print 'Rows: True 0 | True 1'
print confusion_matrix(y_test, y_hat)

# Precision - Recall
print 'Precision: ',precision_score(y_test, y_hat)
print 'Recall: ', recall_score(y_test, y_hat)

# model improvement: Out of bag parameter = True, number of trees = 33
model = RandomForestClassifier(n_estimators=33, oob_score=True)
model.fit(X_train, y_train)

print 'Accuracy score:', model.score(X_test, y_hat)
print 'Out of bag score:', model.oob_score_

# Feature importance: sklearn, Feature indexes
feature_importance = pd.Series(model.feature_importances_, index=feature_names)
feature_importance = feature_importance.sort_values(ascending=False)
feature_importance_ = feature_importance.sort_values()
feature_indexes = np.argsort(feature_importance)

featureimportances_top5 = feature_importance.sort_values(ascending=False).head(5)


print ' Top 5 - Feature importances', featureimportances_top5


# Standard deviation for feature importances across all trees
std_dev = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)


# Feature score


# Plot  - Forest feature importances
plt.figure()
feature_importance_.plot(kind='barh', xerr= std_dev[feature_indexes], alpha=0.3)
plt.title('Feature importances')
plt.xlim(-.05, .25)
plt.show()


# Number of tree testing
trees = range(5, 100, 5)
forests = xrange(5)
acc = [] # accuracy

for tree in trees:
  total_score = 0
  for forest in forests:
    model = RandomForestClassifier(n_estimators=tree)
    model.fit(X_train, y_train)
    total_score += model.score(X_test, y_test)
  acc.append(total_score / len(forests))

plt.plot(trees, acc)
plt.show()




# Number of features testing

features = range(1, len(feature_names)+1)
fforests = xrange(5)
acc_fea = []

for feature in features:
    total_score_fea = 0
    for forest in fforests:
        model = RandomForestClassifier(max_features=feature)
        model.fit(X_train, y_train)
        total_score_fea += model.score(X_test, y_test)
    acc_fea.append(total_score_fea / len(fforests))

plt.plot(features, acc_fea)
plt.show()



# Comparison with other classifiers

def get_scores(classifier, X_train, X_test, y_train, y_test, **kwargs):
    model = classifier(**kwargs)
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    return model.score(X_test, y_test), precision_score(y_test, y_hat), recall_score(y_test, y_hat)

print 'Model: Accuracy, Precision, Recall'
print ' Random Forest:', get_scores(RandomForestClassifier, X_train, X_test, y_train, y_test, n_estimators=25, max_features=5)
print ' Logistic Regression:', get_scores(LogisticRegression, X_train, X_test, y_train, y_test)
print ' Decision tree:', get_scores(DecisionTreeClassifier, X_train, X_test, y_train, y_test)
print ' SVM:', get_scores(SVC, X_train, X_test, y_train, y_test)
print ' Naive Bayes:', get_scores(MultinomialNB, X_train, X_test, y_train, y_test)


# Protting ROC curve

def plot_roc_(X, y, clf_class, **kwargs):
    normalization = StandardScaler()
    X = normalization.fit_transform(X)
    kf = KFold(len(y), n_folds=5, shuffle=True)
    yprob = np.zeros((len(y), 2))
    tpr_mean = 0.0
    fpr_mean = np.linspace(0,1,100)

    for i, (train_index, test_index) in enumerate(kf):
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        model = clf_class(**kwargs)
        model.fit(X_train, y_train)

        yprob[test_index] = model.predict_proba(X_test)
        fpr, tpr, threshold = roc_curve(y[test_index], yprob[test_index, 1])
        tpr_mean += interp(fpr_mean, fpr, tpr)
        tpr_mean[0]=0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold {} (area = {:1.2f})'.format(i, roc_auc))

    tpr_mean /= len(kf)
    tpr_mean[-1] = 1.0
    auc_mean = auc(fpr_mean, tpr_mean)

    plt.plot(fpr_mean, tpr_mean,'k--',label='Mean ROC (area = {:1.2f})'.format(auc_mean), lw=2)
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('{}\n\nReceiver operating characteristic'.format(clf_class))
    plt.legend(loc="lower right")
    plt.show()

for model in (LogisticRegression, DecisionTreeClassifier, KNeighborsClassifier, RandomForestClassifier):
    plot_roc_(X.astype(float), y.astype(float), model)














































