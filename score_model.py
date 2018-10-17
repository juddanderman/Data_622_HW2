import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from sklearn.externals import joblib


if __name__ == "__main__":
    # import training set and unlabeled test set (i.e. "score")
    train = pd.read_csv('train.csv')
    score = pd.read_csv('test.csv')
    out = score
    train = train.assign(Set = 'train')
    score = score.assign(Set = 'score')

    # combine train & score sets to dummy code all categorical values in both
    combined = train.append(score)
    combined = combined.assign(CabinLtr = combined['Cabin'].str[0])
    combined = pd.get_dummies(combined, 
        columns = ['Pclass', 'Sex', 'Embarked', 'CabinLtr'])
    combined = combined.drop(['Name', 'Ticket', 'Cabin'], axis=1)

    train = combined[combined['Set'] == 'train']
    train = train.drop('Set', axis=1)
    score = combined[combined['Set'] == 'score']
    score = score.drop('Set', axis=1)

    # data quality check
    score.describe()
    
    # missing value check
    score.isnull().sum()

    y = train['Survived'].values
    X = train.drop(['PassengerId', 'Survived'], axis=1).values

    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    X = imp.fit_transform(X)

    # use same training/holdout split as in train_model.py
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
        test_size = 0.2, random_state=1, stratify=y)

    # load fitted RF classifier
    rf = joblib.load('rf_classifier.pkl')

    y_pred = rf.predict(X_test)
    y_pred_prob = rf.predict_proba(X_test)[:,1]

    # export classification metric scores on holdout set
    file = open('titanic_holdout_set_classification_metrics.txt', 'w') 
    file.write(classification_report(y_test, y_pred)) 
    file.close()

    # compute AUC and export ROC curve
    y_pred_prob = rf.predict_proba(X_test)[:,1]
    fpr, tpr, thresholds  = roc_curve(y_test, y_pred_prob)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr,tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve, AUC = {0:0.2f}'.format(
        roc_auc_score(y_test, y_pred_prob)))
    plt.savefig('roc_curve.png')
    plt.gcf().clear()

    # compute average precision and export precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
    average_precision = average_precision_score(y_test, y_pred_prob)
    plt.step(recall, precision, color='b', alpha=0.2,
        where='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve, Average Precision = {0:0.2f}'.format(
        average_precision))
    plt.savefig('precision_recall_curve.png')

    # predict classes and positive class probabilities on unlabeled "score" data
    X_score = score.drop(['PassengerId'], axis=1).values
    X_score = imp.fit_transform(X_score)
    out = out.assign(Pred_Class = rf.predict(X_score),
        Pred_Prob = rf.predict_proba(X_score)[:,1])

    # export predictions to csv
    out.to_csv('titanic_unlabeled_test_set_predictions.csv', index=False)
