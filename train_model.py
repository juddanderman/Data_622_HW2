import pandas as pd
import numpy as np
from scipy.stats import randint
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import RandomizedSearchCV
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib


if __name__ == "__main__":
    # import training set and unlabeled test set (i.e. "score")
    train = pd.read_csv('train.csv')
    score = pd.read_csv('test.csv')
    out = score

    # combine train & score sets to dummy code all categorical values in both
    train = train.assign(Set = 'train')
    score = score.assign(Set = 'score')
    combined = train.append(score)
    combined = combined.assign(CabinLtr = combined['Cabin'].str[0])
    combined = pd.get_dummies(combined, 
        columns = ['Pclass', 'Sex', 'Embarked', 'CabinLtr'])
    combined = combined.drop(['Name', 'Ticket', 'Cabin'], axis=1)
    train = combined[combined['Set'] == 'train']
    train = train.drop('Set', axis=1)

    # data quality check
    train.describe()
    
    # missing value check
    train.isnull().sum()

    y = train['Survived'].values
    X = train.drop(['PassengerId', 'Survived'], axis=1).values

    # use mean value imputation for missing values
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    X = imp.fit_transform(X)

    # split training set into model training and holdout sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
        test_size = 0.2, random_state=1, stratify=y)

    # instantiate random forest classifier
    clf = RandomForestClassifier(random_state=1)

    # create random forest hyperparameter tuning search space
    param_dist = {"n_estimators": [10, 100, 1000],
        "max_depth": [3, None],
        "max_features": randint(4, 20),
        "min_samples_split": randint(2, 8),
        "bootstrap": [True, False],
        "criterion": ["gini", "entropy"]}

    # use randomized search over parameter space defined above
    # with 20 iterations & 5-fold CV to find best fitting RF model
    random_search = RandomizedSearchCV(clf, 
        param_distributions=param_dist, n_iter=20, cv=5)
    random_search.fit(X_train, y_train)

    # export best fitting RF model as pkl
    joblib.dump(random_search.best_estimator_, 'rf_classifier.pkl', compress=1)
