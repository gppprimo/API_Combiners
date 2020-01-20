import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from Constants import label_classes


def taylor2016appscanner_RF(x_val, y_val, x_train, y_train, samples_test, categorical_labels_test):
    # estimators_number = 150
    estimators_number = 5

    # Define RF classifier with parameters specified in taylor2016appcanner
    clf = RandomForestClassifier(criterion='gini', max_features='sqrt', n_estimators=estimators_number)

    # Training & validation phase
    clf.fit(x_train, y_train)

    val_predictions = clf.predict(x_val)

    accuracy = sklearn.metrics.accuracy_score(y_val, val_predictions)

    recalls = sklearn.metrics.recall_score(y_val, val_predictions, average=None)

    # Test phase
    predictions = clf.predict(samples_test)

    return predictions, accuracy, recalls


def decision_tree(x_val, y_val, x_train, y_train, samples_test, categorical_labels_test):
    # Define DT classifier
    clf = sklearn.tree.DecisionTreeClassifier()

    # Training & validation phase
    clf.fit(x_train, y_train)

    val_predictions = clf.predict(x_val)

    accuracy = sklearn.metrics.accuracy_score(y_val, val_predictions)

    recalls = sklearn.metrics.recall_score(y_val, val_predictions, average=None)

    # Test phase
    predictions = clf.predict(samples_test)

    return predictions, accuracy, recalls


def gaussian_naive_bayes(x_val, y_val, x_train, y_train, samples_test, categorical_labels_test):
    # Define NB classifier
    clf = GaussianNB()

    # Training & validation phase
    clf.fit(x_train, y_train)

    val_predictions = clf.predict(x_val)

    accuracy = sklearn.metrics.accuracy_score(y_val, val_predictions)

    recalls = sklearn.metrics.recall_score(y_val, val_predictions, average=None)

    # Test phase
    predictions = clf.predict(samples_test)

    return predictions, accuracy, recalls



