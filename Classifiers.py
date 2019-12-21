import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


def taylor2016appscanner_RF(x_val, y_val, x_train, y_train):

    # estimators_number = 150
    estimators_number = 50

    # Define RF classifier with parameters specified in taylor2016appcanner
    clf = RandomForestClassifier(criterion='gini', max_features='sqrt', n_estimators=estimators_number)

    # Training phase
    clf.fit(x_train, y_train)

    predictions = clf.predict(x_val)

    # Test with predict_classes: test_model(model, transformed_samples_test, categorical_labels_test)
    accuracy = sklearn.metrics.accuracy_score(y_val, predictions)

    return predictions, accuracy


def decision_tree(x_val, y_val, x_train, y_train):

    # Define RF classifier with parameters specified in taylor2016appcanner
    clf = sklearn.tree.DecisionTreeClassifier()

    # Training phase
    clf.fit(x_train, y_train)

    predictions = clf.predict(x_val)

    # Test with predict_classes: test_model(model, transformed_samples_test, categorical_labels_test)
    accuracy = sklearn.metrics.accuracy_score(y_val, predictions)

    return predictions, accuracy


def gaussian_naive_bayes(x_val, y_val, x_train, y_train):

    # Define RF classifier with parameters specified in taylor2016appcanner
    clf = GaussianNB()

    # Training phase
    clf.fit(x_train, y_train)

    predictions = clf.predict(x_val)

    # Test with predict_classes: test_model(model, transformed_samples_test, categorical_labels_test)
    accuracy = sklearn.metrics.accuracy_score(y_val, predictions)

    return predictions, accuracy

