from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from xgboost import XGBClassifier
import numpy as np
import pandas as pd


def predict(x_train, x_test, y_train, y_test, clf):
    # Train
    clf.fit(x_train, y_train)
    # Predicting test set
    y_pred_test = clf.predict(x_test)
    cm_test = confusion_matrix(y_pred_test, y_test)

    # Predicting train set
    y_pred_train = clf.predict(x_train)
    cm_train = confusion_matrix(y_pred_train, y_train)

    accuracy_for_train = np.round((cm_train[0][0] + cm_train[1][1]) / len(y_train), 2)
    accuracy_for_test = np.round((cm_test[0][0] + cm_test[1][1]) / len(y_test), 2)

    print('Accuracy for training set = {}'.format(accuracy_for_train))
    print('Accuracy for test set = {}'.format(accuracy_for_test))
    print('')


def run():
    df = pd.read_csv('cleveland.csv', header=None)
    df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol',
                  'fbs', 'restecg', 'thalach', 'exang',
                  'oldpeak', 'slope', 'ca', 'thal', 'target']
    df['target'] = df.target.map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})
    df['thal'] = df.thal.fillna(df.thal.mean())
    df['ca'] = df.ca.fillna(df.ca.mean())

    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    # Split Data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Question 1 KNN -> A
    print('KNN Classifier')
    knn_clf = KNeighborsClassifier(n_neighbors=5, weights='uniform',
                                   algorithm='auto', leaf_size=30, p=2, metric='minkowski')
    predict(x_train, x_test, y_train, y_test, knn_clf)

    # Question 2 SVM -> B
    print('SVM Classifier')
    svm_clf = SVC(kernel='rbf', random_state=42, C=1.0, gamma='scale')
    predict(x_train, x_test, y_train, y_test, svm_clf)

    # Question 3 Naive Bayes _> C
    print('Naive Bayes Classifier')
    nb_clf = GaussianNB()
    predict(x_train, x_test, y_train, y_test, nb_clf)

    # Question 4 Decision Tree -> D
    print("Decision Tree Classifier")
    dt_clf = DecisionTreeClassifier(criterion='gini', max_depth=10,
                                    min_samples_split=2, random_state=42, ccp_alpha=0.0)
    predict(x_train, x_test, y_train, y_test, dt_clf)

    # Question 5 Random forest -> A
    print("Random forest Classifier")
    rdf_clf = RandomForestClassifier(criterion='gini', max_depth=10, min_samples_split=2,
                                     n_estimators=10, random_state=42, min_samples_leaf=1, max_features='sqrt')
    predict(x_train, x_test, y_train, y_test, rdf_clf)

    # Question 6 AdaBoost -> B
    print("AdaBoost Classifier")
    ada_boost_clf = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=42)
    predict(x_train, x_test, y_train, y_test, ada_boost_clf)

    # Question 7 GradientBoost -> C
    print("GradientBoost Classifier")
    gra_boost_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42,
                                               subsample=1.0, min_samples_split=2, max_depth=3
                                               )
    predict(x_train, x_test, y_train, y_test, gra_boost_clf)

    # Question 8 XGBoost -> C (1, 0.87) ?
    print("XGBoost Classifier")
    xg_boost_clf = XGBClassifier(objective="binary:logistic", random_state=42, n_estimators=100)
    predict(x_train, x_test, y_train, y_test, xg_boost_clf)

    # Question 9 Stacking -> A (0,9 - 0,9) ?
    print("Stacking Classifier")
    stack_clf = [('dtc', dt_clf), ('rfc', rdf_clf), ('knn', knn_clf),
                 ('gc', gra_boost_clf), ('ad', ada_boost_clf), ('svc', svm_clf)]
    classifier = StackingClassifier(estimators=stack_clf, final_estimator=xg_boost_clf)
    predict(x_train, x_test, y_train, y_test, classifier)


run()
