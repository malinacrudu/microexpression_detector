from sklearn import svm, metrics
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from utils import log_message, createConfusionMatrix


# creates a SVM model with the help of sklearn library
# as arguments it has input data (X) and output data (y)
# and it runs the train-test process 20 times
def svm_model(X, y):
    log_message("--------------------------SVM--------------------------")
    trials = 20
    ok = 1
    while trials:
        kf = KFold(n_splits=5, shuffle=True)
        kf.get_n_splits(X)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True)
            clf = svm.SVC(kernel='linear', verbose=1)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_pred_train = clf.predict(X_train)
            from sklearn import metrics
            log_message("TRIAL: " + str(trials) + " Accuracy testing: " + str(metrics.accuracy_score(y_test, y_pred)))
            log_message(
                "TRIAL: " + str(trials) + " Accuracy training: " + str(metrics.accuracy_score(y_train, y_pred_train)))
            log_message('Precision: ' + str(precision_score(y_test, y_pred, average='weighted')))
            log_message('Recall: ' + str(recall_score(y_test, y_pred, average='weighted')))
            log_message('F1 Score: ' + str(f1_score(y_test, y_pred, average='weighted')))
            if trials == 20 and ok == 1:
                createConfusionMatrix(y_pred, y_test)
                ok += 1

        trials -= 1


def bayes_model(X, y):
    log_message("--------------------------NAIVE BAYES--------------------------")
    trials = 20
    ok = 1
    while trials:
        kf = KFold(n_splits=5, shuffle=True)
        kf.get_n_splits(X)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            gnb = GaussianNB()
            y_pred = gnb.fit(X_train, y_train).predict(X_test)
            y_pred_train = gnb.predict(X_train)
            from sklearn import metrics
            log_message("TRIAL: " + str(trials) + " Accuracy testing: " + str(metrics.accuracy_score(y_test, y_pred)))
            log_message(
                "TRIAL: " + str(trials) + " Accuracy training: " + str(metrics.accuracy_score(y_train, y_pred_train)))
            log_message('Precision: ' + str(precision_score(y_test, y_pred, average='weighted')))
            log_message('Recall: ' + str(recall_score(y_test, y_pred, average='weighted')))
            log_message('F1 Score: ' + str(f1_score(y_test, y_pred, average='weighted')))
            if trials == 20 and ok == 1:
                createConfusionMatrix(y_pred, y_test)
                ok += 1
        trials -= 1


def neural_network_model(X, y):
    log_message("--------------------------MLP--------------------------")
    trials = 20
    ok = 1
    while trials:
        kf = KFold(n_splits=5, shuffle=True)
        kf.get_n_splits(X)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            mlp = MLPClassifier(hidden_layer_sizes=(512, 128, 16), activation='relu', solver='adam', max_iter=100, shuffle=True, verbose=True, early_stopping=True)
            mlp.fit(X_train, y_train)
            y_pred = mlp.predict(X_test)
            y_pred_train = mlp.predict(X_train)
            log_message("TRIAL: " + str(trials) + " Accuracy testing: " + str(metrics.accuracy_score(y_test, y_pred)))
            log_message(
                "TRIAL: " + str(trials) + " Accuracy training: " + str(metrics.accuracy_score(y_train, y_pred_train)))
            log_message('Precision: ' + str(precision_score(y_test, y_pred, average='weighted')))
            log_message('Recall: ' + str(recall_score(y_test, y_pred, average='weighted')))
            log_message('F1 Score: ' + str(f1_score(y_test, y_pred, average='weighted')))
            if trials == 20 and ok == 1:
                createConfusionMatrix(y_pred, y_test)
                ok += 1
        trials -= 1
