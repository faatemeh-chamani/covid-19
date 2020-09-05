from sklearn.metrics import plot_roc_curve, r2_score, accuracy_score, plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree, svm, naive_bayes
from sklearn.tree import export_text
import matplotlib.pyplot as plt
from termcolor2 import colored


# Decision Tree
# SVM
# Naive Bayes
# Logistic Regression
# Random Forest
# K-NN

# decision tree
def decision_tree(X_train, y_train, X_test, y_test):
    print(colored('---- performance of  Decision Tree algorithm -----', color='blue'))
    max_depth_tree = [1, 5, 15]  # implement decision tree using different depth
    criteria = ['gini', 'entropy']  # implement decision tree using different criteria
    for Criterion in criteria:
        for max_depth in max_depth_tree:
            d_tree = tree.DecisionTreeClassifier(max_depth=max_depth, criterion=Criterion)
            d_tree.fit(X_train, y_train)
            y_pred1 = d_tree.predict(X_test)  # prediction for test data
            y_pred2 = d_tree.predict(X_train)  # prediction for training data
            print(colored(f'\tusing maximum {d_tree.max_depth} depth with {Criterion} criteria', color='green'))
            print(
                f'accuracy(correct predictions out of whole predictions) for test data: {accuracy_score(y_test, y_pred1)}')  # accuracy
            print(
                f'error rate(incorrect predictions out of whole predictions)for test data: {1 - accuracy_score(y_test, y_pred1)}')  # error rate
            print(
                f'accuracy(correct predictions out of whole predictions) for training data: {accuracy_score(y_train, y_pred2)}')  # accuracy
            print(
                f'error rate(incorrect predictions out of whole predictions) for training data: {1 - accuracy_score(y_train, y_pred2)}')  # error rate
            fig, axes = plt.subplots(figsize=(4, 4), dpi=180)
            tree.plot_tree(d_tree, filled=True, ax=axes, feature_names=['age', 'gender', 'fever'],
                           class_names=['Deceased', 'Recoveredً'], rounded=False,
                           proportion=True, rotate=False, precision=50, node_ids=True)  # plot tree in different depths
            plot_confusion_matrix(d_tree, X_test, y_test)
            plot_roc_curve(d_tree, X_test, y_test)  # plot roc curve and show auc value
            plt.title(f'max-depth={max_depth}, criterion={Criterion}', color='red')
            plt.show()
            print(export_text(d_tree, feature_names=['age', 'gender', 'fever']))


# SVM (support vector machine)
def SVM(X_train, y_train, X_test, y_test):
    print(colored('---- performance of  SVM algorithm -----', color='blue'))
    # consider different kernels and different regularization parameters
    Kernel = ['linear', 'poly', 'rbf', 'sigmoid']
    C = [0.01, 1000]
    for kernel in Kernel:
        for c in C:
            print(colored(f' --- svm with {kernel} kernel and c={c}', color='green'))
            clf = svm.SVC(C=c, kernel=kernel)  # use classification svm(SVC) dou to classification target
            clf.fit(X_train, y_train)
            y_pred1 = clf.predict(X_test)
            y_pred2 = clf.predict(X_train)
            print(
                f'accuracy(correct predictions out of whole predictions)for test data: {accuracy_score(y_test, y_pred1)}')
            print(
                f'error rate(incorrect predictions out of whole predictions)for test data: {1 - accuracy_score(y_test, y_pred1)}')
            print(
                f'accuracy(correct predictions out of whole predictions)for training data: {accuracy_score(y_train, y_pred2)}')
            print(
                f'error rate(incorrect predictions out of whole predictions)for training data: {1 - accuracy_score(y_train, y_pred2)}')
            plot_confusion_matrix(clf, X_test, y_test)  # plot confusion matrix
            plot_roc_curve(clf, X_test, y_test)  # plot ROC curve and show AUC value
            plt.title(f'svm ROC and AUC using {kernel} kernel and C={c}', color='red')
            plt.show()


# naive-bayes
def Naive_Bayes(X_train, y_train, X_test, y_test):
    print(colored('---- performance of  Naive_Bayes algorithm -----', color='blue'))
    clf = naive_bayes.GaussianNB()  # use gaussian naive bayes dou to classification target with continues features
    clf.fit(X_train, y_train)  # fit the model
    y_pred1 = clf.predict(X_test)  # predict test data
    y_pred2 = clf.predict(X_train)  # predict test data
    print(f'accuracy(correct predictions out of whole predictions) for test data: {accuracy_score(y_test, y_pred1)}')
    print(
        f'error rate(incorrect predictions out of whole predictions) for test data: {1 - accuracy_score(y_test, y_pred1)}')
    print(
        f'accuracy(correct predictions out of whole predictions)for training data: {accuracy_score(y_train, y_pred2)}')
    print(
        f'error rate(incorrect predictions out of whole predictions) for training data: {1 - accuracy_score(y_train, y_pred2)}')
    plot_confusion_matrix(clf, X_test, y_test)  # plot confusion matrix
    plot_roc_curve(clf, X_test, y_test)  # plot ROC curve and show AUC value
    plt.title('Naive-Bayes ROC curve and AUC', color='red')
    plt.show()


# k-nearest-neighborhood(K-NN)
def k_nn(X_train, y_train, X_test, y_test):
    print(colored('---- performance of k-nearest-neighborhood(K-NN) algorithm -----', color='blue'))
    # using different algorithm to implement k-nearest-neighborhood method
    K = [1, 5]
    algorithm = ['ball_tree', 'kd_tree', 'brute']
    for alg in algorithm:
        for k in K:
            print(colored(f'\t using {alg} algorithm with {k} neighbours', color='green'))
            K_NN = KNeighborsClassifier(n_neighbors=k, algorithm=alg)
            K_NN.fit(X_train, y_train)  # fit KNeighborsClassifier according to train data
            y_pred1 = K_NN.predict(X_test)  # predict test data
            y_pred2 = K_NN.predict(X_train)  # predict test data
            print(
                f'accuracy(correct predictions out of whole predictions)for test data: {accuracy_score(y_test, y_pred1)}')  # accuracy
            print(
                f'error rate(incorrect predictions out of whole predictions)for test data: {1 - accuracy_score(y_test, y_pred1)}')  # errror rate
            print(
                f'accuracy(correct predictions out of whole predictions)for training data: {accuracy_score(y_train, y_pred2)}')  # accuracy
            print(
                f'error rate(incorrect predictions out of whole predictions)for training data: {1 - accuracy_score(y_train, y_pred2)}')  # errro
            plot_confusion_matrix(K_NN, X_test, y_test)  # plot confusion matric
            plot_roc_curve(K_NN, X_test, y_test)  # plot ROC curve and show AUC value
            plt.title(f'K_NN ROC curve and AUC using {alg} algorithm with {k} neighbours', color='red')
            plt.show()


# random forest
def random_forest(X_train, y_train, X_test, y_test):
    print(colored('---- performance of Random-forest algorithm -----', color='blue'))
    criteria = ['gini', 'entropy']  # split method
    estimator = [3, 10]  # number of trees
    for Criterion in criteria:
        for estimate in estimator:
            print(colored(f'\t using {Criterion} ceriteria with {estimate} trees ', color='yellow'))
            rfc = RandomForestClassifier(n_estimators=estimate,
                                         criterion=Criterion,
                                         random_state=1)  # random forest classifier for classification targets
            rfc.fit(X_train, y_train)
            y_pred1 = rfc.predict(X_test)  # predict labels for test data
            y_pred2 = rfc.predict(X_train)
            print(f'the importance of each feature: {rfc.feature_importances_}')
            print(f'coefficient of determination: {r2_score(y_test, y_pred1)}')
            print(
                f'accuracy(correct predictions out of whole predictions)for test data: {accuracy_score(y_test, y_pred1)}')  # accuracy
            print(
                f'error rate(incorrect predictions out of whole predictions) for test data: {1 - accuracy_score(y_test, y_pred1)}')  # error rate
            print(
                f'accuracy(correct predictions out of whole predictions) for training data: {accuracy_score(y_train, y_pred2)}')  # accuracy
            print(
                f'error rate(incorrect predictions out of whole predictions) for training data: {1 - accuracy_score(y_train, y_pred2)}')  # error rate
            plot_confusion_matrix(rfc, X_test, y_test)  # plot confusion matrix
            plot_roc_curve(rfc, X_test, y_test)  # plot ROC curve and show AUC value
            plt.title(f'Random-forest ROC curve and AUC with {estimate} trees', color='red')
            plt.show()


# logistic Regression
def logistic_regression(X_train, y_train, X_test, y_test):
    print(colored('---- performance of  Logistic-Regression algorithm -----', color='blue'))
    log_reg = LogisticRegression(max_iter=100)  # logistic regression with maximum 200 iteration
    log_reg.fit(X_train, y_train)  # fit model according to training data
    y_pred1 = log_reg.predict(X_test)  # predict test data
    y_pred2 = log_reg.predict(X_train)  # predict test data
    print(
        f'accuracy(correct predictions out of whole predictions for test data): {accuracy_score(y_test, y_pred1)}')  # accuracy
    print(
        f'error rate(incorrect predictions out of whole predictions) for test data: {1 - accuracy_score(y_test, y_pred1)}')  # error rate

    print(
        f'accuracy(correct predictions out of whole predictions for training data): {accuracy_score(y_train, y_pred2)}')  # accuracy
    print(
        f'error rate(incorrect predictions out of whole predictions) for training data: {1 - accuracy_score(y_train, y_pred2)}')  # error rate
    plot_confusion_matrix(log_reg, X_test, y_test)
    plot_roc_curve(log_reg, X_test, y_test)  # plot ROC curve and show AUC value
    plt.title(f'Logistic-Regression ROC curve and AUC', color='red')
    plt.show()
