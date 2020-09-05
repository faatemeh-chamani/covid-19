from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import learning


# load data
dataSet = pd.read_csv('covid-data.csv')

X = dataSet.iloc[:, [1, 2, 3]].values  # extract features
y = dataSet.iloc[:, 4].values  # extract labels
# encoding features
encoder_X = LabelEncoder()
X[:, 1] = encoder_X.fit_transform(X[:, 1])  # convert second feature(gender) from text type to binary[0,1]
# encoding labels
encoder_y = LabelEncoder()
y = encoder_y.fit_transform(y)  # convert text labels to 0-1 labels --> 0:deceased, 1:recovered

# split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8, shuffle=True, random_state=1)

# Training
learning.decision_tree(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
learning.SVM(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
learning.Naive_Bayes(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
learning.random_forest(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
learning.k_nn(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
learning.logistic_regression(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
