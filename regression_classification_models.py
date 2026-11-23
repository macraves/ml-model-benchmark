"""Regression and Classification Models"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score, confusion_matrix, accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


class DataSet:
    """
    Reads a CSV file and separates it into independent and dependent variables.
    For further implementation of different data set init only holds attributes.
    """

    def __init__(self, filepath: str):
        """
        Initializes the DataSet object. For Futher implementation
        current structure init method only holds attributes.

        Args:
            filepath (str): The path to the CSV file.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError("File does not exist")
        self.file_path = filepath
        self.random_state = 0
        self.r2_score = []
        self.dataset = None
        self.independents = None
        self.dependents = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def read_csv(self, independent_indices=(0, -1), dependent_indices=-1):
        """
        Reads a CSV file and separates it into independent and dependent variables.

        Args:
            independent_indices (tuple, optional): Indices between which the
            independent variables are selected. Defaults to (0,-1).
            dependent_indices (int, optional): Index of the dependent variable. Defaults to -1.
        """
        self.dataset = pd.read_csv(self.file_path)
        self.independents = self.dataset.iloc[
            :, independent_indices[0] : independent_indices[1]
        ].values
        self.dependents = self.dataset.iloc[:, dependent_indices].values

    def get_train_test_split(self, test_size=0.20):
        """
        Splits the dataset into training and testing sets.

        Args:
            test_size (float, optional): The proportion of the dataset
            to include in the test split. Defaults to 0.20.
        """
        (
            self.x_train,
            self.x_test,
            self.y_train,
            self.y_test,
        ) = train_test_split(
            self.independents,
            self.dependents,
            test_size=test_size,
            random_state=self.random_state,
        )

    def replace_mean_to_missing_data(self, indices: tuple, strategy="mean"):
        """
        Replaces missing data with the mean value.

        Args:
            indices (tuple): Indices between which the missing data needs to be replaced.
            strategy (str, optional): The strategy to use for
            replacing missing data. Defaults to "mean".
        """
        imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
        imputer.fit(self.independents[:, indices[0] : indices[1]])
        self.independents[:, indices[0] : indices[1]] = imputer.transform(
            self.independents[:, indices[0] : indices[1]]
        )

    def encoding_dependet_variable(self):
        """
        Encodes the dependent variable using label encoding.
        """
        label = LabelEncoder()
        self.dependents = label.fit_transform(self.dependents)

    def simple_linear_regressor(self):
        """
        Trains a Simple Linear Regression model on the training set.

        Returns:
            LinearRegression: The trained regression model.
        """
        regressor = LinearRegression()
        regressor.fit(self.x_train, self.y_train)
        return regressor

    def multiple_linear_regressor(self):
        """
        Trains a Multiple Linear Regression model on the training set.

        Returns:
            LinearRegression: The trained regression model.
        """
        regressor = LinearRegression()
        regressor.fit(self.x_train, self.y_train)
        return regressor

    def polynomial_regressor(self, degree=4):
        """
        Trains a Polynomial Linear Regression model on the training set.

        Args:
            degree (int, optional): The degree of the polynomial features. Defaults to 4.

        Returns:
            tuple: A tuple containing the trained
            PolynomialFeatures object and the trained regression model.
        """
        poly_reg = PolynomialFeatures(degree=degree)
        x_poly = poly_reg.fit_transform(self.x_train)
        regressor = LinearRegression()
        regressor.fit(x_poly, self.y_train)
        return poly_reg, regressor

    def support_vector_regressor(self):
        """
        Trains a Support Vector Regression model on the training set.

        Returns:
            SVR: The trained regression model.
        """
        regressor = SVR(kernel="rbf")
        regressor.fit(self.x_train, self.y_train)
        return regressor

    def decision_tree_regressor(self):
        """
        Trains a Decision Tree Regression model on the training set.

        Returns:
            DecisionTreeRegressor: The trained regression model.
        """
        regressor = DecisionTreeRegressor(random_state=self.random_state)
        regressor.fit(self.x_train, self.y_train)
        return regressor

    def random_forest_regressor(self, estimators=10):
        """
        Trains a Random Forest Regression model on the training set.

        Args:
            estimators (int, optional): The number of trees in the random forest. Defaults to 10.

        Returns:
            RandomForestRegressor: The trained regression model.
        """
        regressor = RandomForestRegressor(
            n_estimators=estimators, random_state=self.random_state
        )
        regressor.fit(self.x_train, self.y_train)
        return regressor

    def classifier_logistic_regression(self, independent_vars, dependent_vars):
        """
        Trains a Logistic Regression Classifier on the training set.

        Args:
            independent_vars (array-like): The independent variables.
            dependent_vars (array-like): The dependent variable.

        Returns:
            LogisticRegression: The trained classifier.
        """
        classifier = LogisticRegression(random_state=self.random_state)
        classifier.fit(independent_vars, dependent_vars)
        return classifier

    def classifier_k_nearest_neighboors(
        self, independent_vars, dependent_vars, neighbors=5
    ):
        """
        Trains a K-Nearest Neighbors Classifier on the training set.

        Args:
            independent_vars (array-like): The independent variables.
            dependent_vars (array-like): The dependent variable.
            neighbors (int, optional): The number of neighbors to consider. Defaults to 5.

        Returns:
            KNeighborsClassifier: The trained classifier.
        """
        classifier = KNeighborsClassifier(
            n_neighbors=neighbors, metric="minkowski", p=2
        )
        classifier.fit(independent_vars, dependent_vars)
        return classifier

    def classifier_support_vector_machine(
        self, independent_vars, dependent_vars, kernel="linear"
    ):
        """
        Trains a Support Vector Machine Classifier on the training set.

        Args:
            independent_vars (array-like): The independent variables.
            dependent_vars (array-like): The dependent variable.
            kernel (str, optional): The kernel function to use. Defaults to "linear".

        Returns:
            SVC: The trained classifier.
        """
        classifier = SVC(kernel=kernel, random_state=self.random_state)
        classifier.fit(independent_vars, dependent_vars)
        return classifier

    def classifier_kernel_svm(self, independent_vars, dependent_vars, kernel="rbf"):
        """
        Trains a Kernel SVM Classifier on the training set.

        Args:
            independent_vars (array-like): The independent variables.
            dependent_vars (array-like): The dependent variable.
            kernel (str, optional): The kernel function to use. Defaults to "rbf".

        Returns:
            SVC: The trained classifier.
        """
        classifier = SVC(kernel=kernel, random_state=self.random_state)
        classifier.fit(independent_vars, dependent_vars)
        return classifier

    def classifier_naive_bayes(self, independent_vars, dependent_vars):
        """
        Trains a Naive Bayes Classifier on the training set.

        Args:
            independent_vars (array-like): The independent variables.
            dependent_vars (array-like): The dependent variable.

        Returns:
            GaussianNB: The trained classifier.
        """
        classifier = GaussianNB()
        classifier.fit(independent_vars, dependent_vars)
        return classifier

    def classifier_decision_tree(self, independent_vars, dependent_vars):
        """
        Trains a Decision Tree Classifier on the training set.

        Args:
            independent_vars (array-like): The independent variables.
            dependent_vars (array-like): The dependent variable.

        Returns:
            DecisionTreeClassifier: The trained classifier.
        """
        classifier = DecisionTreeClassifier(
            criterion="entropy", random_state=self.random_state
        )
        classifier.fit(independent_vars, dependent_vars)
        return classifier

    def classifier_random_forest(self, independent_vars, dependent_vars, estimators=10):
        """
        Trains a Random Forest Classifier on the training set.

        Args:
            independent_vars (array-like): The independent variables.
            dependent_vars (array-like): The dependent variable.
            estimators (int, optional): The number of trees in the random forest. Defaults to 10.

        Returns:
            RandomForestClassifier: The trained classifier.
        """
        classifier = RandomForestClassifier(
            n_estimators=estimators, criterion="entropy", random_state=self.random_state
        )
        classifier.fit(independent_vars, dependent_vars)
        return classifier

    def classification_feature_scaling(self):
        """
        Scales the independent variables using StandardScaler.

        Returns:
            tuple: A tuple containing the scaled x_train and x_test.
        """
        sc = StandardScaler()
        x_train = sc.fit_transform(self.x_train)
        x_test = sc.transform(self.x_test)
        return x_train, x_test

    def classification_scores(self, y_test, y_pred):
        """
        Prints the confusion matrix and accuracy score.

        Args:
            y_test (array-like): The true labels.
            y_pred (array-like): The predicted labels.
        """
        print(confusion_matrix(y_test, y_pred))
        print(accuracy_score(y_test, y_pred=y_pred))

    def classification_r2_score(self, y_test, y_pred):
        """
        Prints the R2 score.

        Args:
            y_test (array-like): The true labels.
            y_pred (array-like): The predicted labels.
        """
        self.r2_score.append(r2_score(y_test, y_pred))
        print(r2_score(y_test, y_pred))

    def print_results(self, y_pred):
        """
        Prints the classification scores and R2 score.

        Args:
            y_pred (array-like): The predicted labels.
        """
        self.classification_scores(self.y_test, y_pred)
        print("-" * 10)

    def run_classification_model_test(self, scalling=True):
        """
        Runs all classification model tests.

        Args:
            scalling (bool, optional): Whether to perform feature scaling. Defaults to True.
        """
        self.get_train_test_split()
        x_train = self.x_train
        x_test = self.x_test
        if scalling:
            x_train, x_test = self.classification_feature_scaling()
        print("Naive Bayes Results")
        classifier = self.classifier_naive_bayes(x_train, self.y_train)
        y_pred = classifier.predict(x_test)
        self.print_results(y_pred)
        print("SVC Results")
        classifier = self.classifier_support_vector_machine(x_train, self.y_train)
        y_pred = classifier.predict(x_test)
        self.print_results(y_pred)
        print("Decision Tree Results")
        classifier = self.classifier_decision_tree(x_train, self.y_train)
        y_pred = classifier.predict(x_test)
        self.print_results(y_pred)
        print("Random Forest Results")
        classifier = self.classifier_random_forest(x_train, self.y_train)
        y_pred = classifier.predict(x_test)
        self.print_results(y_pred)
        print("KNN_MODEL Results")
        classifier = self.classifier_k_nearest_neighboors(x_train, self.y_train)
        y_pred = classifier.predict(x_test)
        self.print_results(y_pred)
        print("Kernel SVM Results")
        classifier = self.classifier_kernel_svm(x_train, self.y_train)
        y_pred = classifier.predict(x_test)
        self.print_results(y_pred)
        print("Logistic Regression Results")
        classifier = self.classifier_logistic_regression(x_train, self.y_train)
        y_pred = classifier.predict(x_test)
        self.print_results(y_pred)
