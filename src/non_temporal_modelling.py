import math
import random
import copy
import os

import numpy as np
import pandas as pd
from sklearn import metrics, tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split


# This class creates dfs that can be used by the learning algorithms. Up till now we have
# assumed binary columns for each class, we will for instance introduce approaches to create
# a single categorical attribute.
class ClassificationPrepareData:

    default_label = 'undefined'
    class_col = 'class'

    # This function creates a single class column based on a set of binary class columns, essentially merging them. 
    # It removes the old label columns.
    def assign_label(
        self, 
        df: pd.DataFrame, 
        class_labels: list[str]
    ) -> pd.DataFrame:
        
        # Find which columns are relevant based on the possibly partial class_label specification.
        labels = []
        for i in range(0, len(class_labels)):
            labels.extend([name for name in list(df.columns) if class_labels[i] == name[0:len(class_labels[i])]])

        # Determine how many class values are label as 'true' in our class columns.
        sum_values = df[labels].sum(axis=1)
        # Create a new 'class' column and set the value to the default class.
        df['class'] = self.default_label
        for i in range(0, len(df.index)):
            # If we have exactly one true class column, we can assign that value, otherwise we keep the default class.
            if sum_values[i] == 1:
                df.iloc[i, df.columns.get_loc(self.class_col)] = df[labels].iloc[i].idxmax(axis=0)
        # And remove our old binary columns.
        df = df.drop(labels, axis=1)

        return df

    # Split a df of a single person for a classificaiton problem with the the specified class columns class_labels.
    # We can have multiple targets if we want. It assumes a list in 'class_labels'
    # If 'like' is specified in matching, we will merge the columns that contain the class_labels into a single
    # columns. We can select a filter for rows where we are unable to identifty a unique
    # class and we can select whether we have a temporal df or not. In the former, we will select the first
    # training_frac of the data for training and the last 1-training_frac for testing. Otherwise, we select points randomly.
    # We return a training set, the labels of the training set, and the same for a test set. We can set the random seed
    # to make the split reproducible.
    def split_classification(
        self, 
        df: pd.DataFrame, 
        class_labels: list[str], 
        matching: str = "like", 
        training_frac: float = 0.7, 
        filter: bool = True, 
        temporal: bool = False, 
        random_state: int = 0
    )-> pd.DataFrame:

        # Create a single class column if we have the 'like' option.
        if matching == 'like':
            df = self.assign_label(df, class_labels)
            class_labels = self.class_col
        elif len(class_labels) == 1:
            class_labels = class_labels[0]

        # Filter NaN is desired and those for which we cannot determine the class should be removed.
        if filter:
            df = df.dropna()
            df = df[df['class'] != self.default_label]

        # The features are the ones not in the class label.
        features = [df.columns.get_loc(x) for x in df.columns if x not in class_labels]
        class_label_indices = [df.columns.get_loc(x) for x in df.columns if x in class_labels]

        # For temporal data, we select the desired fraction of training data from the first part and use the rest as test set.
        if temporal:
            end_training_set = int(training_frac * len(df.index))
            training_set_X = df.iloc[0:end_training_set, features]
            training_set_y = df.iloc[0:end_training_set, class_label_indices]
            test_set_X = df.iloc[end_training_set:len(df.index), features]
            test_set_y = df.iloc[end_training_set:len(df.index), class_label_indices]
        # For non temporal data we use a standard function to randomly split the df.
        else:
            training_set_X, test_set_X, training_set_y, test_set_y = train_test_split(df.iloc[:,features],
                                                                                      df.iloc[:,class_label_indices], test_size=(1-training_frac), stratify=df.iloc[:,class_label_indices], random_state=random_state)
        return training_set_X, test_set_X, training_set_y, test_set_y


class ClassificationAlgorithms:

    # Apply a neural network for classification upon the training data (with the specified composition of
    # hidden layers and number of iterations), and use the created network to predict the outcome for both the
    # test and training set. It returns the categorical predictions for the training and test set as well as the
    # probabilities associated with each class, each class being represented as a column in the data frame.
    # To improve the speed, one can use a CV of 3 only to make it faster
    # Furthermore, you decrease the number of iteration and increase the learning rate, i.e. 0.001 and use 'adam' as a solver
    # Include n_jobs in the GridSearchCV function and set it to -1 to use all processors which could also increase the speed
    def feedforward_neural_network(self, train_X, train_y, test_X, hidden_layer_sizes=(100,), max_iter=500, activation='logistic', alpha=0.0001, learning_rate='adaptive', gridsearch=True, print_model_details=False):

        if gridsearch:
            # With the current parameters for max_iter and Python 3 packages convergence is not always reached, therefore increased +1000.
            tuned_parameters = [{'hidden_layer_sizes': [(5,), (10,), (25,), (100,), (100,5,), (100,10,),], 'activation': [activation],
                                 'learning_rate': [learning_rate], 'max_iter': [2000, 3000], 'alpha': [alpha]}]
            nn = GridSearchCV(MLPClassifier(), tuned_parameters, cv=5, scoring='accuracy')
        else:
            # Create the model
            nn = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, max_iter=max_iter, learning_rate=learning_rate, alpha=alpha, random_state=42)

        # Fit the model
        nn.fit(train_X, train_y.values.ravel())

        if gridsearch and print_model_details:
            print(nn.best_params_)

        if gridsearch:
            nn = nn.best_estimator_

        # Apply the model
        pred_prob_training_y = nn.predict_proba(train_X)
        pred_prob_test_y = nn.predict_proba(test_X)
        pred_training_y = nn.predict(train_X)
        pred_test_y = nn.predict(test_X)
        frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=nn.classes_)
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=nn.classes_)

        return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y

    # Apply a support vector machine for classification upon the training data (with the specified value for
    # C, epsilon and the kernel function), and use the created model to predict the outcome for both the
    # test and training set. It returns the categorical predictions for the training and test set as well as the
    # probabilities associated with each class, each class being represented as a column in the data frame.
    # To improve the speed, one can use a CV of 3 only to make it faster
    # Include n_jobs in the GridSearchCV function and set it to -1 to use all processors which could also increase the speed
    def support_vector_machine_with_kernel(self, train_X, train_y, test_X, C=1,  kernel='rbf', gamma=1e-3, gridsearch=True, print_model_details=False):
        # Create the model
        if gridsearch:
            tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100]}]
            svm = GridSearchCV(SVC(probability=True), tuned_parameters, cv=5, scoring='accuracy')
        else:
            svm = SVC(C=C, kernel=kernel, gamma=gamma, probability=True, cache_size=7000)

        # Fit the model
        svm.fit(train_X, train_y.values.ravel())

        if gridsearch and print_model_details:
            print(svm.best_params_)

        if gridsearch:
            svm = svm.best_estimator_

        # Apply the model
        pred_prob_training_y = svm.predict_proba(train_X)
        pred_prob_test_y = svm.predict_proba(test_X)
        pred_training_y = svm.predict(train_X)
        pred_test_y = svm.predict(test_X)
        frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=svm.classes_)
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=svm.classes_)

        return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y

    # Apply a support vector machine for classification upon the training data (with the specified value for
    # C, epsilon and the kernel function), and use the created model to predict the outcome for both the
    # test and training set. It returns the categorical predictions for the training and test set as well as the
    # probabilities associated with each class, each class being represented as a column in the data frame.
    # To improve the speed, one can use a CV of 3 only to make it faster and use fewer iterations
    def support_vector_machine_without_kernel(self, train_X, train_y, test_X, C=1, tol=1e-3, max_iter=1000, gridsearch=True, print_model_details=False):
        # Create the model
        if gridsearch:
            tuned_parameters = [{'max_iter': [1000, 2000], 'tol': [1e-3, 1e-4],
                         'C': [1, 10, 100]}]
            svm = GridSearchCV(LinearSVC(), tuned_parameters, cv=5, scoring='accuracy')
        else:
            svm = LinearSVC(C=C, tol=tol, max_iter=max_iter)

        # Fit the model
        svm.fit(train_X, train_y.values.ravel())

        if gridsearch and print_model_details:
            print(svm.best_params_)

        if gridsearch:
            svm = svm.best_estimator_

        # Apply the model

        distance_training_platt = 1/(1+np.exp(svm.decision_function(train_X)))
        pred_prob_training_y = distance_training_platt / distance_training_platt.sum(axis=1)[:,None]
        distance_test_platt = 1/(1+np.exp(svm.decision_function(test_X)))
        pred_prob_test_y = distance_test_platt / distance_test_platt.sum(axis=1)[:,None]
        pred_training_y = svm.predict(train_X)
        pred_test_y = svm.predict(test_X)
        frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=svm.classes_)
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=svm.classes_)

        return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y

    # Apply a decision tree approach for classification upon the training data (with the specified value for
    # the minimum samples in the leaf, and the export path and files if print_model_details=True)
    # and use the created model to predict the outcome for both the
    # test and training set. It returns the categorical predictions for the training and test set as well as the
    # probabilities associated with each class, each class being represented as a column in the data frame.
    # Again, use CV of 3 which will increase the speed of your model
    # Also, usage of n_jobs in GridSearchCV could help to increase the speed
    def decision_tree(self, train_X, train_y, test_X, min_samples_leaf=50, criterion='gini', print_model_details=False, export_tree_path='./figures/crowdsignals_ch7_classification/', export_tree_name='tree.dot', gridsearch=True):
        # Create the model
        if gridsearch:
            tuned_parameters = [{'min_samples_leaf': [2, 10, 50, 100, 200],
                                 'criterion':['gini', 'entropy']}]
            dtree = GridSearchCV(DecisionTreeClassifier(), tuned_parameters, cv=5, scoring='accuracy')
        else:
            dtree = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf, criterion=criterion)

        # Fit the model

        dtree.fit(train_X, train_y.values.ravel())

        if gridsearch and print_model_details:
            print(dtree.best_params_)

        if gridsearch:
            dtree = dtree.best_estimator_

        # Apply the model
        pred_prob_training_y = dtree.predict_proba(train_X)
        pred_prob_test_y = dtree.predict_proba(test_X)
        pred_training_y = dtree.predict(train_X)
        pred_test_y = dtree.predict(test_X)
        frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=dtree.classes_)
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=dtree.classes_)

        if print_model_details:
            ordered_indices = [i[0] for i in sorted(enumerate(dtree.feature_importances_), key=lambda x:x[1], reverse=True)]
            print('Feature importance decision tree:')
            for i in range(0, len(dtree.feature_importances_)):
                print(train_X.columns[ordered_indices[i]], end='')
                print(' & ', end='')
                print(dtree.feature_importances_[ordered_indices[i]])
            if not (os.path.exists(export_tree_path)):
                os.makedirs(str(export_tree_path))
            tree.export_graphviz(dtree, out_file=str(export_tree_path) + '/' + export_tree_name, feature_names=train_X.columns, class_names=dtree.classes_)

        return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y
    
    # Apply a random forest approach for classification upon the training data (with the specified value for
    # the minimum samples in the leaf, the number of trees, and if we should print some of the details of the
    # model print_model_details=True) and use the created model to predict the outcome for both the
    # test and training set. It returns the categorical predictions for the training and test set as well as the
    # probabilities associated with each class, each class being represented as a column in the data frame.
    # Use CV of 3 to make things faster
    # Use n_jobs = -1 which will make use of all of your processors. This could speed up also the calculation
    def random_forest(self, train_X, train_y, test_X, n_estimators=10, min_samples_leaf=5, criterion='gini', print_model_details=False, gridsearch=True):

        if gridsearch:
            tuned_parameters = [{'min_samples_leaf': [2, 10, 50, 100, 200],
                                 'n_estimators':[10, 50, 100],
                                 'criterion':['gini', 'entropy']}]
            rf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5, scoring='accuracy')
        else:
            rf = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, criterion=criterion)

        # Fit the model

        rf.fit(train_X, train_y.values.ravel())

        if gridsearch and print_model_details:
            print(rf.best_params_)

        if gridsearch:
            rf = rf.best_estimator_

        pred_prob_training_y = rf.predict_proba(train_X)
        pred_prob_test_y = rf.predict_proba(test_X)
        pred_training_y = rf.predict(train_X)
        pred_test_y = rf.predict(test_X)
        frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=rf.classes_)
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=rf.classes_)

        if print_model_details:
            ordered_indices = [i[0] for i in sorted(enumerate(rf.feature_importances_), key=lambda x:x[1], reverse=True)]
            print('Feature importance random forest:')
            for i in range(0, len(rf.feature_importances_)):
                print(train_X.columns[ordered_indices[i]], end='')
                print(' & ', end='')
                print(rf.feature_importances_[ordered_indices[i]])

        return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y


# Class for evaluation metrics of classification problems.
class ClassificationEvaluation:

    # Returns the accuracy given the true and predicted values.
    def accuracy(self, y_true, y_pred):
        return metrics.accuracy_score(y_true, y_pred)

    # Returns the precision given the true and predicted values.
    # Note that it returns the precision per class.
    def precision(self, y_true, y_pred):
        return metrics.precision_score(y_true, y_pred, average=None)

    # Returns the recall given the true and predicted values.
    # Note that it returns the recall per class.
    def recall(self, y_true, y_pred):
        return metrics.recall_score(y_true, y_pred, average=None)

    # Returns the f1 given the true and predicted values.
    # Note that it returns the recall per class.
    def f1(self, y_true, y_pred):
        return metrics.f1_score(y_true, y_pred, average=None)

    # Returns the area under the curve given the true and predicted values.
    # Note: we expect a binary classification problem here(!)
    def auc(self, y_true, y_pred_prob):
        return metrics.roc_auc_score(y_true, y_pred_prob)

    # Returns the confusion matrix given the true and predicted values.
    def confusion_matrix(self, y_true, y_pred, labels):
        return metrics.confusion_matrix(y_true, y_pred, labels=labels)
    

# Specifies feature selection approaches for classification to identify the most important features.
class ClassificationFeatureSelection:

    # Forward selection for classification which selects a pre-defined number of features (max_features)
    # that show the best accuracy. We assume a decision tree learning for this purpose, but
    # this can easily be changed. It return the best features.
    def forward_selection(self, max_features, X_train, X_test, y_train, y_test, gridsearch):
        # Start with no features.
        ordered_features = []
        ordered_scores = []
        selected_features = []
        ca = ClassificationAlgorithms()
        ce = ClassificationEvaluation()
        prev_best_perf = 0

        # Select the appropriate number of features.
        for i in range(0, max_features):
            # Determine the features left to select.
            features_left = list(set(X_train.columns) - set(selected_features))
            best_perf = 0
            best_attribute = ''

            print("Added feature{}".format(i))
            # For all features we can still select...
            for f in features_left:
                temp_selected_features = copy.deepcopy(selected_features)
                temp_selected_features.append(f)

                # Determine the accuracy of a decision tree learner if we were to add
                # the feature.
                pred_y_train, pred_y_test, prob_training_y, prob_test_y = ca.decision_tree(X_train[temp_selected_features],
                                                                                           y_train,
                                                                                           X_test[temp_selected_features],
                                                                                           gridsearch=False)
                perf = ce.accuracy(y_test, pred_y_test)

                # If the performance is better than what we have seen so far (we aim for high accuracy)
                # we set the current feature to the best feature and the same for the best performance.
                if perf > best_perf:
                    best_perf = perf
                    best_feature = f

            # We select the feature with the best performance.
            selected_features.append(best_feature)
            prev_best_perf = best_perf
            ordered_features.append(best_feature)
            ordered_scores.append(best_perf)

        return selected_features, ordered_features, ordered_scores

    # Backward selection for classification which selects a pre-defined number of features (max_features)
    # that show the best accuracy. We assume a decision tree learning for this purpose, but
    # this can easily be changed. It return the best features.
    def backward_selection(self, max_features, X_train, y_train):
        # First select all features.
        selected_features = X_train.columns.tolist()
        ca = ClassificationAlgorithms()
        ce = ClassificationEvaluation()
        for i in range(0, (len(X_train.columns) - max_features)):
            best_perf = 0
            worst_feature = ''

            # Select from the features that are still in the selection.
            for f in selected_features:
                temp_selected_features = copy.deepcopy(selected_features)
                temp_selected_features.remove(f)

                # Determine the score without the feature.
                pred_y_train, pred_y_test, prob_training_y, prob_test_y = ca.decision_tree(X_train[temp_selected_features], 
                                                                                           y_train, 
                                                                                           X_train[temp_selected_features])
                perf = ce.accuracy(y_train, pred_y_train)

                # If we score better without the feature than what we have seen so far
                # this is the worst feature.
                if perf > best_perf:
                    best_perf = perf
                    worst_feature = f

            # Remove the worst feature.
            selected_features.remove(worst_feature)
            
        return selected_features