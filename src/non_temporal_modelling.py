import copy
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split


# This class creates dfs splits that can be used by classification learning algorithms.
class ClassificationPrepareData:

    default_label = 'undefined'
    class_col = 'class'

    # This function creates a single class column based on a set of binary class columns, essentially merging them. 
    def assign_label(
        self, 
        df: pd.DataFrame, 
        class_labels: list[str],
    ) -> pd.DataFrame:
        
        # Find which columns are relevant based on the possibly partial class_label specification.
        labels = []
        for i in range(0, len(class_labels)):
            labels.extend([name for name in list(df.columns) if class_labels[i] == name[0:len(class_labels[i])]])

        # Determine how many class values are label as 'true' in our class columns.
        sum_values = df[labels].sum(axis=1)
        # Create a new 'class' column and set the value to the default class.
        df[self.class_col] = self.default_label
        for i in range(0, len(df.index)):
            # If we have exactly one true class column, we can assign that value, otherwise we keep the default class.
            if sum_values[i] == 1:
                df.iloc[i, df.columns.get_loc(self.class_col)] = df[labels].iloc[i].idxmax(axis=0)
        
        # And remove our old binary columns.
        df = df.drop(labels, axis=1)

        return df
    
    @staticmethod
    def binarize( 
        df: pd.DataFrame, 
        class_labels: list[str]
    ) -> pd.DataFrame:
            
        # Binarize the class labels.
        df = df.drop(class_labels[:-1], axis=1)
        df[class_labels[-1]] = df[class_labels[-1]].apply(
            lambda x: class_labels[-1] if x == 1 \
                else (str("Non ") + class_labels[-1])
        )
        
        return df

    def split_classification(
        self, 
        df: pd.DataFrame, 
        class_labels: list[str], 
        matching: str = "like", 
        temporal: bool = False,
        training_frac: float = 0.7, 
        filter: bool = True,  
        random_state: int = 0
    )-> pd.DataFrame:

        # Create a single class column if we have the 'like' option.
        if matching == 'like':
            df = self.assign_label(df, class_labels)
            class_labels = self.class_col
        elif matching == 'binary':
            df = self.binarize(df, class_labels)
            class_labels = class_labels[-1]
        elif len(class_labels) == 1:
            class_labels = class_labels[0]

        # Filter NaN is desired and those for which we cannot determine the class should be removed.
        if filter:
            df = df.dropna()
            df = df[df[class_labels] != self.default_label]

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
            df_head = df.drop_duplicates("ID", ignore_index=True)
            training_set_X, test_set_X, training_set_y, test_set_y = train_test_split(
                            df_head.iloc[:, features],
                            df_head.iloc[:, class_label_indices],
                            test_size=(1-training_frac),
                            stratify=df_head.iloc[:, class_label_indices],
                            random_state=random_state,
                        )
            training_id = training_set_X["ID"].values
            testing_id = test_set_X["ID"].values
            training_set_X, training_set_y = (
                df[df["ID"].isin(training_id)].iloc[:, features],
                df[df["ID"].isin(training_id)].iloc[:, class_label_indices],
            )
            test_set_X, test_set_y = (
                df[df["ID"].isin(testing_id)].iloc[:, features],
                df[df["ID"].isin(testing_id)].iloc[:, class_label_indices],
            )

            for df_i in [training_set_X, test_set_X]:
                df_i = df_i.drop(["ID"], axis=1, inplace=True)

        print('Training set length is: ', len(training_set_X.index))
        print('Test set length is: ', len(test_set_X.index))
        
        return training_set_X, test_set_X, training_set_y, test_set_y

# Class for evaluation metrics of classification problems.
class ClassificationEvaluation:

    @staticmethod
    def accuracy(y_true, y_pred):
        return metrics.accuracy_score(y_true, y_pred)

    @staticmethod
    def precision(y_true, y_pred):
        return metrics.precision_score(y_true, y_pred, average=None)

    @staticmethod
    def recall(y_true, y_pred):
        return metrics.recall_score(y_true, y_pred, average=None)

    @staticmethod
    def f1(y_true, y_pred):
        return metrics.f1_score(y_true, y_pred, average=None)

    @staticmethod
    def auc(y_true, y_pred_prob):
        return metrics.roc_auc_score(y_true, y_pred_prob)

    @staticmethod
    def confusion_matrix(y_true, y_pred, labels):
        return metrics.confusion_matrix(y_true, y_pred, labels=labels)

    @staticmethod
    def confusion_matrix_visualize(cm, labels, filepath):
        cm = cm / np.expand_dims(np.sum(cm, axis = 1), axis = 1)
        df_cm = pd.DataFrame(cm, index = labels, columns = labels)
        plt.figure(figsize = (9,5))
        sn.set(font_scale=1)
        sn.heatmap(df_cm, cmap="crest", annot=True, annot_kws={"size": 10})
        plt.savefig(f"{filepath}")

# Class features the most popular and best-performing classification algorithms.
class ClassificationProcedure:

    def __init__(
        self,
        df: pd.DataFrame,
        class_labels: list[str],
        matching: str = "like",
        temporal: bool = False,
        selected: list[str] = None,
    ):
        prepare = ClassificationPrepareData()
        (
            self.train_X, 
            self.test_X, 
            self.train_y, 
            self.test_y
        ) = prepare.split_classification(df, class_labels, matching, temporal)
        self.select = selected if selected is not None \
            else self.train_X.columns

    def feedforward_neural_network(
        self, 
        hidden_layer_sizes: int = (100,), 
        max_iter: int = 500, 
        activation: str = 'logistic', 
        alpha: float = 0.0001, 
        learning_rate: str = 'adaptive', 
        gridsearch: bool = True, 
        print_model_details: bool = False
    ) -> pd.DataFrame:
        (
            train_X, 
            test_X, 
            train_y 
        ) = self.train_X[self.select], self.test_X[self.select], self.train_y

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

    def support_vector_machine(
        self, 
        C: int = 1,  
        kernel: str = 'rbf', 
        gamma: float = 1e-3, 
        gridsearch: bool = True, 
        print_model_details: bool = False
    ) -> pd.DataFrame:
        (
            train_X, 
            test_X, 
            train_y 
        ) = self.train_X[self.select], self.test_X[self.select], self.train_y

        if gridsearch:
            tuned_parameters = [{'kernel': ['linear', 'sigmoid', 'poly', 'rbf'], 
                                 'gamma': [1e-3, 1e-4],
                                 'C': [1, 10, 100],
                                 'class_weight':['balanced']}]
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
    
    def random_forest(
        self, 
        n_estimators: int = 10, 
        min_samples_leaf: int = 5, 
        criterion: str ='gini', 
        print_model_details: bool = False, 
        gridsearch: bool = True
    ) -> pd.DataFrame:
        (
            train_X, 
            test_X, 
            train_y 
        ) = self.train_X[self.select], self.test_X[self.select], self.train_y

        if gridsearch:
            tuned_parameters = [{'min_samples_leaf': [2, 10, 50, 100, 200],
                                 'n_estimators':[10, 50, 100],
                                 'criterion':['gini', 'entropy'],
                                 'class_weight':['balanced']}]
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
    
    @staticmethod
    def decision_tree(
        train_X: pd.DataFrame, 
        train_y: pd.DataFrame, 
        test_X: pd.DataFrame, 
        min_samples_leaf: int = 50, 
        criterion: str = 'gini'
    ) -> pd.DataFrame:
        dtree = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf, criterion=criterion)
        dtree.fit(train_X, train_y.values.ravel())

        # Apply the model
        pred_prob_training_y = dtree.predict_proba(train_X)
        pred_prob_test_y = dtree.predict_proba(test_X)
        pred_training_y = dtree.predict(train_X)
        pred_test_y = dtree.predict(test_X)
        frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=dtree.classes_)
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=dtree.classes_)

        return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y

    def forward_selection(
        self,
        max_features: int = 10
    ) -> list:
        (
            train_X, 
            test_X, 
            train_y, 
            test_y 
        ) = self.train_X[self.select], self.test_X[self.select], self.train_y, self.test_y

        # Start with no features.
        ordered_features, ordered_scores, selected_features  = [[] for _ in range(3)]

        # Select the appropriate number of features.
        for i in range(0, max_features):
            # Determine the features left to select.
            features_left = list(set(train_X.columns) - set(selected_features))
            best_perf = 0

            print("Added feature {}".format(i))
            # For all features we can still select...
            for f in features_left:
                temp_selected_features = copy.deepcopy(selected_features)
                temp_selected_features.append(f)

                # Determine the accuracy of a decision tree learner if we were to add the feature.
                _, pred_y_test, _, _ = self.decision_tree(train_X[temp_selected_features],
                                                          train_y,
                                                          test_X[temp_selected_features])
                perf = ClassificationEvaluation.accuracy(test_y, pred_y_test)

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

        print("The {} most important features, in order:".format(max_features))
        for name in ordered_features:
            print("  - {}".format(name))

        return selected_features, ordered_features, ordered_scores