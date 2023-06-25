import copy
import numpy as np 
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras import layers, optimizers

from prepare import ClassificationPrepareData


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
class NonTemporalClassification:

    def __init__(
        self,
        df: pd.DataFrame,
        class_labels: list[str],
        matching: str = "like",
        temporal: bool = True,
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
            tuned_parameters = [{'kernel': ['rbf'], 
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

        # Apply the model
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
        train_X, train_y = self.train_X, self.train_y

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
                pred_y_train, _, _, _ = self.decision_tree(train_X[temp_selected_features],
                                                           train_y,
                                                           train_X[temp_selected_features])
                perf = ClassificationEvaluation.accuracy(train_y, pred_y_train)

                # If the performance is better than what we have seen so far (we aim for high accuracy)
                # we set the current feature to the best feature and the same for the best performance.
                if perf > best_perf:
                    best_perf = perf
                    best_feature = f

            # We select the feature with the best performance.
            selected_features.append(best_feature)
            ordered_features.append(best_feature)
            ordered_scores.append(best_perf)

        print("The {} most important features, in order:".format(max_features))
        for name in ordered_features:
            print("  - {}".format(name))

        return selected_features, ordered_features, ordered_scores


# Class features two recurrent neural network models: 
class TemporalClassification:

    def __init__(
        self,
        df: pd.DataFrame,
        class_labels: list[str],
        step: int = 5,
        time_intervals: int = 50
    ):    
        if time_intervals % 5 != 0:
            raise ValueError("Time intervals must be divisible by 5.")
        
        self.model = None
        prepare = ClassificationPrepareData()
        (
            self.train_X, 
            self.test_X,
            self.train_y, 
            self.test_y
        ) = prepare.split_classification_nn(df, class_labels, step, time_intervals)
        self.input = [self.train_X.shape[1], self.train_X.shape[2]]
        
    def lstm(self, print_model_details: bool = False) -> None:
        model = Sequential([
            layers.LSTM(units=128, input_shape=(self.input[0], self.input[1])), # RNN layer
            layers.Dropout(0.5), # Dropout layer for regularization
            layers.Dense(units=64, activation='relu'), # Hidden dense layer with ReLu
            layers.Dense(self.train_y.shape[1], activation='softmax') # Softmax layer
        ])
        
        if print_model_details:
            model.summary()

        # Prepared for fitting
        self.model = model

    def conv_lstm(self, print_model_details: bool = False) -> None:
        model = Sequential([
            layers.Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(self.input[0], self.input[1])), # Convolutional layer
            layers.LSTM(units=128),
            layers.Dropout(0.5), 
            layers.Dense(units=64, activation='relu'), 
            layers.Dense(self.train_y.shape[1], activation='softmax')
        ])
        
        if print_model_details:
            model.summary()

        # Prepared for fitting
        self.model = model

    def time_conv_lstm(self, print_model_details: bool = False) -> None:

        # Reshape into smaller time segments for time distribution
        n_steps, n_length = 5, (self.input[0] // 5)
        self.train_X = self.train_X.reshape((self.train_X.shape[0], n_steps, n_length, self.input[1]))
        self.test_X = self.test_X.reshape((self.test_X.shape[0], n_steps, n_length, self.input[1]))

        model = Sequential([
            layers.TimeDistributed(
                layers.Conv1D(filters=32, kernel_size=3, activation='relu'), input_shape=(None, n_length, self.input[1])
            ),
            layers.TimeDistributed(
                layers.Conv1D(filters=32, kernel_size=3, activation='relu')
            ),
            layers.TimeDistributed(layers.Dropout(0.5)),
            layers.TimeDistributed(layers.MaxPooling1D(pool_size=2)),
            layers.TimeDistributed(layers.Flatten()),
            layers.LSTM(units=128),
            layers.Dropout(0.5), 
            layers.Dense(units=64, activation='relu'), 
            layers.Dense(self.train_y.shape[1], activation='softmax')
        ])
        
        if print_model_details:
            model.summary()

        # Prepared for fitting
        self.model = model

    def gru(self, print_model_details: bool = False) -> None:
        model = Sequential([
            layers.GRU(units=128, input_shape=(self.input[0], self.input[1])), # RNN layer
            layers.Dropout(0.5), # Dropout layer for regularization
            layers.Dense(units=64, activation='relu'), # Hidden dense layer with ReLu
            layers.Dense(self.train_y.shape[1], activation='softmax') # Softmax layer
        ])
        
        if print_model_details:
            model.summary()

        # Prepared for fitting
        self.model = model
        
    def fit(
        self, 
        model: Sequential = None,  
        epochs: int = 50,
        batch_size: int = 128,
        learning_rate: float = 0.002,
        l2_reg: float = 0.0015,
    ):
        if self.model is not None:
            model = self.model
        else:
            raise Exception('No model has been set. Please run a model method first e.g., class.lstm()')
        
        train_X, test_X, train_y, test_y = self.train_X, self.test_X, self.train_y, self.test_y 
        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate, decay=l2_reg),
            loss='categorical_crossentropy', 
            metrics=['accuracy']
        )

        print("Training model...")
        history = model.fit(train_X, 
                            train_y, 
                            epochs = epochs, 
                            validation_split = 0.20,
                            batch_size = batch_size, 
                            verbose = 1)

        print("Evaluating model...")
        loss, accuracy = model.evaluate(test_X, 
                                        test_y, 
                                        batch_size = batch_size, 
                                        verbose = 1)
        
        print("Test Accuracy :", accuracy)
        print("Test Loss :", loss)

        return model, history
    
    def prediction(self, model: Sequential = None,) -> np.ndarray:
        test_X, test_y = self.test_X, self.test_y

        print("Predicting...")
        predictions = model.predict(test_X)

        # Get original & predicted classes 
        max_test = np.argmax(test_y, axis=1)
        max_pred = np.argmax(predictions, axis=1)

        return confusion_matrix(max_test, max_pred)

    def fit_predict(
        self,
        model: str = "lstm",
        epochs: int = 50,
        batch_size: int = 128,
        learning_rate: float = 0.002,
        l2_reg: float = 0.0015,
        print_model_details: bool = False,
    ):
        if model == "lstm":
            self.lstm(print_model_details=print_model_details)
        elif model == "conv_lstm":
            self.conv_lstm(print_model_details=print_model_details)
        elif model == "time_conv_lstm":
            self.time_conv_lstm(print_model_details=print_model_details)
        elif model == "gru":
            self.gru(print_model_details=print_model_details)
        else:
            raise Exception('Model not recognised. Please use one of the following: lstm, conv_lstm, time_conv_lstm, gru')
        
        model, history = self.fit(
            epochs = epochs,
            batch_size = batch_size,
            learning_rate = learning_rate,
            l2_reg = l2_reg
        )

        results = self.prediction(model)

        return model, history, results