import copy
import pandas as pd
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import weight_norm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

from prepare import ClassificationPrepareData, ClassificationEvaluation


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
    

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TemporalClassification(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TemporalClassification, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        y1 = self.tcn(inputs)  
        o = self.linear(y1[:, :, -1])
        return F.log_softmax(o, dim=1)
    


