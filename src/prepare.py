import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn import metrics
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

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


class TCNDataset(Dataset):
    def __init__(self,
        df: pd.DataFrame = None,
        class_columns: list[str] = None,
        split: str = "train",
    ):

        if split == "train":
            X, y, _, _ = self.data_split(df, class_columns)
        elif split == "test":
            _, _, X, y = self.data_split(df, class_columns)
        else:
            raise ValueError("split must be either 'train' or 'test'")
        
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values).squeeze()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def data_split(
        self,
        df: pd.DataFrame = None, 
        class_columns: list[str] = None, 
        filter: bool = True,
        not_hot: bool = True,
        train_frac: float = 0.7
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        
        if filter:
            df = df.dropna()
            df = df[df[class_columns].sum(axis=1) < 2]
            
        if not_hot:
            df["class"] = df[class_columns].idxmax(axis=1)
            df["class"] = pd.factorize(df["class"], sort=True)[0]
            
            df = df.drop(class_columns, axis=1)
            class_columns = "class"

        features = [df.columns.get_loc(x) for x in df.columns if x not in class_columns]
        class_label_indices = [df.columns.get_loc(x) for x in df.columns if x in class_columns]

        end_training_set = int(train_frac * len(df.index))
        training_set_X = df.iloc[0:end_training_set, features]
        training_set_y = df.iloc[0:end_training_set, class_label_indices]
        test_set_X = df.iloc[end_training_set:len(df.index), features]
        test_set_y = df.iloc[end_training_set:len(df.index), class_label_indices]

        return training_set_X, training_set_y, test_set_X, test_set_y