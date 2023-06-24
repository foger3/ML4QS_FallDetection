import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy import stats

# This class creates dfs splits that can be used by classification learning algorithms.
class ClassificationPrepareData:

    default_label = 'undefined'
    class_col = 'class'
    
    @staticmethod
    def binarize( 
        df: pd.DataFrame, 
        class_labels: list[str]
    ) -> pd.DataFrame:
            
        # Binarize the class labels.
        df = df.drop(class_labels[1:], axis=1)
        df[class_labels[0]] = df[class_labels[0]].apply(
            lambda x: class_labels[0] if x == 1 \
                else (str("Non ") + class_labels[0])
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
        
        # desirable to drop NAs and instances with multiple classes
        if filter: 
            df = df.dropna()
            df = df[df[class_labels].sum(axis=1) < 2]

        # Create a single class column if we have the 'like' option.
        if matching == 'like':
            df[self.class_col] = df[class_labels].idxmax(axis=1)
            df = df.drop(class_labels, axis=1)
            class_labels = self.class_col
        elif matching == 'binary':
            df = self.binarize(df, class_labels)
            class_labels = class_labels[0]
        elif len(class_labels) == 1:
            class_labels = class_labels[0]

        # The features are the ones not in the class label.
        features = [df.columns.get_loc(x) for x in df.columns if x not in class_labels]
        class_label_indices = [df.columns.get_loc(x) for x in df.columns if x in class_labels]

        # For temporal data, we select the desired fraction of training data from the first part and use the rest as test set.
        if temporal:
            training_set_X, test_set_X, training_set_y, test_set_y = train_test_split(
                df.iloc[:, features],
                df.iloc[:, class_label_indices],
                test_size = (1- training_frac), 
                shuffle=False
            )

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
    
    def data_reshape(
        self,
        df: pd.DataFrame, 
        class_labels: list[str],  
        step: int = 5,
        time_intervals: int = 50,          
    ) -> np.ndarray:
        sensor_labels = [col for col in df.columns if col not in class_labels]

        # Create a single class column to extract mode
        df[self.class_col] = df[class_labels].idxmax(axis=1)
        df[self.class_col] = pd.factorize(df[self.class_col], sort=True)[0]

        # Reshape input for neural network in to [samples, time_intervals, features]
        segments = []
        labels = []
        for i in range(0,  df.shape[0]- time_intervals, step):  
            feat_temp = []
            for feature in sensor_labels:
                xs = df[feature].values[i: i + time_intervals]
                feat_temp.append(xs)
            label = stats.mode(df[self.class_col][i: i + time_intervals], keepdims=True)[0][0]

            segments.append(feat_temp)
            labels.append(label)

        reshaped_segments = np.asarray(
            segments, dtype= np.float32
        ).reshape(-1, time_intervals, len(sensor_labels))
        labels = np.asarray(pd.get_dummies(labels), dtype = np.float32)

        return reshaped_segments, labels
        
    def split_classification_nn(
        self,
        df: pd.DataFrame, 
        class_labels: list[str],  
        step: int = 5,
        time_intervals: int = 50,
        training_frac: float = 0.7, 
        filter: bool = True
    ) -> np.ndarray:
        # desirable to drop NAs and those with multiple classes
        if filter: 
            df = df.dropna()
            df = df[df[class_labels].sum(axis=1) < 2]

        reshaped_x, reshaped_y = self.data_reshape(df, class_labels, step, time_intervals)

        # Only option for recurrent neural network to split temporally
        X_train, X_test, y_train, y_test = train_test_split(
            reshaped_x, 
            reshaped_y, 
            test_size = (1- training_frac), 
            shuffle=False
        )
        
        return X_train, X_test, y_train, y_test