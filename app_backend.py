#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 22:52:13 2019

@author: zeyuyan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 18:04:17 2019

@author: zeyuyan
"""

# Import necessary dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import h5py
import pickle
import json
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from keras.layers import Input, Dense, Dropout, LSTM
from keras.models import Model, Sequential
from keras import optimizers
from keras.utils import to_categorical


class Model_Generation:
    def __init__(self):
        self.df = pd.read_csv("data/correct_fred_data.csv")
    
    def preprocess_df(self):
        """
        Return a preprocessed dataframe, date is from earliest to latest
        """
        df_copy = copy.deepcopy(self.df)
        
        # Remove duplicated date columns
        date_index_max = int((df_copy.shape[1] / 2) - 1)
        
        remove_list = []
        for i in range(1, date_index_max + 1):
            remove_list.append("DATE." + str(i))
        df_copy.drop(remove_list, axis=1, inplace=True)
        
        # Rename the columns
        for i in range(1, df_copy.shape[1]):
            colname = df_copy.columns[i]
            new_colname = colname.split("/")[1]
            df_copy.rename(columns={df_copy.columns[i]: new_colname}, inplace=True)
        
        # Remove the rows after 5/20/2019
        index_520 = df_copy[df_copy["DATE"] == "5/20/2019"].index.values[0]
        remove_index_list = list(df_copy.index[:index_520])
        df_copy.drop(remove_index_list, inplace=True)
        df_copy.reset_index(drop=True, inplace=True)
        
        # Remove the data before year 2002
        index_1999_last = df_copy[df_copy["DATE"] == "12/31/2002"].index.values[0]
        remove_index_list = list(df_copy.index[index_1999_last:])
        df_copy.drop(remove_index_list, inplace=True)
        
        # Deal with NaN values
        df_copy.fillna(method="bfill", inplace=True)
        
        # Remove columns with many Nan values
        num_nas = df_copy.isna().sum()
        remove_index_list = list(num_nas[num_nas > 100].index)
        df_copy.drop(remove_index_list, axis=1, inplace=True)
        
        # Further remove rows with NaN values
        df_copy.dropna(how="any", inplace=True)
        df_copy.reset_index(drop=True, inplace=True)
        
        # Reverse the df
        reversed_df = df_copy.iloc[::-1]
        reversed_df.reset_index(drop=True, inplace=True)
        
        return reversed_df
    
    def get_dates(self):
        """
        Return a list of all of the available dates from reversed_df
        """
        df = self.preprocess_df()
        full_dates_list = list(df["DATE"])
        return full_dates_list
        
    def generate_DFF_plot(self):
        """
        Generate and save the plot of DFF vs. DATE
        """
        df = self.preprocess_df()
        ax = df.plot(figsize=(16, 8), x="DATE", y="DFF", legend=False)
        ax.set_ylabel("DFF")
        plt.savefig(os.getcwd() + "/images/DFF.png")
    
    def generate_X_df(self):
        """
        Return a Dataframe without DFF, set DATE as index
        """
        df = self.preprocess_df()
        minus_DFF_df = df.drop(["DFF"], axis=1)
        df_X = minus_DFF_df.set_index("DATE")
        return df_X
    
    def generate_Y_series(self):
        """
        Return a Series of DFF values
        """
        df = self.preprocess_df()
        DFF_series = df["DFF"]
        return DFF_series
    
    def correlation_matrix_plot(self):
        """
        Generate and save the plot of correlation matrix
        """
        df_X = self.generate_X_df()
        col_names = list(df_X.columns)
        correlations = df_X.corr()
        fig = plt.figure(figsize=(18, 16))
        ax = fig.add_subplot(111)
        cax = ax.matshow(correlations, vmin=-1, vmax=1)
        fig.colorbar(cax)
        ticks = np.arange(0, len(col_names), 1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(col_names, rotation=90)
        ax.set_yticklabels(col_names)
        plt.show()
        plt.savefig(os.getcwd() + "/images/correlation_matrix.png")
    
    def genetate_feature_matrix(self, step):
        """
        Return an multi-dimensional np array, as the feature matrix to train the models
        Step means how many days of data is used for the predictions
        For example, if step=2, this means 1st and 2nd days' data is used to predict the variation trend
        of the DDF value of the 3rd day compared to the 2nd day
        """
        df_X = self.generate_X_df()
        DFF_series = self.generate_Y_series()
        feature_matrix = np.zeros((df_X.shape[0] - step, step * (df_X.shape[1] + 1)))
        for i in range(len(feature_matrix)):
            for j in range(step):
                if j == 0:
                    row_array = np.append(df_X.values[i + j], DFF_series.values[i + j])
                else:
                    row_array = np.append(row_array, df_X.values[i + j])
                    row_array = np.append(row_array, DFF_series.values[i + j])
            feature_matrix[i] = row_array  
        
        return feature_matrix
 
    def generate_outputs(self, step):
        """
        Return an multi-dimensional np array
        0th column: variation trend, 0: DFF up; 1: DFF keeps the same, 2: DFF down
        1st column: the DFF value of the previous day
        2nd column: the DFF value of the day of interest
        """
        DFF_series = self.generate_Y_series()
        outputs_array = np.zeros((len(DFF_series) - step, 3))
        for i in range(len(outputs_array)):
            y_next_day = DFF_series.values[i + step]
            y_prev_day = DFF_series.values[i + step -1]
            if y_next_day > y_prev_day:
                outputs_array[i, 0] = 0
            elif y_next_day == y_prev_day:
                outputs_array[i, 0] = 1
            else:
                outputs_array[i, 0] = 2
            outputs_array[i, 1] = y_prev_day
            outputs_array[i, 2] = y_next_day
                
        return outputs_array
    
    def train_test_gen(self, step, scaler=MinMaxScaler()):
        """
        Return the train and test sets
        """
        feature_matrix = self.genetate_feature_matrix(step)
        feature_matrix_rescaled = scaler.fit_transform(feature_matrix)
        outputs_array = self.generate_outputs(step)
        
        X_train, X_test, Y_train, Y_test = train_test_split(
                                feature_matrix_rescaled, outputs_array[:, 0], 
                                test_size=0.2, 
                                random_state=23
                                )
        
        return X_train, X_test, Y_train, Y_test
    
    def non_NN_algorithms_spot_check(self, step):
        """
        Compare series of machine learning algorithms, including:
        LR, KNN, CART, NB, SVM
        """
        X_train, X_test, Y_train, Y_test = self.train_test_gen(step)
        
        # Save the dates
        full_dates_list = self.get_dates()
        total_len = len(X_train) + len(X_test)
        dates_list = full_dates_list[step:(total_len + step)]
        
        # Spot-check algorithms
        models = []
        models.append(('LR', LogisticRegression())) 
        models.append(('KNN', KNeighborsClassifier())) 
        models.append(('CART', DecisionTreeClassifier())) 
        models.append(('NB', GaussianNB())) 
        models.append(('SVM', SVC(probability=True)))
    
        results = []
        names = []
        for name, model in models:
            kfold = KFold(n_splits=10, random_state=7)
            cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
            results.append(cv_results)
            names.append(name)
            print(f"{name}: {cv_results.mean()} ({cv_results.std()})")
            
            # Save the model
            model.fit(X_train, Y_train)
            with open(f"models/classifiers/{name}/model/model.pickle", "wb") as f:
                pickle.dump(model, f)
            print(f"{name} model saved to disk...")
            
            # Evaluate the model on the test sets
            result = model.score(X_test, Y_test)
            print(f"Test set acc: {result}")
            
            with open(f"data/json_files/dates/classifiers/{name}/dates.json", "w") as json_file:
                json.dump(dates_list, json_file)
            
            
    def naive_NN_classifier(self, step):
        """
        Naive Neural Network model
        """
        X_train, X_test, Y_train, Y_test = self.train_test_gen(step)
        Y_train_oh = to_categorical(Y_train)
        Y_test_oh = to_categorical(Y_test)
        
        # Save the dates
        full_dates_list = self.get_dates()
        total_len = len(X_train) + len(X_test)
        dates_list = full_dates_list[step:(total_len + step)]
        
        with open(f"data/json_files/dates/classifiers/naive_NN/dates.json", "w") as json_file:
            json.dump(dates_list, json_file)
            
        # Basic NN model
        num_features = X_train.shape[1]
        num_classes = Y_train_oh.shape[1]
        
        inputs = Input(shape=(num_features, ))
        output_hid_1 = Dense(700, kernel_initializer='random_normal', activation='relu')(inputs)
        output_hid_2 = Dense(700, kernel_initializer='random_normal', activation='relu')(output_hid_1)
        predictions = Dense(num_classes, kernel_initializer='random_normal', activation='softmax')(output_hid_2)
        model = Model(inputs=inputs, outputs=predictions)
        
        print(model.summary())
        
        # Compile the model
        model.compile(loss='categorical_crossentropy', 
                      optimizer=optimizers.Adam(lr=0.001), 
                      metrics=['accuracy'])
        
        # Fit the model
        model.fit(X_train, Y_train_oh, 
                  batch_size=32, 
                  epochs=50)
        
        # Evaluate the model on the test sets
        scores = model.evaluate(X_test, Y_test_oh)
        print(f"Test set loss: {scores[0]}")
        print(f"Test set acc: {scores[1]}")
        
        # Save the model to a json file
        model_json = model.to_json()
        with open("models/classifiers/naive_NN/model/model.json", "w") as json_file:
            json_file.write(model_json)
        print("Saved model to disk...")
        
        # Serialize weights to HDF5
        model.save_weights("models/classifiers/naive_NN/weights/weights.h5")
        print("Saved weights to disk...")
        
    def LSTM_classifier(self, step):
        """
        Recurrent Neural Network (LSTM) model
        """
        X_train, X_test, Y_train, Y_test = self.train_test_gen(step)
        Y_train_oh = to_categorical(Y_train)
        Y_test_oh = to_categorical(Y_test)
        
        # Reshape the inputs
        X_train_reshaped = X_train.reshape(X_train.shape[0], step, int(X_train.shape[1] / step))
        
        # Save the dates
        full_dates_list = self.get_dates()
        total_len = len(X_train) + len(X_test)
        dates_list = full_dates_list[step:(total_len + step)]
        
        with open(f"data/json_files/dates/classifiers/LSTM/dates.json", "w") as json_file:
            json.dump(dates_list, json_file)
        
        # LSTM model (many-to-one model)
        model = Sequential()
        model.add(LSTM(50, input_shape=(step, X_train_reshaped.shape[2])))
        model.add(Dense(Y_train_oh.shape[1], activation='softmax'))
        
        print(model.summary())
        
        # Compile the model
        model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
        
        # Fit the model
        model.fit(X_train_reshaped, Y_train_oh, batch_size=32, epochs=50)
        
        # Reshape the test input
        X_test_reshaped = X_test.reshape(X_test.shape[0], step, int(X_test.shape[1] / step))
        
        # Evaluate the model on the test sets
        scores = model.evaluate(X_test_reshaped, Y_test_oh)
        print(f"Test set loss: {scores[0]}")
        print(f"Test set acc: {scores[1]}")
        
        # Save the model to a json file
        model_json = model.to_json()
        with open("models/classifiers/LSTM/model/model.json", "w") as json_file:
            json_file.write(model_json)
        print("Saved model to disk...")
        
        # Serialize weights to HDF5
        model.save_weights("models/classifiers/LSTM/weights/weights.h5")
        print("Saved weights to disk...")
        
    def find_input_output_data_by_date(self, model_abbr, date, step, scaler=MinMaxScaler()):
        """
        This function is used to generate the JSON data
        """
        feature_matrix = self.genetate_feature_matrix(step)
        feature_matrix_rescaled = scaler.fit_transform(feature_matrix)
        outputs_array = self.generate_outputs(step)
        full_dates_list = self.get_dates()
        dates_list = full_dates_list[step:(len(feature_matrix) + step)]
        
        for i in range(len(dates_list)):
            if dates_list[i] == date:
                index = i
                break
        
        if model_abbr == "LSTM":
            X_in = feature_matrix_rescaled.reshape(feature_matrix_rescaled.shape[0], step, int(feature_matrix_rescaled.shape[1] / step))[index]
            X_in = X_in.reshape((1, X_in.shape[0], X_in.shape[1]))
        else:
            X_in = feature_matrix_rescaled[index]
        
        Y_to_pred = outputs_array[index, 2]
        Y_prev = outputs_array[index, 1]
        
        return X_in, Y_to_pred, Y_prev
    
    def generate_col_names(self):
        """
        Returns a list of column names
        """
        df_X = self.generate_X_df()
        col_list = list(df_X.columns)
        
        return col_list
        

if __name__ == "__main__":
    MG = Model_Generation()
    MG.generate_DFF_plot()
    MG.correlation_matrix_plot()
    MG.non_NN_algorithms_spot_check(2)
    MG.naive_NN_classifier(2)
    MG.LSTM_classifier(7)
