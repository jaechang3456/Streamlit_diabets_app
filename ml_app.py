import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow.keras
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
import pickle
import os
import tensorflow as tf
import h5py
import joblib
from sklearn.impute import  SimpleImputer
from imblearn.over_sampling import  SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

def run_ml_app() :
    df = pd.read_csv('diabetes.csv')

    X = df.drop('Outcome', axis=1)
    fill = SimpleImputer(missing_values=0, strategy='mean')
    y = df['Outcome']
    X = fill.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=10)


    def FitModel(X_train,y_train,X_test,y_test,algo_name,algorithm,gridSearchParams,cv):
        np.random.seed(10)

        grid = GridSearchCV(
            estimator=algorithm,
            param_grid = gridSearchParams,
            cv = cv,
            n_jobs = -1,
            scoring = 'accuracy',
            verbose = 1,
        )
        grid_result = grid.fit(X_train, y_train)
        best_params = grid_result.best_params_

        y_pred = grid_result.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        st.write('y pred : {}'.format(y_pred))
        st.write('Best Params : {}'.format(best_params))
        st.write('Accuracy Score : {}'.format(accuracy_score(y_test, y_pred)))
        st.write('Confusion Matrix : \n{}'.format(cm))


    sm = SMOTE(random_state=10)
    X_res, y_res = sm.fit_resample(X, y)
    pd.Series(y_res).value_counts()
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size = 0.2, random_state = 10)

    param2 = { 'n_estimators' : [100, 500, 1000, 1500, 2000] }
    FitModel(X_train, y_train, X_test, y_test, 'Random Forest', RandomForestClassifier(), param2, cv=5)
    st.subheader('Machine Learning')

    pre = st.number_input('임신횟수를 입력하세요.', min_value=0)

    glu = st.number_input('공복혈당을 입력하세요.', min_value=0)

    bp = st.number_input('혈압을 입력하세요.', min_value=0)

    sti = st.number_input('피부두께를 입력하세요.',min_value=0)

    age = st.number_input('나이를 입력하세요.',min_value=0, max_value = 120)

    ins = st.number_input('인슐린 농도를 입력하세요.')

    BMI = st.number_input('BMI를 입력하세요', min_value=0)

    new_data = np.array( [pre, glu, bp, sti, age, ins, BMI ] )
    new_data = new_data.reshape(1,-1)

    y_pred = RandomForestClassifier(new_data)





