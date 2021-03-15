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
    st.subheader('Machine Learning')

    model = joblib.load('data/best_model.pkl')

    pre = st.number_input('임신횟수를 입력하세요.', min_value=0)

    glu = st.number_input('공복혈당을 입력하세요.', min_value=0)

    bp = st.number_input('혈압을 입력하세요.', min_value=0)

    sti = st.number_input('피부두께를 입력하세요.',min_value=0)

    ins = st.number_input('인슐린 농도를 입력하세요.')

    BMI = st.number_input('BMI를 입력하세요')

    DPF = st.number_input('당뇨 내력 가중치를 입력하세요.')

    age = st.number_input('나이를 입력하세요.',min_value=0, max_value = 120)

    new_data = np.array( [pre, glu, bp, sti, ins, BMI, DPF, age ] )
    new_data = new_data.reshape(1,-1)
    # print(new_data)

    y_pred = model.predict(new_data)

    if st.button('결과 확인'):
        if y_pred == 1:
            st.write('당뇨병이 의심됩니다. 혈당관리가 필요 합니다.')
        else :
            st.write('당뇨병이 의심되지 않습니다. 건강합니다.')




