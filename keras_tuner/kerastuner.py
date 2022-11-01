import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential
import keras_tuner as kt
from sklearn.preprocessing import StandardScaler

def build_model(hp):
    model1 = Sequential()
    model1.add(Dense(32,activation='relu',input_dim =8))
    model1.add(Dense(1,activation='sigmoid'))
    
    optimizer = hp.Choice('optimizer',values = ['adam','sgd','rmsprop','adadelta'])
    
    model1.compile(loss ='binary_crossentropy',optimizer = optimizer,metrics=['accuracy'])
    return model1

df = pd.read_csv("/home/dell/Desktop/My_learning/DL_tutorial/keras_tuner/diabetes.csv")
# df.shape
print(df.head())
X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values    
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)

tuner = kt.RandomSearch(build_model,objective="val_accuracy",max_trials=10,project_name="my_project")
tuner.search(X_train,y_train,validation_data = (X_test,y_test),epochs=10)
print(tuner.get_best_hyperparameters()[0].values)