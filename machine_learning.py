import pickle
import random
import pandas as pd
import numpy as np

# importing the model

def predict_iris():
    # loading the iris dataset
    iris_data = pd.read_csv('Iris.csv')

    # converting the csv to df
    iris_df = pd.DataFrame(iris_data)

    # simulating values using random max and min from dataset
    SepalLengthCm = round(random.uniform(iris_df['SepalLengthCm'].min(),
                                         iris_df['SepalLengthCm'].max()), 1)

    SepalWidthCm = round(random.uniform(iris_df['SepalWidthCm'].min(),
                                         iris_df['SepalWidthCm'].max()), 1)

    PetalLengthCm = round(random.uniform(iris_df['PetalLengthCm'].min(),
                                      iris_df['PetalLengthCm'].max()), 1)

    PetalWidthCm = round(random.uniform(iris_df['PetalWidthCm'].min(),
                                      iris_df['PetalWidthCm'].max()), 1)

    loaded_model = pickle.load(open('svm_iris.pkl', 'rb'))
    str = np.array2string(loaded_model.predict([[SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]]))
    return str

predict_iris()