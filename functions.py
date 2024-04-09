import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import shap
import pandas as pd

# do the same changes to multiple data frames
def combine_columns(df, columns):
    if len(columns) > 0:
        for i in range(len(columns)):
            #combine column names to a string
            combined_name = ', '.join(columns[i])
            #combine the columns
            df[combined_name] = df.apply(lambda x: tuple(x[columns[i]]), axis=1)
            #delete the original columns
            df = df.drop(columns[i], axis=1)
    return df

def split_columns(df, combined_name):
    #get the original names
    columns = combined_name.split(', ')
    #split the combined column
    df = pd.concat([df, df[combined_name].apply(pd.Series)], axis=1)
    #delete the combined column
    df = df.drop(combined_name, axis=1)
    #rename the newly created columns
    df.columns = df.columns[:len(df.columns)-len(columns)].tolist() + columns
    return df

def split_all_columns(df):
    tmp = df
    #split all combined columns
    for column in tmp.columns:
        if column.find(", ") >= 0:
            tmp = split_columns(tmp, column)

    return tmp

def dimreduce_columns(df, combined_name):
    #split the combined column
    df[combined_name] = df[combined_name].apply(lambda x: ', '.join(map(lambda y: str(round(y, 2)),x)))
    return df

def dimreduce_all_columns(df):
    tmp = pd.DataFrame(df)
    #split all combined columns
    for column in tmp.columns:
        if isinstance(tmp[column][0],tuple):
            tmp = dimreduce_columns(tmp, column)

    return np.array(tmp)



def calc_shap_values(data, background, model, columns):
    explainer = shap.KernelExplainer(lambda x: detangled_predict(x, model, column_names=columns), background, columns, keep_index=True)
    shap_values = explainer(data)
    shap_values.data = dimreduce_all_columns(shap_values.data)
    return shap_values

def detangled_predict(data, model, column_names):
    return model.predict(split_all_columns(data).reindex(columns=column_names))