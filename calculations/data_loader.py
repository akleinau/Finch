import pickle
import pandas as pd
import io

def load_weather_data():
    # load weather files
    file_nn = open('weather_data/weather_nn.pkl', 'rb')
    nn = pickle.load(file_nn)
    file_testdata = open('weather_data/weather_testdata.csv', 'rb')
    testdata = pd.read_csv(file_testdata)
    file_nn.close()

    return nn, testdata

def load_data(file_data):
    if file_data is None:
        return load_weather_data()[1]
    else:
        return pd.read_csv(io.BytesIO(file_data))


def load_nn(file_nn):
    if file_nn is None:
        return load_weather_data()[0]
    else:
        return pickle.load(io.BytesIO(file_nn))
