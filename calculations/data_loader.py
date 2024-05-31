import pickle
import pandas as pd
import io
import calculations.column_functions as column_functions
from panel.viewable import Viewer
import functions as feature


class DataLoader(Viewer):
    def __init__(self, file=None, nn_file=None):
        super().__init__()
        if file is None or nn_file is None:
            self.data = load_weather_data()[0:200]
            self.nn = load_weather_nn()
        else:
            self.data = load_data(file)[0:200]
            self.nn = load_nn(nn_file)

        self.means = feature.get_means(self.data)
        self.classes = self.nn.classes_
        self.columns = [col for col in self.data.columns]


def load_weather_data():
    # load weather files
    file_testdata = open('weather_data/weather_testdata.csv', 'rb')
    testdata = pd.read_csv(file_testdata)

    return testdata


def load_weather_nn():
    # load weather files
    file_nn = open('weather_data/weather_nn.pkl', 'rb')
    nn = pickle.load(file_nn)
    file_nn.close()

    return nn


def load_data(file_data):
    return pd.read_csv(io.BytesIO(file_data))


def load_nn(file_nn):
    return pickle.load(io.BytesIO(file_nn))
