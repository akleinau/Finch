import pickle
import pandas as pd
import io
from panel.viewable import Viewer
from calculations import shap_set_functions


class DataLoader(Viewer):
    def __init__(self, file=None, nn_file=None):
        super().__init__()
        if file is None or nn_file is None:
            self.data = load_weather_data()[0:1000]
            self.nn = load_weather_nn()
        else:
            self.data = load_data(file)[0:1000]
            self.nn = load_nn(nn_file)

        self.means = get_means(self.data)
        self.classes = self.nn.classes_
        self.columns = [col for col in self.data.columns]
        self.data_and_probabilities = self.combine_data_and_results()

    def combine_data_and_results(self):
        classes = self.nn.classes_
        all_predictions = self.nn.predict_proba(self.data)
        all_predictions = pd.DataFrame(all_predictions, columns=['prob_' + str(name) for name in classes])
        all_predictions['prediction'] = all_predictions.idxmax(axis=1)
        # merge X_test, shap, predictions
        all_data = pd.concat([self.data, all_predictions], axis=1)
        return all_data


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


def get_means(data):
    return data.mean().to_frame().T
