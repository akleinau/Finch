import pickle
import pandas as pd
import io
from panel.viewable import Viewer


class DataLoader(Viewer):
    def __init__(self, file=None, nn_file=None, truth_file=None):
        super().__init__()
        if file is None or nn_file is None:
            self.data = load_weather_data()
            self.columns = [col for col in self.data.columns]
            self.nn = load_weather_nn()
            truth = load_weather_truth()

        else:
            self.data = load_data(file)
            self.nn = load_nn(nn_file)
            self.columns = [col for col in self.data.columns]
            truth = None
            if truth_file is not None:
                truth = load_data(truth_file)

        self.type = 'classification' if hasattr(self.nn, 'classes_') else 'regression'

        if truth is not None:
            if self.type == 'classification':
                self.data["truth"] = truth
                for label in set(truth.iloc[:, 0].values):
                    col_name = 'truth_' + str(label)
                    self.data[col_name] = (truth == label)
                    self.data[col_name] = self.data[col_name].apply(lambda x: 1 if x else 0)
            else:
                self.data["truth"] = truth
                self.data["truth_Y"] = truth

        self.means = get_means(self.data[self.columns])

        self.predict = self.nn.predict_proba if self.type == 'classification' else self.nn.predict

        #in case of MLPClassifier
        if self.type == 'classification':
            self.classes = ['prob_' + str(name) for name in self.nn.classes_]
        else:
            self.classes = ['prob_Y']

        self.data_and_probabilities = self.combine_data_and_results()

    def combine_data_and_results(self, data=None):
        if data is None:
            data = self.data

        #drop old columns: all starting with 'prob_' and 'prediction'
        data = data.drop(columns=[col for col in data.columns if col.startswith('prob_') or col == 'prediction'])

        # get new probabilities
        all_predictions = self.predict(data[self.columns])
        all_predictions = pd.DataFrame(all_predictions, columns=self.classes)
        all_predictions['prediction'] = all_predictions.idxmax(axis=1)
        # merge X_test, shap, predictions
        all_data = pd.concat([data, all_predictions], axis=1)
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

def load_weather_truth():
    # load weather files
    file_truth = open('weather_data/weather_testtruth.csv', 'rb')
    truth = pd.read_csv(file_truth)

    return truth


def load_data(file_data):
    return pd.read_csv(io.BytesIO(file_data))


def load_nn(file_nn):
    return pickle.load(io.BytesIO(file_nn))


def get_means(data):
    return data.mean().to_frame().T
