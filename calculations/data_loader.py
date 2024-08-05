import io
import pickle

import pandas as pd
from panel.viewable import Viewer


class DataLoader(Viewer):
    def __init__(self, file=None, nn_file=None, truth_file=None):
        super().__init__()
        if file is None or nn_file is None:
            self.data = load_bike_data()
            self.columns = [col for col in self.data.columns]
            self.nn = load_bike_nn()
            truth = load_bike_truth()

        else:
            self.data = load_data(file)
            self.nn = load_nn(nn_file)
            self.columns = [col for col in self.data.columns]
            truth = None
            if truth_file is not None:
                truth = load_data(truth_file)

        self.type = 'classification' if hasattr(self.nn, 'classes_') else 'regression'

        # in case a ground truth is provided
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

        # in case of MLPClassifier
        if self.type == 'classification':
            self.classes = ['prob_' + str(name) for name in self.nn.classes_]
        else:
            self.classes = ['prob_Y']

        self.data_and_probabilities = self.combine_data_and_results()

    def combine_data_and_results(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """
        combines the data with the results of the neural network

        :param data: pd.DataFrame
        :return:
        """

        if data is None:
            data = self.data

        # drop old columns: all starting with 'prob_' and 'prediction'
        data = data.drop(columns=[col for col in data.columns if col.startswith('prob_') or col == 'prediction'])

        # get new probabilities
        all_predictions = self.predict(data[self.columns])
        all_predictions = pd.DataFrame(all_predictions, columns=self.classes)
        all_predictions['prediction'] = all_predictions.idxmax(axis=1)
        # merge X_test, shap, predictions
        all_data = pd.concat([data, all_predictions], axis=1)
        return all_data


def load_bike_data() -> pd.DataFrame:
    file_testdata = open('bike_data/bike_testdata.csv', 'rb')
    testdata = pd.read_csv(file_testdata)
    return testdata


def load_bike_nn() -> pickle:
    file_nn = open('bike_data/bikeREG_nn.pkl', 'rb')
    nn = pickle.load(file_nn)
    file_nn.close()
    return nn


def load_bike_truth() -> pd.DataFrame:
    file_truth = open('bike_data/bike_testtruth.csv', 'rb')
    truth = pd.read_csv(file_truth)
    return truth


def load_data(file_data) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_data))


def load_nn(file_nn) -> pickle:
    return pickle.load(io.BytesIO(file_nn))


def get_means(data: pd.DataFrame) -> pd.DataFrame:
    return data.mean().to_frame().T
