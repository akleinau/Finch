import io
import pickle

import pandas as pd
from panel.viewable import Viewer


class DataLoader(Viewer):
    def __init__(self, file=None, nn_file=None, truth_file=None):
        super().__init__()
        if file is None or nn_file is None:
            self.data = load_bike_data()
            self.nn = load_bike_nn()
            truth = load_bike_truth()

        else:
            error = False
            try:
                self.data = load_data(file)
            except:
                print("Could not load data. Reverting to bike dataset.")
                error = True

            try:
                self.nn = load_nn(nn_file)
            except:
                print("Could not load neural network. Reverting to bike dataset.")
                error = True

            truth = None
            if truth_file is not None:
                try:
                    truth = load_data(truth_file)
                except:
                    print("Could not load truth data.")

            if error:
                self.data = load_bike_data()
                self.nn = load_bike_nn()
                truth = load_bike_truth()

        nn_columns = [name for name in self.nn.feature_names_in_]

        # only keep columns that are in the model
        for column in self.data.columns:
            if column not in nn_columns:
                self.data.drop(column, axis=1, inplace=True)

        # make sure all columns exist
        for column in nn_columns:
            if column not in self.data.columns:
                print("Column", column, "not found in data. Adding column with zeros.")
                self.data[column] = 0

        self.columns = nn_columns

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

        self.column_details = self.get_column_details()


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

    def get_column_details(self):

        correlation_matrix = get_correlation_matrix(self.data, self.columns)

        details = {}
        for col in self.columns:
            details[col] = {
                'mean': self.means[col],
                'std': self.data[col].std(),
                'min': self.data[col].min(),
                'max': self.data[col].max(),
                'range': self.data[col].max() - self.data[col].min(),
                'type': get_column_type(self.data[col]),
                'bin_size': get_bin_size(self.data[col]),
                'similarity_boundary': 0.05,  # default similarity boundary
                'correlated_features': get_highly_correlated_columns(correlation_matrix, col),
            }
        return details


def get_correlation_matrix(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    # get pearson correlation matrix for the specified columns
    return data[columns].corr(method='pearson', numeric_only=True).abs()


def get_highly_correlated_columns(correlation_matrix: pd.DataFrame, column: str, threshold: float = 0.8) -> list:
    return [other_col for other_col in correlation_matrix.index if other_col != column and correlation_matrix.loc[column, other_col] > threshold]

def get_column_type(column: pd.Series) -> str:
    # determine if column is categorical or continuous
    unique = column.nunique()
    if unique < 24:
        return 'categorical'
    else:
        return 'continuous'

def get_bin_size(column: pd.Series) -> int:
    # calculate bin size
    return (column.max() - column.min()) / 50 if column.nunique() > 24 else 1



def load_bike_data() -> pd.DataFrame:
    file_testdata = open('bike_data/bike_traindata.csv', 'rb')
    testdata = pd.read_csv(file_testdata)
    return testdata


def load_bike_nn() -> pickle:
    file_nn = open('bike_data/bikeREG_nn.pkl', 'rb')
    nn = pickle.load(file_nn)
    file_nn.close()
    return nn


def load_bike_truth() -> pd.DataFrame:
    file_truth = open('bike_data/bike_traintruth.csv', 'rb')
    truth = pd.read_csv(file_truth)
    return truth


def load_data(file_data) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_data))


def load_nn(file_nn) -> pickle:
    return pickle.load(io.BytesIO(file_nn))


def get_means(data: pd.DataFrame) -> pd.DataFrame:
    return data.mean().to_frame().T
