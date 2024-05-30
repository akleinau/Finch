import pickle
import pandas as pd
import io
import calculations.column_functions as column_functions


def load_weather_data():
    # load weather files
    file_nn = open('weather_data/weather_nn.pkl', 'rb')
    nn = pickle.load(file_nn)
    file_testdata = open('weather_data/weather_testdata.csv', 'rb')
    testdata = pd.read_csv(file_testdata)
    file_nn.close()

    return nn, testdata


def load_data(file_data=None, file_nn=None):
    if file_data is None or file_nn is None:
        return load_weather_data()[1]
    else:
        return pd.read_csv(io.BytesIO(file_data))


def load_nn(file_nn=None, file_data=None):
    if file_nn is None or file_data is None:
        return load_weather_data()[0]
    else:
        return pickle.load(io.BytesIO(file_nn))


def data_changed(event, widgets, file, nn_file):
    col = widgets[0]
    cur_feature = widgets[1]
    all_selected_cols = widgets[2]
    widget = widgets[3]
    data = widgets[4]
    nn = widgets[5]

    data.value = load_data(file, nn_file)
    nn.value = load_nn(file, nn_file)

    col.options = widget[5]
    col.value = col.options[0]
    cur_feature.options = all_selected_cols
    cur_feature.value = cur_feature.options[0]

    column_functions.init_groups(widget)
