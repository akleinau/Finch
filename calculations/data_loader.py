import pickle


def load_weather_data():
    # load weather files
    file_nn = open('weather_data/weather_nn.pkl', 'rb')
    nn = pickle.load(file_nn)
    file_testdata = open('weather_data/weather_testdata.pkl', 'rb')
    testdata = pickle.load(file_testdata)
    file_nn.close()

    return nn, testdata
