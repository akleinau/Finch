import pickle


def load_weather_data():
    # load weather files
    file_nn = open('weather_nn.pkl', 'rb')
    nn = pickle.load(file_nn)
    file_means = open('weather_means.pkl', 'rb')
    means = pickle.load(file_means)
    file_testdata = open('weather_testdata.pkl', 'rb')
    testdata = pickle.load(file_testdata)
    file_nn.close()
    file_means.close()

    return nn, means, testdata
