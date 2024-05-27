def pred_diff(x1, x2, prediction):
    y1 = x1[prediction].values[0]
    y2 = x2[prediction].values[0]
    diff = y1 - y2
    return diff


def test_setup(data, columns, prediction):
    output = ""
    x1 = data.iloc[[26]]
    x2 = data.iloc[[27]]

    output += "The difference between the two predictions is: " + str(pred_diff(x1, x2, prediction)) + "\n"

    return output


def l2_loss(data, prediction):
    # calculate mean prediction across group
    mean_prediction = data[prediction].mean()
    # calculate difference between each prediction and the mean
    diff = data[prediction] - mean_prediction
    # square the differences
    squared_diff = diff ** 2
    # return the sum of the squared differences
    return squared_diff.sum()


def get_relative_groups(data, col, index):
    item = data.iloc[index]
    labels = {'forestgreen': 'All'}

    if len(col) > 1:
        item_val = item[col[1]]
        data["scatter_group"] = data[col[1]].apply(
            lambda x: 'saddlebrown' if x >= item_val else 'midnightblue')
        labels['midnightblue'] = 'Higher ' + str(col[1])
        labels['saddlebrown'] = 'Lower ' + str(col[1])
    else:
        data["scatter_group"] = 'forestgreen'

    data["scatter_label"] = data["scatter_group"].apply(lambda x: labels[x])

    return data
