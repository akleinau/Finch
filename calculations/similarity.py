from sklearn.tree import DecisionTreeRegressor
from bokeh.palettes import Category20
import numpy as np
from sklearn.tree import _tree

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
    labels = {'#228b22': 'All'}

    if len(col) > 1:
        item_val = item[col[1]]
        data["scatter_group"] = data[col[1]].apply(
            lambda x: 'saddlebrown' if x >= item_val else 'midnightblue')
        labels['midnightblue'] = 'Higher ' + str(col[1])
        labels['saddlebrown'] = 'Lower ' + str(col[1])
    else:
        data["scatter_group"] = '#228b22'

    data["scatter_label"] = data["scatter_group"].apply(lambda x: labels[x])

    return data


def get_tree_rules(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []

    def recurse(node, path, paths):

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"({name} <= {np.round(threshold, 3)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"({name} > {np.round(threshold, 3)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            #add class name at the end
            path += [node]

            paths += [path]

    recurse(0, path, paths)

    rules = {}
    for path in paths:
        rule = ""

        for p in path[:-2]:
            if rule != "":
                rule += " and "
            rule += str(p)
        # get class name
        class_label = path[-1]
        rules[class_label] = rule

    return rules


def get_tree_groups(data, all_selected_cols, cur_col, prediction):
    #remove the current column from the list of all selected columns
    all_selected_cols = [col for col in all_selected_cols if col != cur_col]

    if (len(all_selected_cols) > 0):
        tree = DecisionTreeRegressor(max_leaf_nodes=2)
        tree.fit(data[all_selected_cols], data[prediction])

        data["group"] = tree.apply(data[all_selected_cols])

        # create a color for each group
        data["scatter_group"] = data["group"].apply(lambda x: Category20[20][x])

        #create human-readable labels for each group containing the path to the group
        rules = get_tree_rules(tree, all_selected_cols)

        data["scatter_label"] = data["group"].apply(lambda x: rules[x])
    else:
        data["scatter_group"] = '#228b22'
        data["scatter_label"] = 'All'

    return data
