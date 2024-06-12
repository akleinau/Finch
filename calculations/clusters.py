import pandas
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import bokeh.palettes as palettes
import numpy as np
from sklearn.tree import _tree
from calculations.similarity import get_similar_items


class Clustering:
    def __init__(self, cluster_type, data_and_probabilities, all_selected_cols, cur_feature, prediction, item_index,
                 exclude_col=True, num_leafs=6):
        self.data = get_clustering(cluster_type, data_and_probabilities, all_selected_cols, cur_feature, prediction,
                                   item_index, exclude_col, num_leafs)


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
            # add class name at the end
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
        if rule == "":
            rule = "All"

        # get class name
        class_label = path[-1]
        rules[class_label] = rule

    return rules


def _make_readable_axiom(axiom):
    if (axiom['operator'] == '>' and axiom['value'] == '0.5'):
        return 'higher '
    elif (axiom['operator'] == '<=' and axiom['value'] == '-0.5'):
        return 'lower '
    elif (axiom['operator'] == '>' and axiom['value'] == '-0.5'):
        return 'similar/ higher '
    elif (axiom['operator'] == '<=' and axiom['value'] == '0.5'):
        return 'similar/ lower '
    else:
        return 'similar '


def make_readable(x):
    axioms = x.split(' and ')
    axioms = [axiom.split() for axiom in axioms]
    if len(axioms[0]) < 3:
        return x

    # the feature names may contain spaces, so we need to merge it back together
    axioms = [[' '.join(axiom[:-2]), *axiom[-2:]] for axiom in axioms]


    axioms = pd.DataFrame(axioms, columns=['feature', 'operator', 'value'])
    new_rules = []
    axioms['new_rule'] = axioms.apply(_make_readable_axiom, axis=1)

    # if both'similar/ higher' and 'similar/ lower' exist for a feature, merge them together into 'similar'
    rule_groups = axioms.groupby('feature')
    rule_list = []
    for name, group in rule_groups:
        if 'similar/ higher ' in group['new_rule'].values and 'similar/ lower ' in group['new_rule'].values:
            rule_list.append('similar ' + group['feature'].values[0])
        else:
            rule_list.extend(group['new_rule'].values + group['feature'].values)

    # now merge the rules together
    return ' and '.join(rule_list)


def shorten_rules(x):
    axioms = x.split(' and ')
    axioms = [axiom[1:-1] for axiom in axioms]
    axioms = [axiom.split() for axiom in axioms]
    if len(axioms[0]) < 3:
        return x

    # the feature names may contain spaces, so we need to merge it back together
    axioms = [[' '.join(axiom[:-2]), *axiom[-2:]] for axiom in axioms]

    grouped_axioms = pd.DataFrame(axioms, columns=['feature', 'operator', 'value'])
    #make value numeric
    grouped_axioms['value'] = pd.to_numeric(grouped_axioms['value'])

    # now group
    grouped_axioms = grouped_axioms.groupby(['feature', 'operator'])
    grouped_axioms = grouped_axioms['value'].aggregate(['max', 'min'])
    grouped_axioms = grouped_axioms.reset_index()
    grouped_axioms['value'] = grouped_axioms.apply(lambda x: x['max'] if '>' in x['operator'] else x['min'], axis=1)

    # now create the new label
    grouped_axioms = grouped_axioms['feature'] + ' ' + grouped_axioms['operator'] + ' ' + grouped_axioms['value'].map(lambda a : str(a))
    return grouped_axioms.str.cat(sep=' and ')


def get_tree_groups(data, all_selected_cols, cur_col, prediction, item, exclude_col=True, num_leafs=6, similarity=False):
    # remove the current column from the list of all selected columns
    if exclude_col:
        columns = [col for col in all_selected_cols if col != cur_col]
    else:
        columns = all_selected_cols

    if len(columns) > 0:
        tree = DecisionTreeRegressor(max_leaf_nodes=num_leafs, min_samples_leaf=0.05)
        if similarity:
            similar_data = get_similar_items(data, item, [])
            tree.fit(similar_data[columns], similar_data[prediction])
        else:
            tree.fit(data[columns], data[prediction])

        data["group"] = tree.apply(data[columns])
        item.group = tree.apply(pd.DataFrame([{col: item.data_series[col] for col in columns}]))[0]

        sorted_groups = data.groupby("group")[prediction].mean().sort_values().index
        group_to_index = {group: i for i, group in enumerate(sorted_groups)}

        # create a color for each group
        data["scatter_group"] = data["group"].apply(lambda x: palettes.Set2[8][group_to_index[x] % 8])
        item.scatter_group = palettes.Set2[8][group_to_index[item.group] % 8]

        # create human-readable labels for each group containing the path to the group
        rules = get_tree_rules(tree, columns)
        rules = {k: shorten_rules(v) for k, v in rules.items()}

        data["scatter_label"] = data["group"].apply(lambda x: rules[x])
        item.scatter_label = rules[item.group]
    else:
        data["scatter_group"] = '#228b22'
        data["scatter_label"] = 'All'

    return data


def get_relative(x, item_val, range):
    # lambda x: 'saddlebrown' if x >= item_val else 'midnightblue'
    if (x < item_val - range):
        return -1
    elif (x > item_val + range):
        return 1
    else:
        return 0


def get_relative_tree_groups(data, all_selected_cols, cur_col, prediction, item, exclude_col=True, num_leafs=6):
    # remove the current column from the list of all selected columns
    if exclude_col:
        columns = [col for col in all_selected_cols if col != cur_col]
    else:
        columns = all_selected_cols

    if (len(columns) > 0):
        # make new relative columns
        relative_data = pd.DataFrame()
        for col in columns:
            item_val = item.data_series[col]
            range = data[col].max() - data[col].min()

            relative_data[col] = data[col].apply(lambda x: get_relative(x, item_val, range / 20))

        tree = DecisionTreeRegressor(max_leaf_nodes=num_leafs, min_samples_leaf=0.05)
        tree.fit(relative_data, data[prediction])

        data["group"] = tree.apply(relative_data)
        item.group = tree.apply(pd.DataFrame([{col: 0} for col in columns]))[0]
        # sort by group mean probability
        sorted_groups = data.groupby("group")[prediction].mean().sort_values().index
        group_to_index = {group: i for i, group in enumerate(sorted_groups)}

        # create a color for each group
        data["scatter_group"] = data["group"].apply(lambda x: palettes.Set2[8][group_to_index[x] % 8])
        item.scatter_group = palettes.Set2[8][group_to_index[item.group] % 8]

        # create human-readable labels for each group containing the path to the group
        rules = get_tree_rules(tree, columns)
        rules = {k: shorten_rules(v) for k, v in rules.items()}
        rules = {k: make_readable(v) for k, v in rules.items()}

        data["scatter_label"] = data["group"].apply(lambda x: rules[x])
        item.scatter_label = rules[item.group]
    else:
        data["scatter_group"] = '#228b22'
        data["scatter_label"] = 'All'
        item.scatter_group = '#228b22'
        item.scatter_label = 'All'

    return data


def get_clustering(cluster_type, data, all_selected_cols, cur_col, prediction, item, exclude_col=True, num_leafs=6):
    if cluster_type == 'Relative Decision Tree' and item.type != 'global':
        return get_relative_tree_groups(data, all_selected_cols, cur_col, prediction, item, exclude_col, num_leafs)
    elif cluster_type == 'Similarity Decision Tree' and item.type != 'global':
        return get_tree_groups(data, all_selected_cols, cur_col, prediction, item, exclude_col, num_leafs, similarity=True)
    else:
        return get_tree_groups(data, all_selected_cols, cur_col, prediction, item, exclude_col, num_leafs)
