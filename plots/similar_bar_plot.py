import pandas as pd
from bokeh.models import HoverTool, FactorRange, Span
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from calculations.similarity import get_similar_items


def similar_bar_plot(data, item, all_selected_cols, predict_class, predict_label):
    groups = []
    #standard group
    groups.append({'group_name': 'standard', 'group_label': 'standard', 'probability': data[predict_class].mean(),
                   'color': 'grey', 'alpha': 1})

    #group of item
    item_group = data[data['scatter_label'] == item.scatter_label]
    groups.append({'group_name': 'cluster', 'group_label': item.scatter_label, 'probability': item_group[predict_class].mean(),
                   'color': item.scatter_group, 'alpha': 0.5})

    #group of similar items in same cluster
    similar_item_group = get_similar_items(item_group, item, all_selected_cols)
    #similar_item_group = similar_item_group[similar_item_group['scatter_label'] == item.scatter_label] #necessary if I group first, then filter
    groups.append({'group_name': 'similar', 'group_label': 'similar items with ' + item.scatter_label,
                   'probability': similar_item_group[predict_class].mean(), 'color': item.scatter_group, 'alpha': 1.0})



    cluster_data = data[[predict_class, "scatter_group", "scatter_label"]].groupby("scatter_group")
    clusters = pd.DataFrame()
    clusters['mean'] = cluster_data[predict_class].mean()
    clusters['mean_short'] = clusters['mean'].apply(lambda x: "{:.2f}".format(x) + " ")
    clusters['std'] = cluster_data[predict_class].std()
    clusters['count'] = cluster_data[predict_class].count()
    clusters['median'] = cluster_data[predict_class].median()
    clusters['scatter_label'] = cluster_data['scatter_label'].first()
    clusters.reset_index(level=0, inplace=True)
    clusters.sort_values(by='mean', inplace=True)

    groups = pd.DataFrame(groups)

    y_range = groups['group_label'].values

    title = "Clusters for " + ", ".join(all_selected_cols)

    plot = figure(title=title, y_range=y_range, x_range=[0,1], width=800)
    plot.hbar(
        y='group_label',
        right='probability',
        fill_color="color",
        fill_alpha="alpha",
        line_width=0,
        height=0.5,
        source=groups,
        nonselection_fill_alpha=0.7,
    )

    # add vertical line for item
    item_prob = item.data_prob_raw[predict_class]
    item_label = "selected item (" + "{:.2f}".format(item_prob) + ")"
    item_line = Span(location=item_prob, dimension='height', line_color='red', line_width=2, line_dash='dashed')
    plot.add_layout(item_line)


    plot.xaxis.axis_label = "Probability of " + predict_label

    return plot