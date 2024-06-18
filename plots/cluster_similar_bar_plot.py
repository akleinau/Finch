import numpy as np
import pandas as pd
from bokeh.models import HoverTool, FactorRange, Legend, BoxAnnotation
from bokeh.plotting import figure
from calculations.similarity import get_similar_items


def cluster_similar_bar_plot(data, item, all_selected_cols, predict_class, predict_label):
    relative_coloring = False

    cluster_data = data[[predict_class, "scatter_group", "scatter_label", "group"]].groupby("group")
    clusters = pd.DataFrame()
    clusters['mean'] = cluster_data[predict_class].mean()
    clusters['mean_short'] = clusters['mean'].apply(lambda x: "{:.2f}".format(x) + " ")
    clusters['scatter_label'] = cluster_data['scatter_label'].first()
    clusters['scatter_group'] = cluster_data['scatter_group'].first()
    clusters.sort_values(by='mean', inplace=True)
    clusters.reset_index(level=0, inplace=True)
    # I can't directly use the labels for positioning, but will instead use the index
    clusters['scatter_label_index'] = clusters.index + 0.2
    clusters['focus_color'] = clusters['group'].map(lambda x: 'purple' if x == item.group else 'grey')
    clusters['color'] = clusters['focus_color'] if relative_coloring else clusters['scatter_group']

    #extract mapping of scatter_label to scatter_label_index
    scatter_label_to_index = dict(zip(clusters['scatter_label'], clusters['scatter_label_index']))

    similar_item_group = get_similar_items(data, item, all_selected_cols)
    similar_cluster_data = similar_item_group[[predict_class, "scatter_group", "scatter_label", "group"]].groupby("group")
    similar_clusters = pd.DataFrame()
    similar_clusters['mean'] = similar_cluster_data[predict_class].mean()
    similar_clusters['mean_short'] = similar_clusters['mean'].apply(lambda x: "{:.2f}".format(x) + " ")
    similar_clusters['count'] = similar_cluster_data[predict_class].count()
    #similar_clusters = similar_clusters[similar_clusters['count'] > 10] #filters out small sample sizes
    similar_clusters['scatter_label'] = similar_cluster_data['scatter_label'].first()
    similar_clusters['scatter_group'] = similar_cluster_data['scatter_group'].first()
    similar_clusters.reset_index(level=0, inplace=True)
    # scatter_label_index should be the same as in clusters
    similar_clusters['scatter_label_index'] = similar_clusters['scatter_label'].map(scatter_label_to_index) - 0.2
    similar_clusters['label'] = "similar items"
    similar_clusters['focus_color'] = similar_clusters['group'].map(lambda x: 'purple' if x == item.group else 'grey')
    similar_clusters['color'] = similar_clusters['focus_color'] if relative_coloring else similar_clusters['scatter_group']

    if (len(all_selected_cols) != len(item.data_raw.columns)):
        title = "Clusters for " + ", ".join(all_selected_cols)
    else:
        title = "Clusters for all columns"

    x_range = [np.floor(min(clusters['mean'].min(), similar_clusters['mean'].min(), item.data_prob_raw[predict_class])),
               np.ceil(max(clusters['mean'].max(), similar_clusters['mean'].max(), item.data_prob_raw[predict_class]))]

    plot = figure(title=title, x_range=x_range, width=800)
    plot.add_layout(Legend(), "above")
    plot.legend.orientation = "horizontal"

    #color background left and right of item y
    item_prob = item.data_prob_raw[predict_class]
    if relative_coloring:
        plot.add_layout(BoxAnnotation(left=0, right=item_prob, fill_alpha=0.1, fill_color='crimson', level='underlay'))
        plot.add_layout(BoxAnnotation(left=item_prob, right=1, fill_alpha=0.1, fill_color='steelblue', level='underlay'))


    old_bars = plot.hbar(
        y='scatter_label_index',
        right='mean',
        fill_color="color",
        line_width=0,
        height=0.5,
        source=clusters,
        nonselection_fill_alpha=0.7,
        alpha=0.5,
        legend_label="overall",
        border_radius=5,
    )


    plot.xaxis.axis_label = predict_label

    # now replace the index with the scatter_label
    plot.yaxis.ticker = similar_clusters['scatter_label_index']
    plot.yaxis.major_label_overrides = dict(zip(similar_clusters['scatter_label_index'], similar_clusters['scatter_label']))

    # add item
    if item.type != 'global':

        plot.hbar(
            y='scatter_label_index',
            right='mean',
            fill_color="color",
            line_width=0,
            height=0.35,
            source=similar_clusters,
            nonselection_fill_alpha=1,
            alpha=1,
            legend_label="similar items",
            border_radius=5,
        )

        # format to two decimal places
        plot.text(x='mean', y='scatter_label_index', text="mean_short", text_align='right', text_baseline='middle',
                  text_font_size='10pt', text_color="white", source=similar_clusters)

        y = similar_clusters[similar_clusters['scatter_label'] == item.scatter_label]['scatter_label_index']
        prob_item_label = "selected item (" + "{:.2f}".format(item.data_prob_raw[predict_class]) + ")"
        item_scatter = plot.scatter(y=y, x=[item.data_prob_raw[predict_class]], color='purple', size=7, name=prob_item_label,
                                      legend_label="selected item")

        scatter_hover = HoverTool(renderers=[item_scatter], tooltips=[('', '$name')])
        plot.add_tools(scatter_hover)

        # create explaining labels on the bin with smallest y, which is the first one
        if (len(similar_clusters) > 0):
            # get the first element (smallest y
            min_prob = clusters['mean'].values[0]
            min_prob_index = clusters['scatter_label_index'].values[0]

            prob_cluster_values = similar_clusters[round(similar_clusters['scatter_label_index']) == round(min_prob_index - 0.2)]['mean'].values
            if len(prob_cluster_values) == 0:
                prob_cluster = 0
            else:
                prob_cluster = prob_cluster_values[0]
            #prob_cluster = similar_clusters[round(similar_clusters['scatter_label_index']) == round(min_prob_index - 0.2)]['mean'].values[0]
            plot.text(x=prob_cluster + 0.01, y=[min_prob_index - 0.1], text=["similar items"], text_align='left', text_baseline='middle',
                        text_font_size='11pt', text_color="black")
            plot.text(x=min_prob + 0.01, y=[min_prob_index + 0.15], text=["all items"], text_align='left', text_baseline='middle',
                        text_font_size='11pt', text_color="grey")



    return plot