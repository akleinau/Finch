import pandas as pd
from bokeh.models import HoverTool, FactorRange
from bokeh.plotting import figure
from bokeh.transform import factor_cmap


def cluster_bar_plot(data, item, index, all_selected_cols, predict_class):
    cluster_data = data[[predict_class, "scatter_group", "scatter_label"]].groupby("scatter_group")
    clusters = pd.DataFrame()
    clusters['mean'] = cluster_data[predict_class].mean()
    clusters['std'] = cluster_data[predict_class].std()
    clusters['count'] = cluster_data[predict_class].count()
    clusters['median'] = cluster_data[predict_class].median()
    clusters['scatter_label'] = cluster_data['scatter_label'].first()
    clusters.reset_index(level=0, inplace=True)
    clusters.sort_values(by='mean', inplace=True)

    y_range = clusters['scatter_label'].values

    title = "Clusters for " + ", ".join(all_selected_cols)

    plot = figure(title=title, y_range=y_range, x_range=[0,1], width=800)
    plot.hbar(
        y='scatter_label',
        right='mean',
        fill_color="scatter_group",
        line_width=0,
        height=0.5,
        source=clusters,
        nonselection_fill_alpha=0.7,
    )

    plot.xaxis.axis_label = "Probability"

    # add item
    item = data.iloc[index]

    item_scatter = plot.scatter(y=[item['scatter_label']], x=[item[predict_class]], color='purple', size=7, name="selected item",
                                  legend_label="selected item")

    scatter_hover = HoverTool(renderers=[item_scatter], tooltips=[('', '$name')])
    plot.add_tools(scatter_hover)



    return plot