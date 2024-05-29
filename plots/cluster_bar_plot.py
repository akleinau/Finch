import pandas as pd
from bokeh.models import HoverTool, FactorRange
from bokeh.plotting import figure
from bokeh.transform import factor_cmap


def cluster_bar_plot(data, col, all_selected_cols, prob, index, chart_type, prob_wo_selected_cols=None):
    cluster_data = data[[prob, "scatter_group", "scatter_label"]].groupby("scatter_group")
    clusters = pd.DataFrame()
    clusters['mean'] = cluster_data[prob].mean()
    clusters['std'] = cluster_data[prob].std()
    clusters['count'] = cluster_data[prob].count()
    clusters['median'] = cluster_data[prob].median()
    clusters['scatter_label'] = cluster_data['scatter_label'].first()
    clusters.reset_index(level=0, inplace=True)
    clusters.sort_values(by='mean', inplace=True)

    y_range = clusters['scatter_label'].values

    chart2 = figure(title="cluster means", y_range=y_range, x_range=[0,1], width=800)
    chart2.hbar(
        y='scatter_label',
        right='mean',
        fill_color="scatter_group",
        line_width=0,
        height=0.5,
        source=clusters,
        nonselection_fill_alpha=0.7,
    )

    # add item
    item = data.iloc[index]

    item_scatter = chart2.scatter(y=[item['scatter_label']], x=[item[prob]], color='purple', size=7, name="selected item",
                                  legend_label="selected item")

    scatter_hover = HoverTool(renderers=[item_scatter], tooltips=[('', '$name')])
    chart2.add_tools(scatter_hover)

    chart2.grid.grid_line_color = "black"
    chart2.grid.grid_line_alpha = 0.05

    return chart2