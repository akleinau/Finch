from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from bokeh.models import ColumnDataSource


def set_col(data, item_source, col):
    if len(item_source.selected.indices) > 0:
        if len(item_source.selected.indices) > 1:
            item_source.selected.indices = item_source.selected.indices[1:2]
        select = data.iloc[item_source.selected.indices]
        select = select['feature'].values[0]
        col[0].value = select  # col[0], bc the widget had to be wrapped in a list to be changed


def shap_tornado_plot(data, col):
    item_source = ColumnDataSource(data=data)
    chart2 = figure(title="relevance", y_range=data['feature'], x_range=(-1, 1), tools='tap')
    chart2.hbar(
        y='feature',
        right='shap_value',
        fill_color=factor_cmap("positive", palette=["steelblue", "crimson"], factors=["pos", "neg"]),
        line_width=0,
        source=item_source,
        nonselection_fill_alpha=0.7,
        selection_hatch_pattern='horizontal_wave',
        selection_hatch_scale=7,
        selection_hatch_weight=1.5,
        selection_hatch_color='purple'
    )
    chart2.grid.grid_line_color = "black"
    chart2.grid.grid_line_alpha = 0.05

    chart2.on_event('tap', lambda event: set_col(data, item_source, col))
    return chart2
