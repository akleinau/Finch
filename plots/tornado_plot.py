from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from bokeh.models import ColumnDataSource, HoverTool
from plots.render_plot import add_style


def set_col(data, item_source, col):
    if len(item_source.selected.indices) > 0:
        if len(item_source.selected.indices) > 1:
            item_source.selected.indices = item_source.selected.indices[1:2]
        select = data.iloc[item_source.selected.indices]
        select = select['feature'].values[0]
        col[0].value = select  # col[0], bc the widget had to be wrapped in a list to be changed


def shap_tornado_plot(data, col):
    shap = data.shap
    item_source = ColumnDataSource(data=shap)
    #get last item
    col[0].value = shap['feature'].values[-1]

    plot = figure(title="Feature Set Relevance", y_range=shap['feature_label_short'], x_range=(-1, 1), tools='tap')

    back_bars_left = plot.hbar(
        y='feature_label_short',
        right=-1,
        fill_color="lavender",
        line_width=0,
        source=item_source,
        nonselection_fill_alpha=0.0,
        selection_fill_alpha=1,
    )

    back_bars_right = plot.hbar(
        y='feature_label_short',
        right=1,
        fill_color="lavender",
        line_width=0,
        source=item_source,
        nonselection_fill_alpha=0.0,
        selection_fill_alpha=1,
    )

    bars = plot.hbar(
        y='feature_label_short',
        right='shap_value',
        fill_color=factor_cmap("positive", palette=["steelblue", "crimson"], factors=["pos", "neg"]),
        fill_alpha=1,
        nonselection_fill_alpha=1,
        line_width=0,
        source=item_source,
    )

    plot.xaxis.axis_label = "shap value"

    plot = add_style(plot)

    plot.on_event('tap', lambda event: set_col(shap, item_source, col))

    hover = HoverTool(renderers=[back_bars_left, back_bars_right], tooltips=[('', '@feature_label')])
    plot.add_tools(hover)

    return plot
