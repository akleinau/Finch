from bokeh.plotting import figure
from bokeh.models import Band, ColumnDataSource, HoverTool, Legend, LegendItem, ColorBar
import numpy as np
import pandas as pd
import bokeh.colors
from bokeh.palettes import viridis

from bokeh.models import (BasicTickFormatter, ColumnDataSource,
                          CustomJSTickFormatter, FixedTicker,
                          LinearAxis, LinearColorMapper, MultiLine, Range1d)


def bokeh_parallel_plot(data, all_selected_cols, prob):
    curr_data = data[all_selected_cols]

    """From a dataframe create a parallel coordinate plot
    """
    npts = curr_data.shape[0]
    ndims = len(all_selected_cols)

    cmap = LinearColorMapper(high=0,
                             low=1,
                             palette=viridis(256))

    data_source = ColumnDataSource(dict(
        xs=np.arange(ndims)[None, :].repeat(npts, axis=0).tolist(),
        ys=np.array((curr_data - curr_data.min()) / (curr_data.max() - curr_data.min())).tolist(),
        color=data[prob].tolist()))

    p = figure(title="parallel_plot",
               x_range=(-1, ndims),
               y_range=(0, 1),
               width=800)
    p.add_layout(Legend(), 'right')

    # Create x axis ticks from columns contained in dataframe
    fixed_x_ticks = FixedTicker(
        ticks=np.arange(ndims), minor_ticks=[])
    formatter_x_ticks = CustomJSTickFormatter(
        code="return columns[index]", args={"columns": curr_data.columns})
    p.xaxis.ticker = fixed_x_ticks
    p.xaxis.formatter = formatter_x_ticks

    p.yaxis.visible = False
    p.y_range.start = 0
    p.y_range.end = 1
    p.y_range.bounds = (-0.1, 1.1)  # add a little padding around y axis
    p.xgrid.visible = False
    p.ygrid.visible = False

    # Create extra y axis for each dataframe column
    tickformatter = BasicTickFormatter(precision=1)
    for index, col in enumerate(all_selected_cols):
        start = curr_data[col].min()
        end = curr_data[col].max()
        bound_min = start + abs(end - start) * (p.y_range.bounds[0] - p.y_range.start)
        bound_max = end + abs(end - start) * (p.y_range.bounds[1] - p.y_range.end)
        p.extra_y_ranges.update(
            {col: Range1d(start=bound_min, end=bound_max, bounds=(bound_min, bound_max))})

        fixedticks = FixedTicker(
            ticks=np.linspace(start, end, 8), minor_ticks=[])

        p.add_layout(LinearAxis(fixed_location=index, y_range_name=col,
                                ticker=fixedticks, formatter=tickformatter), 'right')

        # create the data renderer ( MultiLine )

        line_style = dict(line_color={'field': 'color', 'transform': cmap}, line_width=1, line_alpha=0.3)

        parallel_renderer = p.multi_line(
            xs="xs", ys="ys", source=data_source, **line_style)

        # Specify selection style
        lines = MultiLine(**line_style)

        parallel_renderer.selection_glyph = lines
        parallel_renderer.nonselection_glyph = lines
    p.y_range.start = p.y_range.bounds[0]
    p.y_range.end = p.y_range.bounds[1]

    # add color legend
    color_mapper = LinearColorMapper(palette=viridis(256), low=0, high=1)
    color_bar = ColorBar(color_mapper=color_mapper, width=8, location=(0, 0))
    p.add_layout(color_bar, 'right')

    return p


def parallel_plot(data, col, all_selected_cols, prob, item, chart_type):
    p = bokeh_parallel_plot(data, all_selected_cols, prob)

    return p
