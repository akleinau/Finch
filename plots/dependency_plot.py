from bokeh.plotting import figure
from bokeh.models import Band, ColumnDataSource, HoverTool, Legend, LegendItem
import numpy as np
from scipy.stats import gaussian_kde
import pandas as pd
import bokeh.colors


# for the contours
def kde(x, y, N):
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    X, Y = np.mgrid[xmin:xmax:N * 1j, ymin:ymax:N * 1j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([x, y])
    kernel = gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)

    return X, Y, Z

def dependency_scatterplot(data, col, all_selected_cols, prob, index, chart_type):
    item = data.iloc[index]
    sorted_data = data.sort_values(by=col)

    x_range = (sorted_data[col].min(), sorted_data[col].max())

    chart3 = figure(title="example", y_axis_label=prob, tools="tap", y_range=(0, 1), x_range=x_range, width=800)
    chart3.grid.level = "overlay"
    chart3.grid.grid_line_color = "black"
    chart3.grid.grid_line_alpha = 0.05
    chart3.add_layout(Legend(), 'right')

    # create bands and contours for each group
    legend_items = []
    colors = sorted_data["scatter_group"].unique()
    for i, color in enumerate(colors):
        filtered_data = sorted_data[sorted_data["scatter_group"] == color].sort_values(by=col)
        if len(filtered_data) > 0:
            cluster_label = filtered_data["scatter_label"].iloc[0]
            window = max(len(filtered_data) // 10, 10)
            rolling = filtered_data[prob].rolling(window=window, center=True, min_periods=1).agg(
                {'lower': lambda ev: ev.quantile(.25, interpolation='lower'),
                 'upper': lambda ev: ev.quantile(.75, interpolation='higher'),
                 'median': 'median',
                 'count': 'count'})
            rolling = rolling.rolling(window=window, center=True, min_periods=1).mean()
            combined = pd.concat([filtered_data, rolling], axis=1)
            combined = ColumnDataSource(combined.reset_index())

            if "contour" in chart_type:
                # only use subset of data for performance reasons
                if len(filtered_data) > 1000:
                    data_subset = filtered_data.sample(n=1000)
                else:
                    data_subset = filtered_data

                x, y, z = kde(data_subset[col], data_subset[prob], 100)

                # use the color to create a palette
                rgb = color = tuple(int(color[1:][i:i+2], 16) for i in (0, 2, 4)) # convert hex to rgb
                # to bokeh
                cur_color = bokeh.colors.RGB(*rgb)
                palette = [cur_color]
                for i in range(0, 3):
                    palette.append(palette[i].lighten(0.2))

                palette = [c.to_hex() for c in palette]  # convert to hex
                palette = palette[::-1]  # invert the palette

                levels = np.linspace(np.min(z), np.max(z), 7)
                contour = chart3.contour(x, y, z, levels[1:], fill_color=palette, line_color=palette, fill_alpha=0.8)
                contour.fill_renderer.name = cluster_label

                #contour.fill_renderer
                contour_hover = HoverTool(renderers=[ contour.fill_renderer], tooltips=[('', '$name')])
                chart3.add_tools(contour_hover)

                # add legend items
                dummy_for_legend = chart3.line(x=[1, 1], y=[1, 1], line_width=15, color=color, name='dummy_for_legend')
                legend_items.append((cluster_label, [dummy_for_legend]))

            if "band" in chart_type:
                band = chart3.varea(x=col, y1='lower', y2='upper', source=combined,
                                    legend_label=cluster_label, fill_color=color,
                                    alpha=0.3, name=cluster_label)
                band_hover = HoverTool(renderers=[band], tooltips=[('', '$name')])
                chart3.add_tools(band_hover)

            if "line" in chart_type:
                line = chart3.line(col, 'median', source=combined, color=color, line_width=2,
                                   legend_label=cluster_label,
                                   name=cluster_label)
                line_hover = HoverTool(renderers=[line], tooltips=[('', '$name')])
                chart3.add_tools(line_hover)



    if "scatter" in chart_type:
        alpha = 0.3
        chart3.scatter(col, prob, color="scatter_group", source=sorted_data,
                       alpha=alpha, marker='circle', size=3, name="scatter_label", legend_group="scatter_label")

    # add the selected item
    item_scatter = chart3.scatter(item[col], item[prob], color='purple', size=7, name="selected item", legend_label="selected item")

    scatter_hover = HoverTool(renderers=[item_scatter], tooltips=[('', '$name')])
    chart3.add_tools(scatter_hover)

    # add legend
    chart3.legend.items.extend([LegendItem(label=x,renderers=y) for (x,y) in legend_items])
    chart3.legend.location = "right"
    return chart3
