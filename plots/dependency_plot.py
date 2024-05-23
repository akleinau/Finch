from bokeh.plotting import figure
from bokeh.models import Band, ColumnDataSource
import numpy as np
from scipy.stats import gaussian_kde
import pandas as pd
import bokeh.colors


def dependency_scatterplot(data, col, all_selected_cols, prob, index, chart_type):
    item = data.iloc[index]
    sorted_data = data.sort_values(by=col)

    if len(all_selected_cols) > 1:
        item_val = item[all_selected_cols[1]]
        sorted_data["scatter_group"] = sorted_data[all_selected_cols[1]].apply(
            lambda x: 'saddlebrown' if x >= item_val else 'midnightblue')
    else:
        sorted_data["scatter_group"] = 'forestgreen'

    x_range = (sorted_data[col].min(), sorted_data[col].max())

    chart3 = figure(title="example", y_axis_label=prob, tools='tap', y_range=(0, 1), x_range=x_range)
    chart3.grid.level = "overlay"
    chart3.grid.grid_line_color = "black"
    chart3.grid.grid_line_alpha = 0.05

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

    # create bands and contours
    colors = ['midnightblue', 'saddlebrown', 'forestgreen']
    for i, color in enumerate(colors):
        filtered_data = sorted_data[sorted_data["scatter_group"] == color].sort_values(by=col)
        if len(filtered_data) > 0:
            window = max(len(filtered_data) // 10, 10)
            rolling = filtered_data[prob].rolling(window=window, center=True, min_periods=1).agg(
                {'lower': lambda ev: ev.quantile(.25, interpolation='lower'),
                 'upper': lambda ev: ev.quantile(.75, interpolation='higher'),
                 'median': 'median',
                 'count': 'count'})
            rolling = rolling.rolling(window=window, center=True, min_periods=1).mean()
            combined = pd.concat([filtered_data, rolling], axis=1)
            combined = ColumnDataSource(combined.reset_index())

            if "line" in chart_type:
                chart3.line(col, 'median', source=combined, color=color, line_width=2)

            if "band" in chart_type:
                band = Band(base=col, lower='lower', upper='upper', source=combined,
                            fill_color=color)

                chart3.add_layout(band)

            if "contour" in chart_type:
                # only use subset of data for performance reasons
                if len(filtered_data) > 1000:
                    data_subset = filtered_data.sample(n=1000)
                else:
                    data_subset = filtered_data

                x, y, z = kde(data_subset[col], data_subset[prob], 100)

                # use the color to create a palette
                cur_color = bokeh.colors.named.NamedColor.find(color)
                palette = [cur_color]
                for i in range(0, 3):
                    palette.append(palette[i].lighten(0.2))

                palette = [c.to_hex() for c in palette]  # convert to hex
                palette = palette[::-1]  # invert the palette

                levels = np.linspace(np.min(z), np.max(z), 7)
                chart3.contour(x, y, z, levels[1:], fill_color=palette, line_color=palette, fill_alpha=0.8)

    if "scatter" in chart_type:
        alpha = 0.3
        chart3.scatter(sorted_data[col], sorted_data[prob], color=sorted_data["scatter_group"],
                       alpha=alpha, marker='dot', size=20)
    chart3.scatter(item[col], item[prob], color='purple', size=7)

    return chart3
