from bokeh.plotting import figure
from bokeh.models import (Band, ColumnDataSource, HoverTool, Legend, LegendItem, BoxAnnotation, Arrow, NormalHead,
                          LinearAxis, LinearColorMapper, ColorBar, Text, Label)
import numpy as np
from scipy.stats import gaussian_kde
import pandas as pd
import bokeh.colors
from calculations.similarity import get_similar_items


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


def dependency_scatterplot(data, col, all_selected_cols, item, chart_type):
    #colors
    grey = '#606060'
    purple = '#A336B0'
    light_grey = '#A0A0A0'
    light_purple = '#cc98e6'
    positive_color = '#AE0139'
    negative_color = '#3801AC'
    selected_color = "#19b57A"

    truth = "truth" in data.columns
    relative = True
    item_style = "grey_line" # "point", "arrow", "line", "grey_line"
    influence_marker = ["color_axis", "colored_background"] # "colored_lines", "colored_background", "color_axis", "selective_colored_background"
    add_clusters = False
    sorted_data = data.copy().sort_values(by=col)
    mean = data[item.predict_class].mean()
    truth_class = "truth_" + item.predict_class[5:]
    if relative:
        sorted_data[item.predict_class] = sorted_data[item.predict_class].apply(lambda x: x- mean)
        if truth:
            sorted_data[truth_class] = sorted_data[truth_class].apply(lambda x: x - mean)


    x_range = (sorted_data[col].min(), sorted_data[col].max())
    item_x = item.data_prob_raw[col]
    x_std = sorted_data[col].std()
    x_range_padded = [x_range[0], x_range[1]]

    y_range = [sorted_data[item.predict_class].min(), sorted_data[item.predict_class].max()]
    y_range_padded = [y_range[0] - 0.025 * (y_range[1] - y_range[0]), y_range[1] + 0.05 * (y_range[1] - y_range[0])]

    if (len(all_selected_cols) != len(item.data_reduced)):
        title = "Influence of " + ", ".join(all_selected_cols)
    else:
        title = "Influence of all features"

    chart3 = figure(title=title, y_axis_label="influence", tools="tap, xpan, xwheel_zoom", y_range=y_range_padded, x_range=x_range_padded,
                    width=800, toolbar_location=None, active_scroll="xwheel_zoom")
    chart3.grid.level = "overlay"
    chart3.grid.grid_line_color = "black"
    chart3.grid.grid_line_alpha = 0.05
    chart3.add_layout(Legend(), 'above')
    chart3.legend.orientation = "horizontal"
    if item.type != 'global':
        chart3.x_range.start = item_x - x_std
        chart3.x_range.end = item_x + x_std

    # add the "standard probability" line
    chart3.line(x=[x_range[0], x_range[1]], y=[0, 0], line_width=1.5, color='#A0A0A0', alpha=1)

    # create bands and contours for each group
    legend_items = []
    colors = []
    if truth:
        colors.append(light_grey) # lighter grey
        if item.type != 'global' and len(all_selected_cols) > 1:
            colors.append(light_purple) # lighter purple
    if add_clusters:
        colors.append([c for c in sorted_data["scatter_group"].unique()]) # add all colors for the different groups
    colors.append(grey) # add grey for the standard group
    if item.type != 'global' and len(all_selected_cols) > 1:
        colors.append(purple) # add purple for the similar ones
    include_cols = [c for c in all_selected_cols if c != col]
    for i, color in enumerate(colors):
        # choose right data
        if (color == grey) or (color == light_grey):
            filtered_data = sorted_data
        elif (color == purple) or (color == light_purple):
            filtered_data = get_similar_items(sorted_data, item, include_cols)
        else:
            filtered_data = sorted_data[sorted_data["scatter_group"] == color].sort_values(by=col)

        # choose right column
        if (color == light_grey):
            y_col = truth_class
        elif (color == light_purple):
            y_col = truth_class
        else:
            y_col = item.predict_class

        # choose right line
        if (color == light_grey) or (color == light_purple):
            line_type = "dotted"
            alpha = 1
        else:
            line_type = "solid"
            alpha = 1

        if len(filtered_data) > 0:
            # choose right label
            if color == grey:
                cluster_label = 'Prediction'
            elif color == purple:
                cluster_label = 'Neighborhood prediction'
            elif color == light_grey:
                cluster_label = 'Ground truth'
            elif color == light_purple:
                cluster_label = 'Neighborhood ground truth'
            else:
                cluster_label = filtered_data["scatter_label"].iloc[0]


            window = max(len(filtered_data) // 10, 20)
            rolling = filtered_data[y_col].rolling(window=window, center=True, min_periods=1).agg(
                {'lower': lambda ev: ev.quantile(.25, interpolation='lower'),
                 'upper': lambda ev: ev.quantile(.75, interpolation='higher'),
                 'mean': 'mean',
                 'count': 'count'})
            rolling = rolling.rolling(window=window, center=True, min_periods=1).mean()
            combined = pd.concat([filtered_data, rolling], axis=1)
            #combined = ColumnDataSource(combined.reset_index())

            # add legend items
            dummy_for_legend = chart3.line(x=[1, 1], y=[1, 1], line_width=15, color=color, name='dummy_for_legend')
            legend_items.append((cluster_label, [dummy_for_legend]))

            if "contour" in chart_type and color == purple:
                # only use subset of data for performance reasons
                if len(filtered_data) > 1000:
                    data_subset = filtered_data.sample(n=1000)
                else:
                    data_subset = filtered_data

                x, y, z = kde(data_subset[col], data_subset[y_col], 100)

                # use the color to create a palette
                rgb = color = tuple(int(color[1:][i:i + 2], 16) for i in (0, 2, 4))  # convert hex to rgb
                # to bokeh
                cur_color = bokeh.colors.RGB(*rgb)
                palette = [cur_color]
                for i in range(0, 3):
                    new_color = palette[i].copy()
                    new_color.a -= 0.3
                    palette.append(new_color)

                palette = [c.to_hex() for c in palette]  # convert to hex
                palette = palette[::-1]  # invert the palette

                levels = np.linspace(np.min(z), np.max(z), 7)
                contour = chart3.contour(x, y, z, levels[1:], fill_color=palette, line_color=palette, fill_alpha=0.8)
                contour.fill_renderer.name = cluster_label

                # contour.fill_renderer
                contour_hover = HoverTool(renderers=[contour.fill_renderer], tooltips=[('', '$name')])
                chart3.add_tools(contour_hover)

            if "band" in chart_type and color == purple:
                band = chart3.varea(x=col, y1='lower', y2='upper', source=combined,
                                    # legend_label=cluster_label,
                                    fill_color=color,
                                    alpha=0.3, name=cluster_label)
                band_hover = HoverTool(renderers=[band], tooltips=[('', '$name')])
                chart3.add_tools(band_hover)

            if "line" in chart_type:
                line_width = 3.5 if color == purple or color == grey else 2
                if (color == purple or color == light_purple) and "colored_lines" in influence_marker:
                    # add a line that is red over 0 and blue below 0
                    # Segment or MultiLine might both be an easier variant for colored lines
                    combined_over_0 = combined[combined['mean'] >= 0]
                    combined_below_0 = combined[combined['mean'] <= 0]
                    line_over_0 = chart3.line(col, 'mean', source=combined_over_0, color=positive_color, line_width=line_width,
                                             # legend_label=cluster_label,
                                             name=cluster_label, line_dash=line_type, alpha=alpha)
                    line_below_0 = chart3.line(col, 'mean', source=combined_below_0, color=negative_color, line_width=line_width,
                                              # legend_label=cluster_label,
                                              name=cluster_label, line_dash=line_type, alpha=alpha)
                    line_hover = HoverTool(renderers=[line_over_0, line_below_0], tooltips=[('', '$name')])
                    chart3.add_tools(line_hover)

                else:
                    line = chart3.line(col, 'mean', source=combined, color=color, line_width=line_width,
                                       # legend_label=cluster_label,
                                       name=cluster_label, line_dash=line_type, alpha=alpha)
                    line_hover = HoverTool(renderers=[line], tooltips=[('', '$name')])
                    chart3.add_tools(line_hover)

            if "scatter" in chart_type and color == purple:
                alpha = 0.3
                chart3.scatter(col, y_col, color=color, source=filtered_data,
                               alpha=alpha, marker='circle', size=3, name="scatter_label",
                               # legend_group="scatter_label"
                               )

    # add the selected item
    if item.type != 'global':
        line_width = 4
        if (item_style == "absolute_point"):
            item_scatter = chart3.scatter(item.data_prob_raw[col], item.data_prob_raw[item.predict_class], color='purple', size=7, name="selected item",
                                          legend_label="Item")

            scatter_hover = HoverTool(renderers=[item_scatter], tooltips=[('', '$name')])
            chart3.add_tools(scatter_hover)

        elif (item_style == "point"):
            # add the point when only selected cols are used
            if item.prob_only_selected_cols is not None:
                chart3.scatter(x=item.data_prob_raw[col], y=item.prob_only_selected_cols - mean, color='purple',
                               legend_label='influence on item')

        elif (item_style == "arrow"):
            if item.prob_only_selected_cols is not None:
                # add the arrow when only selected cols are used
                color = "mediumblue" if item.prob_only_selected_cols - mean < 0 else "darkred"
                nh = NormalHead(fill_color=color, line_color=color, size=7)
                arrow = Arrow(end=nh, x_start=item_x, y_start=0, x_end=item_x, y_end=item.prob_only_selected_cols - mean,
                            line_color=color, line_width=2)
                chart3.add_layout(arrow)

        elif (item_style == "line"):

            line_blue = chart3.line(x=[item.data_prob_raw[col], item.data_prob_raw[col]], y=[0, y_range[1]], line_width=line_width,
                        color=positive_color, alpha=0.5, legend_label="Item", name=str(item.data_prob_raw[col]), line_cap='round')

            line_red = chart3.line(x=[item.data_prob_raw[col], item.data_prob_raw[col]], y=[y_range[0], 0], line_width=line_width,
                        color=negative_color, alpha=0.5, legend_label="Item", name=str(item.data_prob_raw[col]), line_cap='round')
            itemline_hover = HoverTool(renderers=[line_red, line_blue], tooltips=[(col + " of item", '$name')])
            chart3.add_tools(itemline_hover)

        elif (item_style == "grey_line"):
            chart3.line(x=[item.data_prob_raw[col], item.data_prob_raw[col]], y=[y_range[0], y_range[1]], line_width=line_width, color=selected_color,
                                    legend_label="Item", line_cap='round')

        # add the label
        chart3.add_layout(Label(x=item_x, y=463, y_units="screen", text=col + " = " + str(item_x), text_align='center',
                    text_baseline='bottom', text_font_size='11pt', text_color=selected_color))

    if "color_axis" in influence_marker:
        angle = 0 # np.pi / 2
        chart3.add_layout(
            Label(x=25, x_units="screen", y=0.5 * y_range_padded[1], text="+", text_align='center',
                  text_baseline='middle', text_font_size='30pt', text_color=positive_color, angle=angle))
        chart3.add_layout(
            Label(x=25, x_units="screen", y=0.5 * y_range_padded[0], text="-", text_align='center',
                  text_baseline='middle', text_font_size='30pt', text_color=negative_color, angle=angle))
        chart3.add_layout(
            BoxAnnotation(left=0, left_units="screen", right=10, right_units="screen", top=0, bottom=y_range_padded[0],
                          fill_color=negative_color, fill_alpha=1))
        chart3.add_layout(
            BoxAnnotation(left=0, left_units="screen", right=10, right_units="screen", top=y_range_padded[1], bottom=0,
                          fill_color=positive_color, fill_alpha=1))
    else:
        chart3.add_layout(Label(x=20, x_units="screen", y=1.1*y_range_padded[1], text="positive", text_align='left', text_baseline='top',
                    text_font_size='11pt', text_color="mediumblue"))
        chart3.add_layout(Label(x=20, x_units="screen", y=1.1*y_range_padded[0], text="negative", text_align='left', text_baseline='bottom',
                    text_font_size='11pt', text_color="darkred"))






    # add legend
    chart3.legend.items.extend([LegendItem(label=x, renderers=y) for (x, y) in legend_items])
    chart3.legend.location = "top"

    if "colored_background" in influence_marker:
        # color the background, blue below 0, red above 0
        chart3.add_layout(BoxAnnotation(bottom=y_range_padded[0], top=0, fill_color='#E6EDFF', level='underlay'))
        chart3.add_layout(BoxAnnotation(bottom=0, top=y_range_padded[1], fill_color='#FFE6FF', level='underlay'))

    if "selective_colored_background" in influence_marker:
        # color the background, blue below 0, red above 0
        if (item.prob_only_selected_cols - mean) < 0:
            chart3.add_layout(BoxAnnotation(bottom=y_range[0], top=0, fill_color='#AAAAFF'))
        else:
            chart3.add_layout(BoxAnnotation(bottom=0, top=y_range[1], fill_color='#FFAAAA'))

    return chart3
