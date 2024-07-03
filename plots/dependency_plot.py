import itertools

from bokeh.plotting import figure
from bokeh.models import (Band, ColumnDataSource, HoverTool, Legend, LegendItem, BoxAnnotation, Arrow, NormalHead,
                          LinearAxis, LinearColorMapper, ColorBar, Text, Label)
import numpy as np
from scipy.stats import gaussian_kde
import pandas as pd
import bokeh.colors
from calculations.similarity import get_similar_items, get_pdp_items


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


def dependency_scatterplot(data, col, all_selected_cols, item, chart_type, data_loader):
    #colors
    color_map = {'grey': '#808080', 'purple': '#A336B0', 'light_grey': '#A0A0A0', 'light_purple': '#cc98e6',
              'positive_color': '#AE0139', 'negative_color': '#3801AC', 'selected_color': "#19b57A", 'only_interaction': '#E2B1E7'}

    truth = "truth" in data.columns
    only_interaction = True
    relative = True
    item_style = "grey_line" # "point", "arrow", "line", "grey_line"
    influence_marker = ["color_axis", "colored_background"] # "colored_lines", "colored_background", "color_axis", "selective_colored_background"
    add_clusters = False
    truth_class = "truth_" + item.predict_class[5:]
    sorted_data = data.copy().sort_values(by=col)
    mean = data[item.predict_class].mean()
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

    chart3 = figure(title="", y_axis_label="influence", tools="tap, xpan, xwheel_zoom", y_range=y_range_padded, x_range=x_range_padded,
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
    colors = get_colors(add_clusters, all_selected_cols, item, sorted_data, truth, color_map, only_interaction)
    include_cols = [c for c in all_selected_cols if c != col]
    for i, color in enumerate(colors):
        # choose right data
        y_col = get_group_col(color, item, truth_class, color_map)
        group_data = get_group_data(color, include_cols, item, sorted_data, color_map, y_col, col, data_loader)
        alpha, line_type = get_group_style(color, color_map)

        if len(group_data) > 0:
            # choose right label
            cluster_label = get_group_label(color, group_data, color_map)

            # add legend items
            dummy_for_legend = chart3.line(x=[1, 1], y=[1, 1], line_width=15, color=color, name='dummy_for_legend')
            legend_items.append((cluster_label, [dummy_for_legend]))

            if "contour" in chart_type and color == color_map['purple']:
                color = create_contour(chart3, cluster_label, col, color, group_data, y_col)

            if "band" in chart_type and color == color_map['purple']:
                create_band(chart3, cluster_label, col, color, group_data)

            if "line" in chart_type:
                create_line(alpha, chart3, cluster_label, col, color, group_data, influence_marker, line_type,
                            color_map)

            if "scatter" in chart_type and color == color_map['purple']:
                data = get_filtered_data(color, include_cols, item, sorted_data, color_map)
                create_scatter(chart3, col, color, data, y_col)

    # add the selected item
    if item.type != 'global':
        add_item(chart3, col, item, item_style, item_x, mean, y_range, color_map)

    add_axis(chart3, influence_marker, y_range_padded, color_map)

    # add legend
    chart3.legend.items.extend([LegendItem(label=x, renderers=y) for (x, y) in legend_items])
    chart3.legend.location = "top"

    add_background(chart3, influence_marker, item, mean, y_range, y_range_padded)

    return chart3


def get_rolling(data, y_col, col):
    #first get mean per value of the col
    mean_data = data.groupby(col).agg({y_col: 'mean'})

    # then smooth the line
    window = max(1, min(int(len(mean_data)/15), 30))
    rolling = mean_data[y_col].rolling(window=window, center=False, min_periods=1).agg(
        {'lower': lambda ev: ev.quantile(.25, interpolation='lower'),
         'upper': lambda ev: ev.quantile(.75, interpolation='higher'),
         'mean': 'mean'})

    rolling = rolling.rolling(window=window, center=False, min_periods=1).mean()

    mean_data = mean_data.drop(columns=[y_col])

    combined = pd.concat([mean_data, rolling], axis=1)
    #print(combined.head())
    # combined = ColumnDataSource(combined.reset_index())
    return combined

def get_filtered_data(color, include_cols, item, sorted_data, color_map):
    if (color == color_map['grey']) or (color == color_map['light_grey']):
        filtered_data = sorted_data
    elif (color == color_map['purple']) or (color == color_map['light_purple']):
        filtered_data = get_similar_items(sorted_data, item, include_cols)
    else:
        filtered_data = sorted_data[sorted_data["scatter_group"] == color]
    return filtered_data

def get_group_data(color, include_cols, item, sorted_data, color_map, y_col, col, data_loader):
    if color == color_map['only_interaction']:
        #get current prediction
        filtered_data = get_similar_items(sorted_data, item, include_cols[:-1])
        #filtered_data = data_loader.combine_data_and_results(filtered_data)
        # PDP: TODO now I'd have to get the new predictions and substract the mean
        combined = get_rolling(filtered_data, y_col, col)

        # get previous prediction
        #filtered_data = get_similar_items(sorted_data, item, include_cols[:-1])
        # PDP: TODO now I'd have to get the new predictions and substract the mean
        #filtered_data = data_loader.combine_data_and_results(filtered_data)

        #sub_combined = get_rolling(filtered_data, y_col, col)
        # rename mean to sub_mean
        #sub_combined = sub_combined.rename(columns={'mean': 'sub_mean', 'upper': 'sub_upper', 'lower': 'sub_lower'})
        # inner join combined and sub_combined on col
        #combined = pd.merge(combined, sub_combined, on=col, how='inner')
        # substract the mean of sub_combined from combined
        #combined['mean'] = combined['mean'] - combined['sub_mean']
        #combined['upper'] = combined['upper'] - combined['sub_upper']
        #combined['lower'] = combined['lower'] - combined['sub_lower']
        # delete sub_mean again
        #combined = combined.drop(columns=['sub_mean', 'sub_upper', 'sub_lower'])
        #print(combined.head())
    else:
        filtered_data = get_filtered_data(color, include_cols, item, sorted_data, color_map)
        # TODO now I'd have to get the new predictions and substract the mean
        #filtered_data = data_loader.combine_data_and_results(filtered_data)
        combined = get_rolling(filtered_data, y_col, col)

    return combined


def add_background(chart3, influence_marker, item, mean, y_range, y_range_padded):
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


def add_axis(chart3, influence_marker, y_range_padded, colors):
    if "color_axis" in influence_marker:
        angle = 0  # np.pi / 2
        chart3.add_layout(
            Label(x=25, x_units="screen", y=0.5 * y_range_padded[1], text="+", text_align='center',
                  text_baseline='middle', text_font_size='30pt', text_color=colors['positive_color'], angle=angle))
        chart3.add_layout(
            Label(x=25, x_units="screen", y=0.5 * y_range_padded[0], text="-", text_align='center',
                  text_baseline='middle', text_font_size='30pt', text_color=colors['negative_color'], angle=angle))
        chart3.add_layout(
            BoxAnnotation(left=0, left_units="screen", right=10, right_units="screen", top=0, bottom=y_range_padded[0],
                          fill_color=colors['negative_color'], fill_alpha=1))
        chart3.add_layout(
            BoxAnnotation(left=0, left_units="screen", right=10, right_units="screen", top=y_range_padded[1], bottom=0,
                          fill_color=colors['positive_color'], fill_alpha=1))
    else:
        chart3.add_layout(Label(x=20, x_units="screen", y=1.1 * y_range_padded[1], text="positive", text_align='left',
                                text_baseline='top',
                                text_font_size='11pt', text_color="mediumblue"))
        chart3.add_layout(Label(x=20, x_units="screen", y=1.1 * y_range_padded[0], text="negative", text_align='left',
                                text_baseline='bottom',
                                text_font_size='11pt', text_color="darkred"))


def add_item(chart3, col, item, item_style, item_x, mean, y_range, colors):
    line_width = 4
    if (item_style == "absolute_point"):
        item_scatter = chart3.scatter(item.data_prob_raw[col], item.data_prob_raw[item.predict_class], color='purple',
                                      size=7, name="selected item",
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

        line_blue = chart3.line(x=[item.data_prob_raw[col], item.data_prob_raw[col]], y=[0, y_range[1]],
                                line_width=line_width,
                                color=colors['positive_color'], alpha=0.5, legend_label="Item", name=str(item.data_prob_raw[col]),
                                line_cap='round')

        line_red = chart3.line(x=[item.data_prob_raw[col], item.data_prob_raw[col]], y=[y_range[0], 0],
                               line_width=line_width,
                               color=colors['negative_color'], alpha=0.5, legend_label="Item", name=str(item.data_prob_raw[col]),
                               line_cap='round')
        itemline_hover = HoverTool(renderers=[line_red, line_blue], tooltips=[(col + " of item", '$name')])
        chart3.add_tools(itemline_hover)

    elif (item_style == "grey_line"):
        chart3.line(x=[item.data_prob_raw[col], item.data_prob_raw[col]], y=[y_range[0], y_range[1]],
                    line_width=line_width, color=colors['selected_color'],
                    legend_label="Item", line_cap='round')
    # add the label
    chart3.add_layout(Label(x=item_x, y=495, y_units="screen", text=col + " = " + str(item_x), text_align='center',
                            text_baseline='bottom', text_font_size='11pt', text_color=colors['selected_color']))


def create_scatter(chart3, col, color, filtered_data, y_col):
    alpha = 0.3
    chart3.scatter(col, y_col, color=color, source=filtered_data,
                   alpha=alpha, marker='circle', size=3, name="scatter_label",
                   # legend_group="scatter_label"
                   )


def create_line(alpha, chart3, cluster_label, col, color, combined, influence_marker, line_type, colors):
    line_width = 3.5 if color == colors['purple'] else 3 if color == colors['grey'] else 2
    if (color == colors['purple'] or color == colors['light_purple']) and "colored_lines" in influence_marker:
        # add a line that is red over 0 and blue below 0
        # Segment or MultiLine might both be an easier variant for colored lines
        combined_over_0 = combined[combined['mean'] >= 0]
        combined_below_0 = combined[combined['mean'] <= 0]
        line_over_0 = chart3.line(col, 'mean', source=combined_over_0, color=colors['positive_color'], line_width=line_width,
                                  # legend_label=cluster_label,
                                  name=cluster_label, line_dash=line_type, alpha=alpha)
        line_below_0 = chart3.line(col, 'mean', source=combined_below_0, color=colors['negative_color'], line_width=line_width,
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


def create_band(chart3, cluster_label, col, color, combined):
    band = chart3.varea(x=col, y1='lower', y2='upper', source=combined,
                        # legend_label=cluster_label,
                        fill_color=color,
                        alpha=0.3, name=cluster_label)
    band_hover = HoverTool(renderers=[band], tooltips=[('', '$name')])
    chart3.add_tools(band_hover)


def create_contour(chart3, cluster_label, col, color, filtered_data, y_col):
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
    return color


def get_colors(add_clusters, all_selected_cols, item, sorted_data, truth, color_map, only_interaction):
    colors = []
    ## light grey for truth
    if truth and len(all_selected_cols) == 1:
        colors.append(color_map['light_grey'])

    ## light purple for neighbor truth
    if truth and item.type != 'global' and len(all_selected_cols) > 1:
        colors.append(color_map['light_purple'])

    # clusters
    if add_clusters:
        colors.append([c for c in sorted_data["scatter_group"].unique()])

    # grey for the standard group
    colors.append(color_map['grey'])

    # only_interaction
    if only_interaction and len(all_selected_cols) > 2:
        colors.append(color_map['only_interaction'])

    # purple for neighbors
    if item.type != 'global' and len(all_selected_cols) > 1:
        colors.append(color_map['purple'])

    return colors


def get_group_style(color, color_map):
    if (color == color_map['light_grey']) or (color == color_map['light_purple']):
        line_type = "dotted"
        alpha = 1
    else:
        line_type = "solid"
        alpha = 1
    return alpha, line_type


def get_group_label(color, filtered_data, color_map):
    if color == color_map['grey']:
        cluster_label = 'Prediction'
    elif color == color_map['purple']:
        cluster_label = 'Neighborhood prediction'
    elif color == color_map['light_grey']:
        cluster_label = 'Ground truth'
    elif color == color_map['light_purple']:
        cluster_label = 'Neighborhood ground truth'
    elif color == color_map['only_interaction']:
        cluster_label = 'Only Interaction'
    else:
        cluster_label = filtered_data["scatter_label"].iloc[0]
    return cluster_label


def get_group_col(color, item, truth_class, color_map):
    if (color == color_map['light_grey']) or (color == color_map['light_purple']):
        y_col = truth_class
    else:
        y_col = item.predict_class
    return y_col
