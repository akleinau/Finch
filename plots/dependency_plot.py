import bokeh.colors
import pandas as pd
import panel as pn
import param
from bokeh.models import (Band, HoverTool, Legend, LegendItem, BoxAnnotation, LinearAxis, Label)
from bokeh.plotting import figure
from panel.viewable import Viewer

from calculations.data_loader import DataLoader
from calculations.item_functions import Item
from calculations.similarity import get_similar_items, get_window_size, get_window_items
from plots.similar_plot import get_data, add_scatter
from plots.styling import add_style, style_truth, style_additive


class DependencyPlot(Viewer):
    """
    Class to create a dependency plot
    """

    plot = param.ClassSelector(class_=figure)
    density_plot = param.ClassSelector(class_=figure)

    def __init__(self, **params):
        super().__init__(**params)
        self.plot = None
        self.density_plot = None
        self.relative = True
        self.item_style = "grey_line"  # "point", "arrow", "line", "grey_line"
        self.influence_marker = ["color_axis",
                                 "colored_background"]  # "colored_lines", "colored_background", "color_axis", "selective_colored_background"
        self.col = None
        self.item_x = None
        self.truth = False

        # colors
        self.color_map = {'base': '#606060', 'neighborhood': '#A336B0', 'ground_truth': '#909090',
                          'neighborhood_truth': '#aa76c4',
                          'positive_color': '#AE0139', 'negative_color': '#3801AC', 'selected_color': "#19b57A",
                          'previous_prediction': '#b6a0c7', 'additive_prediction': '#595bb0'}

        self.truth_widget = pn.widgets.Toggle(name='show ground truth of the neighborhood', value=False, visible=False,
                                              stylesheets=[style_truth], align="end", icon="timeline")
        self.truth_widget.param.watch(self.truth_changed, parameter_names=['value'], onlychanged=False)
        additive_name = 'show prediction assuming independence of the new feature'
        self.additive_widget = pn.widgets.Toggle(name=additive_name, value=False, visible=False,
                                                 stylesheets=[style_additive], align="end", icon="timeline")
        self.additive_widget.param.watch(self.additive_changed, parameter_names=['value'], onlychanged=False)

    def update_plot(self, data: pd.DataFrame, all_selected_cols: list, item: Item, chart_type: list,
                    data_loader: DataLoader, show_process: bool = True, simple_next: bool = True):
        """
        updates the plot with the new data

        :param data: pd.DataFrame
        :param all_selected_cols: list
        :param item: calculations.item_functions.Item
        :param chart_type: list
        :param data_loader: calculations.data_loader.DataLoader
        :param show_process: bool
        :param simple_next: bool
        """

        self.truth = "truth" in data.columns
        self.truth_class = "truth_" + item.predict_class[5:]

        if len(all_selected_cols) == 0:
            self.truth_widget.visible = False
            self.additive_widget.visible = False
            self.plot = None
            self.density_plot = None
            self.col = None
        else:
            self.truth_widget.visible = self.truth
            self.truth_widget.name = 'show ground truth' if len(all_selected_cols) == 1 else \
                'show ground truth of the neighborhood'
            self.additive_widget.visible = len(all_selected_cols) > 1 and show_process

            col = all_selected_cols[0]
            if (self.col != col) or (self.item_x != item.data_prob_raw[col]):
                plot = self.create_figure(col, data, item, data_loader)
                add_axis(plot, self.influence_marker, self.y_range_padded, self.color_map)
                add_background(plot, self.influence_marker, item, self.mean, self.y_range, self.y_range_padded)
                self.col = col
                plot = add_style(plot)
                self.plot = self.dependency_scatterplot(plot, all_selected_cols, item, chart_type, data_loader,
                                                        show_process, False)
                self.item_x = item.data_prob_raw[col]
            else:
                self.remove_old(self.plot, simple_next, all_selected_cols)
                self.plot = self.dependency_scatterplot(self.plot, all_selected_cols, item, chart_type, data_loader,
                                                        show_process, simple_next)

            self.density_plot = self.create_density_plot(col, item, data_loader, all_selected_cols)

    @param.depends('density_plot')
    def __panel__(self):
        if self.plot is None:
            return pn.Column()

        return pn.Column(
            pn.pane.Markdown("## Feature Interaction:", styles=dict(margin='auto', width='100%'),
                             sizing_mode='stretch_width', min_width=500, max_width=1000, ),
            self.plot,
            pn.pane.Markdown("### Data distribution:", styles=dict(margin='auto', width='100%'),
                             sizing_mode='stretch_width', min_width=500, max_width=1000, ),
            self.density_plot,
            pn.pane.Markdown("### Options:", styles=dict(margin='auto', width='100%', margin_top='15px'),
                             sizing_mode='stretch_width', min_width=500, max_width=1000, ),
            pn.FlexBox(self.truth_widget, self.additive_widget,
                       styles=dict(margin='auto', width='100%'),
                       sizing_mode="stretch_width", min_width=500, max_width=1000, justify_content="start"),
            styles=dict(margin='auto', width='100%'), align="center")

    def dependency_scatterplot(self, plot: figure, all_selected_cols: list, item: Item, chart_type: list,
                               data_loader: DataLoader, previous_prediction: bool = True,
                               simple_next: bool = True) -> figure:
        """
        creates dependency plot

        :param plot: figure
        :param all_selected_cols: list
        :param item: calculations.item_functions.Item
        :param chart_type: list
        :param data_loader: calculations.data_loader.DataLoader
        :param previous_prediction: bool
        :param simple_next: bool
        :return: figure
        """

        col = all_selected_cols[0]

        # add the "standard probability" line
        plot.line(x=[self.x_range[0], self.x_range[1]], y=[0, 0], line_width=1.5, color='#A0A0A0', alpha=1)

        # create bands and contours for each group
        legend_items = []
        colors = get_colors(all_selected_cols, item, self.truth, self.color_map, previous_prediction)

        color_data = {}
        for i, color in enumerate(colors):
            y_col = get_group_col(color, item, self.truth_class, self.color_map)
            color_data[color] = get_filtered_data(color, all_selected_cols, item, self.sorted_data, self.color_map,
                                                  y_col)
            color_data[color] = get_rolling(color_data[color], y_col, col)

        for i, color in enumerate(colors):
            # choose right data
            y_col = get_group_col(color, item, self.truth_class, self.color_map)
            group_data = color_data[color]

            # crop the data to the y_range
            if color == self.color_map['additive_prediction']:
                group_data['mean'] = group_data['mean'].clip(lower=self.y_range[0], upper=self.y_range[1])

            alpha, line_type = get_group_style(color, self.color_map, self.truth_widget.value,
                                               self.additive_widget.value)

            if len(group_data) > 0:
                # choose right label
                cluster_label = get_group_label(color, self.color_map)

                # add legend items
                # dummy_for_legend = plot.line(x=[1, 1], y=[1, 1], line_width=15, color=color, name='dummy_for_legend')
                # legend_items.append((cluster_label, [dummy_for_legend]))

                if "band" in chart_type and color == self.color_map['neighborhood']:
                    create_band(plot, cluster_label, col, color, group_data)

                if "line" in chart_type:
                    self.create_line(plot, alpha, cluster_label, col, color, group_data, line_type,
                                     self.color_map, simple_next)

                if "scatter" in chart_type and color == self.color_map['neighborhood']:
                    data = get_filtered_data(color, all_selected_cols, item, self.sorted_data, self.color_map, y_col)
                    create_scatter(plot, col, color, data, y_col)

        # add influence
        create_influence_band(plot, col, color_data, self.color_map, previous_prediction)

        # add legend
        plot.legend.items.extend([LegendItem(label=x, renderers=y) for (x, y) in legend_items])

        return plot

    def remove_old(self, plot: figure, simple_next: bool, all_selected_cols: list):
        """
        removes the old lines from the plot

        :param plot: figure
        :param simple_next: bool
        :param all_selected_cols: list
        """

        if simple_next:
            keep_colors = [self.color_map['base']]
            if len(all_selected_cols) > 2:
                keep_colors.append(self.color_map['neighborhood'])
            plot.renderers = [r for r in plot.renderers if
                              ("prediction" not in r.glyph.tags)
                              or any([c in r.glyph.tags for c in keep_colors])]

            # change the current neighborhood line to be the previous prediction line
            plot.legend.items = [i for i in plot.legend.items if i.label.value != "Previous prediction"]
            old_line = [r for r in plot.renderers if self.color_map['neighborhood'] in r.glyph.tags]
            for l in old_line:
                # this actually should only  be one line, hopefully
                l.glyph.tags = ["prediction"]
                l.glyph.line_color = self.color_map['previous_prediction']
                l.name = "Previous prediction"
                # add to legend
                plot.legend.items = [i for i in plot.legend.items if i.label.value != "Neighborhood Prediction"]
                plot.legend.items.append(LegendItem(label="Previous prediction", renderers=[l]))

        else:
            plot.renderers = [r for r in plot.renderers if "prediction" not in r.glyph.tags]

    def create_figure(self, col: str, data: pd.DataFrame, item: Item, data_loader: DataLoader) -> figure:
        """
        create the basic figure for the dependency plot. It is reused, to keep user interactions (scrolling, etc) constant

        :param col: str
        :param data: pd.DataFrame
        :param item: Item
        :param data_loader: DataLoader
        :return: figure
        """

        self.sorted_data = data.copy().sort_values(by=col)
        self.mean = data[item.predict_class].mean()
        if self.relative:
            self.sorted_data[item.predict_class] = self.sorted_data[item.predict_class].apply(lambda x: x - self.mean)
            if self.truth:
                self.sorted_data[self.truth_class] = self.sorted_data[self.truth_class].apply(lambda x: x - self.mean)
        self.x_range = (self.sorted_data[col].min(), self.sorted_data[col].max())
        self.item_x = item.data_prob_raw[col]
        x_std = self.sorted_data[col].std()
        x_range_padded = [self.x_range[0], self.x_range[1]]
        self.y_range = [self.sorted_data[item.predict_class].min(), self.sorted_data[item.predict_class].max()]
        self.y_range_padded = [self.y_range[0] - 0.025 * (self.y_range[1] - self.y_range[0]),
                               self.y_range[1] + 0.05 * (self.y_range[1] - self.y_range[0])]
        plot = figure(title="", y_axis_label="change in prediction", tools="tap, xpan, xwheel_zoom",
                      y_range=self.y_range_padded,
                      x_range=x_range_padded, styles=dict(margin='auto', width='100%'),
                      sizing_mode='stretch_both', min_width=500, min_height=400, max_width=1000, max_height=600,
                      toolbar_location=None, active_scroll="xwheel_zoom", x_axis_label=col)
        plot.grid.level = "overlay"
        plot.grid.grid_line_color = "black"
        plot.grid.grid_line_alpha = 0.05
        plot.add_layout(Legend(), 'center')
        plot.legend.orientation = "vertical"
        plot.legend.location = "top_right"
        plot.legend.click_policy = "hide"

        y_range_abs = [self.y_range[0] + self.mean, self.y_range[1] + self.mean]
        plot.extra_y_ranges = {"abs": bokeh.models.Range1d(start=y_range_abs[0], end=y_range_abs[1])}
        plot.add_layout(LinearAxis(y_range_name="abs", axis_label="prediction"), 'right')

        # improve the y-axis by adding a % sign and a plus or minus sign
        if data_loader.type == 'classification':
            plot.yaxis[0].formatter = bokeh.models.CustomJSTickFormatter(args=dict(mean=self.mean),
                                                                         code=""" 
                if (tick == 0) {
                    return 'mean +-0%';
                }
                return (tick > 0 ? '+' : '') + (tick * 100).toFixed(0) + '%'; """)

            # second y-axis for the absolute values
            plot.yaxis[1].formatter = bokeh.models.CustomJSTickFormatter(
                code=""" return (tick * 100).toFixed(0) + '%'; """)


        else:
            plot.yaxis[0].formatter = bokeh.models.CustomJSTickFormatter(
                code="""  
                if (tick == 0) {
                    return 'mean +-0';
                }
                return (tick > 0 ? '+' : '') + tick; """)

            # second y-axis for the absolute values
            plot.yaxis[1].formatter = bokeh.models.CustomJSTickFormatter(
                code=""" return tick; """)

        # add item info
        if item.type != 'global':
            # centers the plot on the item
            # plot.x_range.start = self.item_x - x_std
            # plot.x_range.end = self.item_x + x_std

            # add the label
            plot.add_layout(
                Label(x=self.item_x, y=self.y_range[1], text=col + " = " + str(self.item_x), text_align='center',
                      text_baseline='bottom', text_font_size='11pt', text_color=self.color_map['selected_color']))

            add_item(plot, col, item, self.y_range, self.color_map)

        return plot

    def create_density_plot(self, col: str, item: Item, data_loader: DataLoader, all_selected_cols: list) -> figure:
        """
        creates the density plot for the selected feature, similar to "similar_plot.py"

        :param col: str
        :param item: Item
        :param data_loader: DataLoader
        :param all_selected_cols: list
        :return: figure
        """


        color_similar = "#A336C0"
        color_item = "#19b57A"

        data = get_data(data_loader)
        similar_item_group = get_similar_items(data, item, all_selected_cols[1:])

        plot = figure(title="", toolbar_location=None, tools="tap, xpan, xwheel_zoom", width=900,
                      sizing_mode='stretch_both', min_width=500, min_height=100, max_width=1000, max_height=300,
                      styles=dict(margin='auto', width='100%'),
                      x_range=self.plot.x_range, active_scroll="xwheel_zoom", )
        add_scatter(all_selected_cols, col, color_item, color_similar, data, item, plot, similar_item_group)

        plot = add_style(plot)

        style_axes_main(all_selected_cols, plot)

        # hide legend
        plot.add_layout(Legend(), 'above')
        plot.legend.visible = False

        return plot

    def truth_changed(self, *params):
        """
        changes the visibility of the truth lines and updates the legend
        """

        truth_renderers = [r for r in self.plot.renderers if
                           self.color_map['neighborhood_truth'] in r.glyph.tags or self.color_map[
                               'ground_truth'] in r.glyph.tags]

        for obj in truth_renderers:
            obj.visible = self.truth_widget.value

            # update the legend
            # plot.legend.items.append(LegendItem(label="previous prediction", renderers=[l]))
            legend_item = [i for i in self.plot.legend.items if i.label.value == "Ground truth"]
            for l in legend_item:
                l.renderers = [obj]

    def additive_changed(self, *params):
        """
        changes the visibility of the additive lines and updates the legend
        """

        additive_renderers = [r for r in self.plot.renderers if self.color_map['additive_prediction'] in r.glyph.tags]

        for obj in additive_renderers:
            obj.visible = self.additive_widget.value

            # update the legend
            legend_item = [i for i in self.plot.legend.items if i.label.value == "assuming independence"]
            for l in legend_item:
                l.renderers = [obj]

    def create_line(self, chart3: figure, alpha: float, cluster_label: str, col: str, color: str,
                    combined: pd.DataFrame, line_type: str, colors: dict, simple_next: bool):
        line_width = 3.5 if color == colors['neighborhood'] or color == colors['previous_prediction'] else 3 if color == \
                                                                                                                colors[
                                                                                                                    'base'] else 2
        if not simple_next or (color != colors['previous_prediction'] and color != colors['base']):
            line = chart3.line(col, 'mean', source=combined, color=color, line_width=line_width,
                               legend_label=cluster_label, tags=[color, "prediction"],
                               name=cluster_label, line_dash=line_type, alpha=alpha, visible=False)
            if color == colors['ground_truth'] or color == colors['neighborhood_truth']:
                line.visible = self.truth_widget.value
                line.on_change('visible', self.set_truth)
            elif color == colors['additive_prediction']:
                line.visible = self.additive_widget.value
                line.on_change('visible', self.set_additive)
            else:
                line.visible = True
            line_hover = HoverTool(renderers=[line], tooltips=[('', '$name')])
            chart3.add_tools(line_hover)

    def set_truth(self, attr, old, new):
        self.truth_widget.value = new

    def set_additive(self, attr, old, new):
        self.additive_widget.value = new


def create_influence_band(chart3: figure, col: str, color_data: dict, color_map: dict, show_process: bool):
    """
    creates the influence band in red and blue, highlighting the last influence changes

    :param chart3: figure
    :param col: str
    :param color_data: dict
    :param color_map: dict
    :param show_process: bool
    :return:
    """

    if show_process and color_map['previous_prediction'] in color_data:
        group_data = color_data[color_map['neighborhood']]
        compare_data = color_data[color_map['previous_prediction']]
    elif color_map['neighborhood'] in color_data:
        group_data = color_data[color_map['neighborhood']]
        compare_data = color_data[color_map['base']]
    else:
        group_data = color_data[color_map['base']]
        compare_data = group_data.copy()
        compare_data['mean'] = 0

    # combine group_data and purple_data to visualize the area between them
    combined = group_data.join(compare_data, lsuffix='_p', rsuffix='_g', how='outer')

    ## fill the missing values with the previous value
    combined = combined.interpolate()
    combined.reset_index(inplace=True)
    # create two bands, a positive and a negative one
    combined['max'] = combined[['mean_g', 'mean_p']].max(axis=1)
    combined['min'] = combined[['mean_g', 'mean_p']].min(axis=1)
    chart3.varea(x=col, y1='mean_g', y2='max', source=combined, fill_color=color_map['positive_color'],
                 fill_alpha=0.4, level='underlay', tags=["prediction"])
    chart3.varea(x=col, y1='mean_g', y2='min', source=combined, fill_color=color_map['negative_color'],
                 fill_alpha=0.4, level='underlay', tags=["prediction"])


def get_rolling(data: pd.DataFrame, y_col: str, col: str) -> pd.DataFrame:
    """
    creates dataframe with rolling mean and quantiles

    :param data: pd.DataFrame
    :param y_col: str
    :param col: str
    :return: pd.Dataframe
    """

    # first get mean per value of the col
    mean_data = data.groupby(col).agg({y_col: 'mean'})

    # then smooth the line
    window = get_window_size(mean_data)
    rolling = mean_data[y_col].rolling(window=window, center=False, min_periods=1).agg(
        {'lower': lambda ev: ev.quantile(.25, interpolation='lower'),
         'upper': lambda ev: ev.quantile(.75, interpolation='higher'),
         'mean': 'mean'})

    second_window = min(window, 10)
    rolling = rolling.rolling(window=second_window, center=False, min_periods=1).mean()

    mean_data = mean_data.drop(columns=[y_col])

    combined = pd.concat([mean_data, rolling], axis=1)
    return combined


def get_filtered_data(color: str, all_selected_cols: list, item: Item, sorted_data: pd.DataFrame,
                      color_map: dict, y_col) -> pd.DataFrame:
    """
    returns the data used for the calculation of the current line

    :param color: str
    :param all_selected_cols: list
    :param item: item_functions.Item
    :param sorted_data: pd.DataFrame
    :param color_map: dict
    :return: pd.DataFrame
    """

    include_cols = all_selected_cols[1:]

    if (color == color_map['base']) or (color == color_map['ground_truth']):
        filtered_data = sorted_data
    elif (color == color_map['neighborhood']) or (color == color_map['neighborhood_truth']):
        filtered_data = get_similar_items(sorted_data, item, include_cols)
    elif color == color_map['previous_prediction']:
        if len(include_cols) == 1:
            filtered_data = sorted_data
        else:
            filtered_data = get_similar_items(sorted_data, item, include_cols[:-1])
    elif color == color_map['additive_prediction']:
        # first get the prediction of just the newest feature alone
        last_col = include_cols[-1]
        single_mean = get_window_items(sorted_data, item, last_col, y_col)[y_col].mean()

        # get previous prediction data
        if len(include_cols) == 0:
            filtered_data = sorted_data.copy()
            filtered_data[y_col] = 0
        elif len(include_cols) == 1:
            filtered_data = sorted_data.copy()
        else:
            filtered_data = get_similar_items(sorted_data, item, include_cols[:-1]).copy()
        filtered_data[y_col] = filtered_data[y_col] + single_mean
    else:
        filtered_data = sorted_data[sorted_data["scatter_group"] == color]
    return filtered_data


def add_background(chart3: figure, influence_marker: list, item: Item, mean: float, y_range: list,
                   y_range_padded: list):
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


def add_axis(chart3: figure, influence_marker: list, y_range_padded: list, colors: dict):
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


def add_item(chart3: figure, col: str, item: Item, y_range: list,
             colors: dict):
    line_width = 4
    chart3.line(x=[item.data_prob_raw[col], item.data_prob_raw[col]], y=[y_range[0], y_range[1]],
                line_width=line_width, color=colors['selected_color'], tags=["item"], line_cap='round', level='overlay')


def create_scatter(chart3: figure, col: str, color: str, filtered_data: pd.DataFrame, y_col: str):
    alpha = 0.3
    chart3.scatter(col, y_col, color=color, source=filtered_data,
                   alpha=alpha, marker='circle', size=3, name="scatter_label",
                   # legend_group="scatter_label"
                   )


def create_band(chart3: figure, cluster_label: str, col: str, color: str, combined: pd.DataFrame):
    band = chart3.varea(x=col, y1='lower', y2='upper', source=combined,
                        fill_color=color,
                        alpha=0.3, name=cluster_label)
    band_hover = HoverTool(renderers=[band], tooltips=[('', '$name')])
    chart3.add_tools(band_hover)


def get_colors(all_selected_cols: list, item: Item, truth: bool, color_map: dict, show_progress: bool) -> list:
    """
    creates a list with all colors/ lines that will be included

    :param all_selected_cols: list
    :param item: item_functions.Item
    :param truth: bool
    :param color_map: dict
    :param show_progress: bool
    :return: list
    """

    colors = []

    # grey for the standard group
    # if not show_progress or len(all_selected_cols) == 1:
    colors.append(color_map['base'])

    # purple for neighbors
    if len(all_selected_cols) > 1:
        colors.append(color_map['neighborhood'])

    ## light grey for truth
    if truth and len(all_selected_cols) == 1:
        colors.append(color_map['ground_truth'])

    ## light purple for neighbor truth
    if truth and item.type != 'global' and len(all_selected_cols) > 1:
        colors.append(color_map['neighborhood_truth'])

    # show_progress
    if show_progress and item.type != 'global' and len(all_selected_cols) > 2:
        colors.append(color_map['previous_prediction'])

    # show additive
    if show_progress and item.type != 'global' and len(all_selected_cols) > 1:
        colors.append(color_map['additive_prediction'])

    return colors


def get_group_style(color: str, color_map: dict, show_truth, show_additive) -> tuple:
    if (color == color_map['ground_truth']) or (color == color_map['neighborhood_truth']):
        line_type = "dotted"
        alpha = 1
    elif color == color_map['additive_prediction']:
        line_type = "dashed"
        alpha = 1
    else:
        line_type = "solid"
        alpha = 1
    return alpha, line_type


def get_group_label(color: str, color_map: dict) -> str:
    if color == color_map['base']:
        cluster_label = 'Base Prediction'
    elif color == color_map['neighborhood']:
        cluster_label = 'Neighborhood Prediction'
    elif color == color_map['ground_truth']:
        cluster_label = 'Ground truth'
    elif color == color_map['neighborhood_truth']:
        cluster_label = 'Ground truth'
    elif color == color_map['previous_prediction']:
        cluster_label = 'Previous prediction'
    elif color == color_map['additive_prediction']:
        cluster_label = 'assuming independence'
    else:
        cluster_label = ''
    return cluster_label


def get_group_col(color: str, item: Item, truth_class: str, color_map: dict) -> str:
    """
    returns the column that is used for the y values

    :param color: str
    :param item: Item
    :param truth_class: str
    :param color_map: dict
    :return: str
    """
    if (color == color_map['ground_truth']) or (color == color_map['neighborhood_truth']):
        y_col = truth_class
    else:
        y_col = item.predict_class
    return y_col


def style_axes_main(all_selected_cols, plot):
    if len(all_selected_cols) > 1:
        # add the label "standard" and "neighborhood" as y-axis ticks, on -1 and 1
        plot.yaxis.ticker = [-1, 1]
        plot.yaxis.major_label_overrides = {-1: 'Base', 1: 'Neighborhood'}
    else:
        plot.yaxis.ticker = [-1]
        plot.yaxis.major_label_overrides = {-1: 'Base'}
