import bokeh.colors
import numpy as np
import pandas as pd
import panel as pn
import param
from bokeh.models import (Band, HoverTool, Legend, LegendItem, BoxAnnotation, LinearAxis, Label)
from bokeh.plotting import figure
from panel.viewable import Viewer

from calculations.data_loader import DataLoader
from calculations.feature_iter import FeatureIter
from calculations.item_functions import Item
from calculations.similarity import get_similar_items, get_window_size, get_window_items
from plots.similar_plot import get_data, add_scatter
from plots.styling import add_style, style_truth, style_additive, style_options


class DependencyPlot(Viewer):
    """
    Class to create a dependency plot
    """

    plot = param.ClassSelector(class_=figure)
    density_plot = param.ClassSelector(class_=figure)

    def __init__(self, simple:bool, **params):
        super().__init__(**params)
        self.plot = None
        self.density_plot = None
        self.relative = True
        self.col = None
        self.item_x = None
        self.truth = False
        self.simple = simple

        # colors
        self.color_map = {'base': '#606060', 'subset': '#A336B0', 'ground_truth': '#909090',
                          'subset_truth': '#aa76c4',
                          'positive_color': '#AE0139', 'negative_color': '#3801AC', 'selected_color': "#19b57A",
                          'previous_prediction': '#b6a0c7', 'additive_prediction': '#595bb0'}

        self.truth_widget = pn.widgets.Toggle(name='show ground truth of the subset', value=False, visible=False,
                                              stylesheets=[style_truth], align="end", icon="timeline")
        self.truth_widget.param.watch(self.truth_changed, parameter_names=['value'], onlychanged=False)
        additive_name = 'show prediction assuming independence of the new feature'
        self.additive_widget = pn.widgets.Toggle(name=additive_name, value=False, visible=False,
                                                 stylesheets=[style_additive], align="end", icon="timeline")
        self.additive_widget.param.watch(self.additive_changed, parameter_names=['value'], onlychanged=False)
        self.normal_widget = pn.widgets.Toggle(name='show normal influence', value=True, visible=True,
                                                  align="end", icon="timeline")
        self.normal_widget.param.watch(self.normal_changed, parameter_names=['value'], onlychanged=False)

        self.toggle_widget = pn.widgets.RadioButtonGroup(options=['change in prediction', 'ground truth',
                                                                  'interaction effect', 'uncertainty'],
                                                         value='change in prediction',
                                                         button_style='outline', stylesheets=[style_options])
        self.toggle_widget.param.watch(self.toggle_changed, parameter_names=['value'], onlychanged=False)

        self.toggle_dict = {
            'change in prediction': '',
            'ground truth': '(ground truth of the subset)',
            'interaction effect': '(highlight interaction effect)',
            'uncertainty': '(show standard deviation of the predictions)'
        }
        self.toggle_help = pn.pane.Markdown(self.toggle_dict[self.toggle_widget.value], styles=dict(margin_left='10px', font_size='15px'))

    def update_plot(self, data: pd.DataFrame, all_selected_cols: list, item: Item,
                    data_loader: DataLoader, feature_iter: FeatureIter, show_process: bool = True, simple_next: bool = True):
        """
        updates the plot with the new data

        :param data: pd.DataFrame
        :param all_selected_cols: list
        :param item: calculations.item_functions.Item
        :param data_loader: calculations.data_loader.DataLoader
        :param show_process: bool
        :param simple_next: bool
        """

        self.truth = "truth" in data.columns
        self.truth_class = "truth_" + item.predict_class[5:]

        toggle_options = ['change in prediction']
        if len(all_selected_cols) >= 1 and self.truth:
            toggle_options.append('ground truth')
        if len(all_selected_cols) > 1 and show_process:
            toggle_options.append('interaction effect')
        toggle_options.append('uncertainty')
        self.toggle_widget.options = toggle_options

        if len(all_selected_cols) == 0:
            self.truth_widget.visible = False
            self.additive_widget.visible = False
            self.plot = None
            self.density_plot = None
            self.col = None
        else:
            self.truth_widget.visible = self.truth
            self.truth_widget.name = 'show ground truth' if len(all_selected_cols) == 1 else \
                'show ground truth of the subset'
            self.additive_widget.visible = len(all_selected_cols) > 1 and show_process

            col = all_selected_cols[0]
            if (self.col != col) or (self.item_x != item.data_prob_raw[col]):
                plot = self.create_figure(col, data, item, data_loader)
                if not self.simple:
                    add_axis(plot, self.y_range_padded, self.color_map)
                add_background(plot, self.y_range_padded)
                self.col = col
                plot = add_style(plot)
                if self.simple:
                    last = all_selected_cols[-1]
                    plot.on_event('tap', lambda event: set_col(last, feature_iter))
                self.plot = self.dependency_scatterplot(plot, all_selected_cols, item, data_loader,
                                                        show_process, False)
                self.item_x = item.data_prob_raw[col]
            else:
                self.remove_old(self.plot, simple_next, all_selected_cols)
                self.plot = self.dependency_scatterplot(self.plot, all_selected_cols, item, data_loader,
                                                        show_process, simple_next)

            self.density_plot = self.create_density_plot(col, item, data_loader, all_selected_cols)
            self.toggle_widget.value = 'change in prediction'

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
            pn.FlexBox(
                pn.pane.Markdown("show ...", styles=dict(font_size='15px')),
                self.toggle_widget,
                self.toggle_help,
               styles=dict(margin='auto', width='100%'),
               sizing_mode="stretch_width", min_width=500, max_width=1000, justify_content="start"),
            styles=dict(margin='auto', width='100%'), align="center")

    def dependency_scatterplot(self, plot: figure, all_selected_cols: list, item: Item,
                               data_loader: DataLoader, previous_prediction: bool = True,
                               simple_next: bool = True) -> figure:
        """
        creates dependency plot

        :param plot: figure
        :param all_selected_cols: list
        :param item: calculations.item_functions.Item
        :param data_loader: calculations.data_loader.DataLoader
        :param previous_prediction: bool
        :param simple_next: bool
        :return: figure
        """

        col = all_selected_cols[0]
        if self.simple:
            last = all_selected_cols[-1]
            plot.title = f"{last} = {item.data_prob_raw[last]:.2f}"
            plot.xaxis.axis_label = last
            plot.title.align = 'center'

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

                self.create_line(plot, alpha, cluster_label, col, color, group_data, line_type,
                                 self.color_map, simple_next)


        # add influence
        create_influence_band(plot, col, color_data, self.color_map, previous_prediction, "normal")
        if len(all_selected_cols) >= 1 and self.truth:
            create_influence_band(plot, col, color_data, self.color_map, previous_prediction, "truth")
        if self.color_map['additive_prediction'] in color_data:
            create_influence_band(plot, col, color_data, self.color_map, previous_prediction, "additive")
        create_uncertainty_band(plot, col, color_data, self.color_map)

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
                keep_colors.append(self.color_map['subset'])
            plot.renderers = [r for r in plot.renderers if
                              ("prediction" not in r.glyph.tags)
                              or any([c in r.glyph.tags for c in keep_colors])]

            # change the current subset line to be the previous prediction line
            plot.legend.items = [i for i in plot.legend.items if i.label.value != "Previous prediction"]
            old_line = [r for r in plot.renderers if self.color_map['subset'] in r.glyph.tags]
            for l in old_line:
                # this actually should only  be one line, hopefully
                l.glyph.tags = ["prediction", self.color_map['previous_prediction']]
                l.glyph.line_color = self.color_map['previous_prediction']
                l.name = "Previous prediction"
                # add to legend
                plot.legend.items = [i for i in plot.legend.items if i.label.value != "Subset Prediction"]
                plot.legend.items.append(LegendItem(label="Previous prediction", renderers=[l]))

        else:
            plot.renderers = [r for r in plot.renderers if "prediction" not in r.glyph.tags]
            plot.legend.items = [i for i in plot.legend.items if i.label.value != "Previous prediction"]

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
        x_range_padded = [self.x_range[0] - 0.025 * (self.x_range[1] - self.x_range[0]),
                          self.x_range[1] + 0.025 * (self.x_range[1] - self.x_range[0])]
        self.y_range = [self.sorted_data[item.predict_class].min(), self.sorted_data[item.predict_class].max()]
        self.y_range_padded = [self.y_range[0] - 0.025 * (self.y_range[1] - self.y_range[0]),
                               self.y_range[1] + 0.05 * (self.y_range[1] - self.y_range[0])]
        if self.simple:
            item_value = item.data_prob_raw[col]
            title = f"{col} = {item_value:.2f}"
            plot = figure(title=title, y_axis_label="prediction", y_range=self.y_range_padded, x_range=x_range_padded,
                          width=250, height=200, toolbar_location=None, x_axis_label=col, tools="tap")
        else:
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
        if not self.simple:
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

            #if self.simple:
                # centers the plot on the item
                #plot.x_range.start = self.item_x - x_std
                #plot.x_range.end = self.item_x + x_std

            if not self.simple:
                # add the label
                plot.add_layout(
                    Label(x=self.item_x, y=self.y_range[1], text=col + " = " + str(self.item_x), text_align='center',
                          text_baseline='bottom', text_font_size='11pt', text_color=self.color_map['selected_color']))

            add_item(plot, col, item, self.y_range, self.color_map)

        # remove stuff for simple
        if self.simple:
            plot = add_style(plot)

            # hide all axes ticks
            plot.xaxis.ticker = []
            plot.yaxis.ticker = []
            plot.yaxis.axis_label_text_font_size = '10pt'

            # center title
            plot.title.align = 'center'

            # remove the legend
            plot.legend.visible = False

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
                           self.color_map['subset_truth'] in r.glyph.tags or self.color_map[
                               'ground_truth'] in r.glyph.tags]

        line = None
        for obj in truth_renderers:
            obj.visible = self.truth_widget.value

            # save the line for the legend
            if obj.glyph.__class__.__name__ == 'Line':
                line = obj

        if line is not None:
            # update the legend
            # plot.legend.items.append(LegendItem(label="previous prediction", renderers=[l]))
            legend_item = [i for i in self.plot.legend.items if i.label.value == "Ground truth"]
            for l in legend_item:
                l.renderers = [line]
                l.visible = self.truth_widget.value

    def additive_changed(self, *params):
        """
        changes the visibility of the additive lines and updates the legend
        """

        additive_renderers = [r for r in self.plot.renderers if self.color_map['additive_prediction'] in r.glyph.tags]

        line = None
        for obj in additive_renderers:
            obj.visible = self.additive_widget.value

            # save the line for the legend
            if obj.glyph.__class__.__name__ == 'Line':
                line = obj

        if line is not None:
            # update the legend
            legend_item = [i for i in self.plot.legend.items if i.label.value == "only main effect"]
            for l in legend_item:
                l.renderers = [line]
                l.visible = self.additive_widget.value

    def prev_line_changed(self, show_line: bool):

        prev_renderers = [r for r in self.plot.renderers if self.color_map['previous_prediction'] in r.glyph.tags]

        if len(prev_renderers) > 0:

            line = None
            for obj in prev_renderers:
                obj.visible = show_line

                # save the line for the legend
                if obj.glyph.__class__.__name__ == 'Line':
                    line = obj

            if line is not None:
                # update the legend
                legend_item = [i for i in self.plot.legend.items if i.label.value == "Previous prediction"]
                for l in legend_item:
                    if line is not None:
                        l.renderers = [line]
                    l.visible = show_line

    def normal_changed(self, *params):

        normal_renderers = [r for r in self.plot.renderers if "influence_normal" in r.glyph.tags]

        for obj in normal_renderers:
            obj.visible = self.normal_widget.value

    def uncertainty_changed(self, show_line: bool):

        uncertainty_renderers = [r for r in self.plot.renderers if "uncertainty" in r.glyph.tags]

        for obj in uncertainty_renderers:
            obj.visible = show_line


    def toggle_changed(self, *params):
        """
        changes the visibility of the truth lines and updates the legend
        """

        if self.toggle_widget.value == "ground truth":
            self.truth_widget.value = True
            self.additive_widget.value = False
            self.normal_widget.value = False
            self.prev_line_changed(False)
            self.uncertainty_changed(False)
        elif self.toggle_widget.value == "interaction effect":
            self.truth_widget.value = False
            self.additive_widget.value = True
            self.normal_widget.value = False
            self.prev_line_changed(True)
            self.uncertainty_changed(False)
        elif self.toggle_widget.value == "uncertainty":
            self.truth_widget.value = False
            self.additive_widget.value = False
            self.normal_widget.value = False
            self.prev_line_changed(False)
            self.uncertainty_changed(True)
        else:
            self.truth_widget.value = False
            self.additive_widget.value = False
            self.normal_widget.value = True
            self.prev_line_changed(True)
            self.uncertainty_changed(False)

        if self.toggle_widget.value is not None:
            self.toggle_help.object = self.toggle_dict[self.toggle_widget.value]

    def create_line(self, chart3: figure, alpha: float, cluster_label: str, col: str, color: str,
                    combined: pd.DataFrame, line_type: str, colors: dict, simple_next: bool):
        line_width = 3.5 if color == colors['subset'] or color == colors['previous_prediction'] else 3 if color == \
                                                                                                                colors[
                                                                                                                    'base'] else 2
        if not simple_next or (color != colors['previous_prediction'] and color != colors['base']):
            if len(combined) > 1:
                line = chart3.line(col, 'mean', source=combined, color=color, line_width=line_width,
                               legend_label=cluster_label, tags=[color, "prediction"],
                               name=cluster_label, line_dash=line_type, alpha=alpha, visible=False)
            else:
                line = chart3.scatter(col, 'mean', source=combined, color=color, size=10, legend_label=cluster_label,
                                     tags=[color, "prediction"], name=cluster_label, alpha=alpha, visible=False)
            if color == colors['ground_truth'] or color == colors['subset_truth']:
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


def create_influence_band(chart3: figure, col: str, color_data: dict, color_map: dict, show_process: bool, type: str):
    """
    creates the influence band in red and blue, highlighting the last influence changes

    :param chart3: figure
    :param col: str
    :param color_data: dict
    :param color_map: dict
    :param show_process: bool
    :return:
    """

    if type == "truth":
        if color_map['subset'] in color_data:
            group_data = color_data[color_map['subset']]
            compare_data = color_data[color_map['subset_truth']]
        else:
            group_data = color_data[color_map['base']]
            compare_data = color_data[color_map['ground_truth']]

    elif type == "additive":
        group_data = color_data[color_map['subset']]
        compare_data = color_data[color_map['additive_prediction']]

    else:

        if show_process and color_map['previous_prediction'] in color_data:
            group_data = color_data[color_map['subset']]
            compare_data = color_data[color_map['previous_prediction']]
        elif color_map['subset'] in color_data:
            group_data = color_data[color_map['subset']]
            compare_data = color_data[color_map['base']]
        else:
            group_data = color_data[color_map['base']]
            compare_data = group_data.copy()
            compare_data['mean'] = 0

    # combine group_data and purple_data to visualize the area between them
    combined = group_data.join(compare_data, lsuffix='_p', rsuffix='_g', how='outer')

    ## fill the missing values with the previous value
    combined = combined.interpolate(method="index")
    combined.reset_index(inplace=True)

    # create two bands, a positive and a negative one
    combined['max'] = combined[['mean_g', 'mean_p']].max(axis=1)
    combined['min'] = combined[['mean_g', 'mean_p']].min(axis=1)

    # tags are needed to display/ hide the influence bands when the user toggles the visibility
    tags = ["prediction"]
    visible = False
    if type == "truth":
        tags.append(color_map['subset_truth'])
    elif type == "additive":
        tags.append(color_map['additive_prediction'])
    else:
        tags.append("influence_normal")
        visible = True

    chart3.varea(x=col, y1='mean_g', y2='max', source=combined, fill_color=color_map['positive_color'],
                 fill_alpha=0.4, level='underlay', tags=tags, visible=visible)
    chart3.varea(x=col, y1='mean_g', y2='min', source=combined, fill_color=color_map['negative_color'],
                 fill_alpha=0.4, level='underlay', tags=tags, visible=visible)

def create_uncertainty_band(chart3: figure, col: str, color_data: dict, color_map: dict):
    if color_map['subset'] in color_data:
        combined = color_data[color_map['subset']].copy()
    else:
        combined = color_data[color_map['base']].copy()

    # color should be a nice mixture of red and blue
    color = "#5324c0"
    cluster_label = "uncertainty"

    # tags are needed to display/ hide the influence bands when the user toggles the visibility
    tags = ["prediction", "uncertainty"]

    band = chart3.varea(x=col, y1='lower', y2='upper', source=combined, level='underlay',
                        fill_color=color, tags=tags, visible=False, fill_alpha=0.4)


def get_rolling(data: pd.DataFrame, y_col: str, col: str) -> pd.DataFrame:
    """
    creates dataframe with rolling mean and quantiles

    :param data: pd.DataFrame
    :param y_col: str
    :param col: str
    :return: pd.Dataframe
    """

    data_subset = data[[col, y_col]].copy()
    individual_values = data_subset[col].unique()


    mean_data = data_subset.groupby(col).agg({y_col: 'mean'})

    # then second smooth of the line
    window = get_window_size(individual_values)
    min_periods = min(window, 10)
    center = window > 1 # for categorical data, we want to uncenter the rolling
    rolling = mean_data[y_col].rolling(window=window, center=center, min_periods=min_periods).agg(
        {'std': 'std',
         'mean': 'mean'})

    # if the data is categorical, we need the std before grouping
    if window == 1:
        rolling['std'] = data_subset.groupby(col).agg({y_col: 'std'})[y_col]

    # smooth the rolling for a nice line
    # alpha is between 0 and 1, and is smaller, the bigger the nr of individual values
    if window > 1: # aka, don't do this for categorical data
        alpha = max(0.01, min(np.sqrt(1/len(individual_values)), 1))
        rolling['std'] = rolling['std'].ewm(alpha=alpha).mean()
        rolling['mean'] = rolling['mean'].ewm(alpha=alpha).mean()


    rolling['upper'] = rolling['mean'] + rolling['std']
    rolling['lower'] = rolling['mean'] - rolling['std']

    mean_data = mean_data.drop(columns=[y_col])

    combined = pd.concat([mean_data, rolling], axis=1)

    # if there are only few values in the data, find where they cross 0 and add a middle value
    if len(combined) < 30:
        for i in range(len(combined) - 1):
            combined = combined.reset_index()
            new_items = []
            if combined['mean'].iloc[i] * combined['mean'].iloc[i+1] < 0:
                new_item = combined.iloc[i].copy()
                a =  abs(combined['mean'].iloc[i]) / (abs(combined['mean'].iloc[i]) + abs(combined['mean'].iloc[i+1]))
                new_item['mean'] = 0
                new_item[col] = round(combined[col].iloc[i] + a * (combined[col].iloc[i+1] - combined[col].iloc[i]), 2)

                # also find new 'upper' and 'lower' values
                new_item['upper'] = round(combined['upper'].iloc[i] + a * (combined['upper'].iloc[i+1] - combined['upper'].iloc[i]), 2)
                new_item['lower'] = round(combined['lower'].iloc[i] + a * (combined['lower'].iloc[i+1] - combined['lower'].iloc[i]), 2)

                new_items.append(new_item)

            combined = pd.concat([combined, pd.DataFrame(new_items)], ignore_index=True)
            combined = combined.sort_values(by=col)
            combined = combined.set_index(col)

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
    elif (color == color_map['subset']) or (color == color_map['subset_truth']):
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


def add_background(chart3: figure, y_range_padded: list):
    # color the background, blue below 0, red above 0
    chart3.add_layout(BoxAnnotation(bottom=y_range_padded[0], top=0, fill_color='#E6EDFF', level='underlay'))
    chart3.add_layout(BoxAnnotation(bottom=0, top=y_range_padded[1], fill_color='#FFE6FF', level='underlay'))


def add_axis(chart3: figure, y_range_padded: list, colors: dict):
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


def add_item(chart3: figure, col: str, item: Item, y_range: list,
             colors: dict):
    line_width = 4
    chart3.line(x=[item.data_prob_raw[col], item.data_prob_raw[col]], y=[y_range[0], y_range[1]],
                line_width=line_width, color=colors['selected_color'], tags=["item"], line_cap='round', level='overlay')


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
        colors.append(color_map['subset'])

    ## light grey for truth
    if truth and len(all_selected_cols) == 1:
        colors.append(color_map['ground_truth'])

    ## light purple for neighbor truth
    if truth and item.type != 'global' and len(all_selected_cols) > 1:
        colors.append(color_map['subset_truth'])

    # show_progress
    if show_progress and item.type != 'global' and len(all_selected_cols) > 2:
        colors.append(color_map['previous_prediction'])

    # show additive
    if show_progress and item.type != 'global' and len(all_selected_cols) > 1:
        colors.append(color_map['additive_prediction'])

    return colors


def get_group_style(color: str, color_map: dict, show_truth, show_additive) -> tuple:
    if (color == color_map['ground_truth']) or (color == color_map['subset_truth']):
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
    elif color == color_map['subset']:
        cluster_label = 'Subset Prediction'
    elif color == color_map['ground_truth']:
        cluster_label = 'Ground truth'
    elif color == color_map['subset_truth']:
        cluster_label = 'Ground truth'
    elif color == color_map['previous_prediction']:
        cluster_label = 'Previous prediction'
    elif color == color_map['additive_prediction']:
        cluster_label = 'only main effect'
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
    if (color == color_map['ground_truth']) or (color == color_map['subset_truth']):
        y_col = truth_class
    else:
        y_col = item.predict_class
    return y_col


def style_axes_main(all_selected_cols, plot):
    if len(all_selected_cols) > 1:
        # add the label "standard" and "subset" as y-axis ticks, on -1 and 1
        plot.yaxis.ticker = [-1, 1]
        plot.yaxis.major_label_overrides = {-1: 'Base', 1: 'Subset'}
    else:
        plot.yaxis.ticker = [-1]
        plot.yaxis.major_label_overrides = {-1: 'Base'}

def set_col(col: str, feature_iter: FeatureIter):
    """
    sets the column to the FeatureIter

    :param col: str
    :param feature_iter: FeatureIter
    """

    feature_iter.add_col(col)