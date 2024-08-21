import numpy as np
import pandas as pd
import panel as pn
import param
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure
from panel.viewable import Viewer

from calculations.similarity import get_window_items, get_similar_items
from plots.styling import add_style


class TornadoPlot(Viewer):
    """
    Class to create the tornado plot anbd buttons for the interaction strengths
    """


    plot_single = param.ClassSelector(class_=figure)
    panel_single = param.ClassSelector(class_=pn.Column)
    plot_overview = param.ClassSelector(class_=figure)
    ranked_buttons = param.ClassSelector(class_=pn.FlexBox)
    all_selected_cols = param.List()

    def __init__(self, columns, feature_iter, recommendation, **params):
        all_selected_cols = feature_iter.all_selected_cols
        super().__init__()
        self.remaining_columns = [col for col in columns if col not in all_selected_cols]


        self.dataset_single = recommendation.dataset_single
        self.plot_single = self.tornado_plot(self.dataset_single, feature_iter, "single")
        self.panel_single = pn.Column("## Interaction strength with selected features: ", self.plot_single)
        self.panel_single_first = pn.Column("## Choose a feature: \n ranked by influence, assuming independence",
                                            self.plot_single)
        if len(all_selected_cols) == 0:
            self.dataset_overview = recommendation.dataset_overview
            self.plot_overview = self.tornado_plot(self.dataset_overview, feature_iter, "overview")
            self.ranked_buttons = self.create_ranked_buttons(self.dataset_overview, feature_iter)

        self.all_selected_cols = all_selected_cols
        self.feature_iter = feature_iter
        self.ranked_buttons_text = "## (advanced) fast interaction selection:"

    @param.depends('ranked_buttons')
    def __panel__(self):
        return pn.Column(self.ranked_buttons_text, self.ranked_buttons) if len(self.all_selected_cols) == 0 \
            else pn.Column(styles={'height': '0'})

    @param.depends('ranked_buttons')
    def get_panel_single(self):
        return self.panel_single_first if len(self.all_selected_cols) == 0 else pn.Column(styles={'height': '0'})

    def hide_all(self):
        self.all_selected_cols = []
        self.ranked_buttons_text = ""
        self.panel_single = None
        self.panel_single_first = None
        self.ranked_buttons = pn.FlexBox()

    def set_col(self, data, item_source, feature_iter, type):
        """
        adds the clicked on feature to the selected features
        """

        if len(item_source.selected.indices) > 0:

            self.hide_all()

            if len(item_source.selected.indices) > 1:
                item_source.selected.indices = item_source.selected.indices[1:2]
            select = data.iloc[item_source.selected.indices]
            select = select['feature'].values[0]
            if type == 'overview':
                select = select.split(', ')
                feature_iter.set_all_selected_cols(select)
            else:
                select = select.split(' = ')
                feature_iter.add_col(select[0])

    def tornado_plot(self, data, feature_iter, type):

        if data is None or len(data) == 0:
            return figure()

        width = 350 if type == 'single' else 600

        item_source = ColumnDataSource(data=data)

        max_shap = data['abs_value'].max()
        min_shap = data['abs_value'].min()
        puffer = (max_shap - min_shap) * 0.1

        x_range = (min_shap - puffer, max_shap)

        plot = figure(title="", y_range=data['feature'], x_range=x_range, tools='tap', width=width,
                      toolbar_location=None)

        back_bars_left = plot.hbar(
            y='feature',
            right=plot.x_range.start,
            fill_color="white",
            line_width=0,
            source=item_source,
            nonselection_fill_alpha=0.0,
            selection_fill_alpha=1,
        )

        back_bars_right = plot.hbar(
            y='feature',
            right=plot.x_range.end,
            fill_color="white",
            line_width=0,
            source=item_source,
            nonselection_fill_alpha=0.0,
            selection_fill_alpha=1,
        )

        bars = plot.hbar(
            y='feature',
            right='abs_value',
            fill_color="#381eaa",
            fill_alpha=1,
            nonselection_fill_alpha=1,
            line_width=0,
            source=item_source,
        )

        plot.xaxis.axis_label = "influence" if type == "single" else "interaction strength"

        plot = add_style(plot)

        plot.on_event('tap', lambda event: self.set_col(data, item_source, feature_iter, type))

        hover = HoverTool(renderers=[back_bars_left, back_bars_right], tooltips=[('', '@feature')])

        # hide x axis
        # plot.xaxis.visible = False

        plot.add_tools(hover)

        return plot

    def create_ranked_buttons(self, data, feature_iter):
        """
        displays a fast selection of buttons for the most important feature interactions
        """

        if data is None or len(data) == 0:
            return pn.FlexBox()

        # iterate from behind to get the last 15 entries of data
        buttons = []
        range = [max(0, len(data) - 15), len(data)]

        for index, row in data.iloc[range[0]:range[1]].iterrows():
            button = pn.widgets.Button(name=row['feature'], button_type='default')
            button.on_click(lambda event: self.ranked_buttons_clicked(event, feature_iter))
            buttons.append(button)
        return pn.FlexBox(*reversed(buttons))

    def ranked_buttons_clicked(self, event, feature_iter):
        self.hide_all()
        feature_iter.set_all_selected_cols(event.obj.name.split(', '))


