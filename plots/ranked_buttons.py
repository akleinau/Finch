import numpy as np
import pandas as pd
import panel as pn
import param
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure
from panel.viewable import Viewer

from calculations.similarity import get_window_items, get_similar_items
from plots.styling import add_style


class RankedButtons(Viewer):
    """
    Class to create the tornado plot anbd buttons for the interaction strengths
    """

    ranked_buttons = param.ClassSelector(class_=pn.FlexBox)
    all_selected_cols = param.List()

    def __init__(self, columns, feature_iter, recommendation, **params):
        all_selected_cols = feature_iter.all_selected_cols
        super().__init__()
        self.remaining_columns = [col for col in columns if col not in all_selected_cols]

        if len(all_selected_cols) == 0:
            self.dataset_overview = recommendation.dataset_overview
            self.ranked_buttons = self.create_ranked_buttons(self.dataset_overview, feature_iter)

        self.all_selected_cols = all_selected_cols
        self.feature_iter = feature_iter
        self.ranked_buttons_text = "## (advanced) fast interaction selection:"

    @param.depends('ranked_buttons')
    def __panel__(self):
        return pn.Column(self.ranked_buttons_text, self.ranked_buttons) if len(self.all_selected_cols) == 0 \
            else pn.Column(styles={'height': '0'})


    def hide_all(self):
        self.all_selected_cols = []
        self.ranked_buttons_text = ""
        self.ranked_buttons = pn.FlexBox()

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


