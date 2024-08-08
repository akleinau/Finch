import pandas as pd
import panel as pn
import param
from bokeh.models import BoxAnnotation
from bokeh.plotting import figure
from panel.viewable import Viewer, Viewable

from calculations.feature_iter import FeatureIter
from calculations.similarity import get_window_items
from plots.dependency_plot import get_rolling
from plots.styling import add_style
import plots.dependency_plot as dependency_plot


class OverviewPlot(Viewer):
    ranked_plots = param.ClassSelector(class_=pn.FlexBox)
    all_selected_cols = param.List()

    def __init__(self):
        super().__init__()

        self.ranked_plots = pn.FlexBox()
        self.all_selected_cols = []

    @param.depends('ranked_plots')
    @param.depends('all_selected_cols')
    def __panel__(self) -> Viewable:
        if len(self.ranked_plots.objects) == 0:
            return pn.Column()
        elif len(self.all_selected_cols) == 0:
            return pn.Column("## Feature Overview: ",
                             pn.Row(
                                 pn.pane.Markdown("**mean prediction per feature value**", styles=dict(color='#606060')),
                                 pn.pane.Markdown("**selected item**", styles=dict(color='#19b57A')),
                                 # pn.pane.Markdown("**mean prediction**", styles=dict(color='#A0A0A0')),
                                 # pn.pane.Markdown("**positive influence**", styles=dict(color='#AE0139')),
                                 # pn.pane.Markdown("**negative influence**", styles=dict(color='#3801AC')),
                             ),
                             self.ranked_plots)
        else:
            return pn.Column()

    def update(self, data, item, y_col, feature_iter, recommendation, data_loader):
        """
        updates the plot with the new data

        :param data: pd.DataFrame
        :param item: Item
        :param y_col: str
        :param columns: list
        :param all_selected_cols: list
        :param feature_iter: FeatureIter
        :return:
        """
        self.all_selected_cols = feature_iter.all_selected_cols

        ranked_plots = pn.FlexBox()

        # go through each row of the dataset recommendation.dataset_single
        for index, row in recommendation.dataset_single.iterrows():

            col = row['feature']
            dp = dependency_plot.DependencyPlot(simple=True)
            dp.update_plot(data, self.all_selected_cols + [col], item, data_loader, feature_iter, True, False)
            #dp.toggle_widget.value = "independence prediction"

            plot = dp.plot
            ranked_plots.append(plot)

        #reverse the order
        ranked_plots.objects = ranked_plots.objects[::-1]

        self.ranked_plots = ranked_plots
