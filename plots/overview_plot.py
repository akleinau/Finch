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


class OverviewPlot(Viewer):

    ranked_plots = param.ClassSelector(class_=pn.FlexBox)
    all_selected_cols = param.List()

    def __init__(self):
        super().__init__()

        self.ranked_plots = pn.FlexBox()

    @param.depends('ranked_plots')
    def __panel__(self) -> Viewable:
        if len(self.ranked_plots.objects) == 0:
            return pn.Column()
        return pn.Column("## Feature Overview: ",
                         pn.Row(
                             pn.pane.Markdown("**mean prediction per feature value**", styles=dict(color='#606060')),
                                pn.pane.Markdown("**selected item**", styles=dict(color='#19b57A')),
                                #pn.pane.Markdown("**mean prediction**", styles=dict(color='#A0A0A0')),
                                #pn.pane.Markdown("**positive influence**", styles=dict(color='#AE0139')),
                                #pn.pane.Markdown("**negative influence**", styles=dict(color='#3801AC')),
                                ),
                         self.ranked_plots)

    def update(self, data, item, y_col, columns, feature_iter):
        """
        updates the plot with the new data

        :param data: pd.DataFrame
        :param item: Item
        :param y_col: str
        :param columns: list
        :param all_selected_cols: list
        :return:
        """

        ranked_plots = pn.FlexBox()


        if len(feature_iter.all_selected_cols) == 0:

            mean_prob = data[y_col].mean()

            # get all singular features
            single_dict = {}
            for col in columns:
                # get prediction of the col on its own
                single_dict[col] = get_window_items(data, item, col, y_col)[y_col].mean() - mean_prob

            # sort the features by their impact
            sorted_single_dict = dict(sorted(single_dict.items(), key=lambda i: i[1], reverse=True))

            # extract sorted list of features
            sorted_single_list = list(sorted_single_dict.keys())

            y_range = [data[y_col].min(), data[y_col].max()]

            for col in sorted_single_list:
                plot = create_plot(data, item, y_col, col, y_range, mean_prob, feature_iter)
                ranked_plots.append(plot)

        self.ranked_plots = ranked_plots


def create_plot(data: pd.DataFrame, item: pd.DataFrame, y_col: str, col: str, y_range: list, mean_prob: float,
                feature_iter: FeatureIter) -> figure:
    """
    creates a PDP plot, only that we show the real data

    :param data: pd.DataFrame
    :param item: Item
    :param y_col: str
    :param col: str
    :param y_range: list
    :param mean_prob: float
    :param feature_iter: FeatureIter
    :return: figure
    """

    item_value = item.data_prob_raw[col]
    title = f"{col} = {item_value}"

    plot = figure(title=title, x_axis_label=col, y_axis_label=y_col, width=250, height=200,
                  toolbar_location=None, y_range=y_range)

    # background
    #plot.background_fill_color = "#FAFAFA"
    plot.add_layout(BoxAnnotation(bottom=y_range[0], top=mean_prob, fill_color='#E6EDFF', level='underlay'))
    plot.add_layout(BoxAnnotation(bottom=mean_prob, top=y_range[1], fill_color='#FFE6FF', level='underlay'))

    rolling = get_rolling(data, y_col, col)

    # mean line
    plot.line(x=[data[col].min(), data[col].max()], y=[mean_prob, mean_prob], line_width=1, color='#A0A0A0', alpha=0.5)

    # line
    plot.line(x=col, y='mean', source=rolling, color='#606060', line_width=3)

    # item
    plot.line(x=[item.data_prob_raw[col], item.data_prob_raw[col]], y=[y_range[0], y_range[1]],
                line_width=2, color='#19b57A', tags=["item"], line_cap='round', level='overlay', alpha=0.5)

    # add interaction
    plot.on_event('tap', lambda event: set_col(col, feature_iter))

    # influence band
    group_data = rolling
    compare_data = group_data.copy()
    compare_data['mean'] = mean_prob

    # combine group_data and purple_data to visualize the area between them
    combined = group_data.join(compare_data, lsuffix='_p', rsuffix='_g', how='outer')

    ## fill the missing values with the previous value
    combined = combined.interpolate()
    combined.reset_index(inplace=True)
    # create two bands, a positive and a negative one
    combined['max'] = combined[['mean_g', 'mean_p']].max(axis=1)
    combined['min'] = combined[['mean_g', 'mean_p']].min(axis=1)
    plot.varea(x=col, y1='mean_g', y2='max', source=combined, fill_color='#AE0139',
                 fill_alpha=0.4, level='underlay', tags=["prediction"])
    plot.varea(x=col, y1='mean_g', y2='min', source=combined, fill_color='#3801AC',
                 fill_alpha=0.4, level='underlay', tags=["prediction"])


    plot = add_style(plot)


    # hide all axes ticks
    plot.xaxis.ticker = []
    plot.yaxis.ticker = []
    plot.yaxis.axis_label_text_font_size = '10pt'

    # center title
    plot.title.align = 'center'

    return plot

def set_col(col: str, feature_iter: FeatureIter):
    """
    sets the column to the FeatureIter

    :param col: str
    :param feature_iter: FeatureIter
    """

    feature_iter.set_all_selected_cols([col])