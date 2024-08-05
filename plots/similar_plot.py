import panel as pn
import param
from bokeh.models import Legend
from bokeh.plotting import figure
from bokeh.transform import jitter
from panel.viewable import Viewer

from calculations.data_loader import DataLoader
from calculations.item_functions import Item
from calculations.similarity import get_similar_items
from plots.styling import add_style


class SimilarPlot(Viewer):
    """
    Class to create visualization of the distribution of similar items per column
    """

    plot = param.ClassSelector(class_=pn.Column)

    def __init__(self, data_loader: DataLoader, item: Item, all_selected_cols: list, **params):
        super().__init__(**params)
        self.plot = similar_plot(data_loader, item, all_selected_cols)

    @param.depends('plot')
    def __panel__(self):
        return self.plot


def similar_plot(data_loader: DataLoader, item: Item, all_selected_cols: list) -> pn.Column:
    """
    creates a plot with the distribution of the data and the similar items per column

    :param data_loader: calculations.data_loader.DataLoader
    :param item: calculations.item_functions.Item
    :param all_selected_cols: list
    :return: pn.Column
    """

    if len(all_selected_cols) == 0:
        return pn.Column()

    cur_feature = all_selected_cols[0]

    color_similar = "#A336C0"
    color_item = "#19b57A"

    column_criteria = "curr"

    include_cols = []
    if column_criteria == "selected":
        include_cols = all_selected_cols
    elif column_criteria == "curr":
        include_cols = [col for col in all_selected_cols if col != cur_feature]

    data = get_data(data_loader)
    similar_item_group = get_similar_items(data, item, include_cols)

    diff = find_order(data, similar_item_group)

    # for each column, create a bokeh plot with the distribution of the data
    plot_list = pn.Column(styles=dict(margin='auto'))

    display_cols = diff.index.tolist()
    include_cols_for_display = all_selected_cols if len(all_selected_cols) > 0 else data_loader.data.columns
    display_cols = [col for col in display_cols if col in include_cols_for_display and col != cur_feature]
    display_cols = [cur_feature, *display_cols]  # make sur cur_feature is first

    for i, col in enumerate(display_cols):

        if i == 0:
            continue

        # create a figure
        x_range = [data[col].min(), data[col].max()]
        plot = figure(title="Similar items", x_range=x_range, toolbar_location=None, height=80, width=300,
                      sizing_mode='fixed', tools='')

        if i == 0:
            plot.add_layout(Legend(), 'above')
            plot.height += 60
            plot.legend.orientation = "horizontal"
        else:
            # hide legend
            plot.add_layout(Legend(), 'above')
            plot.legend.visible = False

        add_scatter(all_selected_cols, col, color_item, color_similar, data, item, plot, similar_item_group)

        # add the mean of the data and of similar_item_group as lines
        # data_mean = data[col].mean()
        # similar_item_group_mean = similar_item_group[col].mean()
        # plot.line([data_mean, data_mean], [0, 2], color='grey', line_width=2, alpha=0.9, legend_label='Standard mean')
        # plot.line([similar_item_group_mean, similar_item_group_mean], [0, 2], color='#9932CC', line_width=2)

        plot = add_style(plot)

        plot.yaxis.axis_label = col + " = " + "{:.2f}".format(item.data_raw[col].values[0])
        plot.yaxis.axis_label_orientation = "horizontal"

        style_axes(plot)

        plot.title.visible = False

        if i == 1:
            # add divider
            plot_list.append("## Neighborhood: \n using **" + str(len(similar_item_group)) + "** similar instances")

        plot_list.append(plot)

    return plot_list


def style_axes(plot):
    # hide ticks of the yaxis but not the label
    plot.yaxis.major_tick_line_color = None
    plot.yaxis.minor_tick_line_color = None
    plot.yaxis.major_label_text_font_size = '0pt'


def add_scatter(all_selected_cols, col, color_item, color_similar, data, item, plot, similar_item_group):
    # add points
    alpha = max(min(100 / len(data), 0.2), 0.05)
    plot.scatter(x=jitter(col, 0.5), y=jitter('fixed2', 2), alpha=alpha, source=data, size=2, color='grey',
                 legend_label='Standard')
    if len(all_selected_cols) > 1:
        alpha = max(min(100 / len(similar_item_group), 0.2), 0.05)  # find a good alpha value based on length
        plot.scatter(x=jitter(col, 0.5), y=jitter('fixed', 2), alpha=alpha, source=similar_item_group, size=5,
                     color=color_similar, legend_label='Neighborhood')

    plot.line([item.data_prob_raw[col], item.data_prob_raw[col]], [-2, 2], color=color_item,
              line_width=4, legend_label='Item')


def get_data(data_loader):
    data = data_loader.data.copy()
    data['fixed'] = 1
    data['fixed2'] = -1
    return data


def find_order(data, similar_item_group):
    # normalize the data
    normalized_data = data.copy()
    normalized_similar_item_group = similar_item_group.copy()
    for col in data.columns.drop('fixed'):
        mean = data[col].mean()
        std = data[col].std()
        normalized_data[col] = (data[col] - mean) / std
        normalized_similar_item_group[col] = (similar_item_group[col] - mean) / std
    # calculate the difference per column
    data_mean = normalized_data.mean()
    similar_item_group_mean = normalized_similar_item_group.mean()
    diff = data_mean - similar_item_group_mean
    diff = diff.drop('fixed')
    diff = diff.abs()
    diff = diff.sort_values(ascending=False)
    return diff
