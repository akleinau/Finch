import param
import panel as pn
import pandas as pd
from panel.viewable import Viewer
import calculations.data_loader as data_loader
import functions as feature
import calculations.item_functions as item_functions
from calculations import column_functions, similarity


class DataStore(Viewer):

    def __init__(self, **params):
        super().__init__(**params)
        self.file_widget = pn.widgets.FileInput(accept='.csv', name='Upload data')
        self.nn_file_widget = pn.widgets.FileInput(accept='.pkl', name='Upload neural network')
        self.nn_widget = pn.widgets.LiteralInput(value=data_loader.load_nn())
        self.nn = self.nn_widget.value
        # self.data = pn.bind(data_loader.load_data, self.file_widget.rx, self.nn_widget.rx)
        self.x_widget = pn.widgets.EditableIntSlider(name='x', start=0, end=100, value=26)
        self.x = self.x_widget.value
        self.data_widget = pn.widgets.LiteralInput(value=data_loader.load_data()[0:200])
        self.data = self.data_widget.value
        self.raw_data = pn.bind(data_loader.load_data, self.file_widget, self.nn_file_widget)
        self.classes = pn.bind(lambda nn: nn.classes_, self.nn)
        self.means = pn.bind(feature.get_means, self.raw_data)

        # columns
        self.columns = pn.bind(lambda data: [col for col in data.columns], self.raw_data)
        self.col = pn.widgets.Select(name='column', options=self.columns)
        self.all_selected_cols = pn.bind(column_functions.return_col, self.col)

        # groups
        self.cur_feature = pn.widgets.Select(name='', options=self.all_selected_cols, align='center')
        self.column_group = []
        self.combined_columns = pn.widgets.LiteralInput(value=[])
        self.num_groups = pn.widgets.LiteralInput(value=1)
        self.remaining_options = pn.widgets.LiteralInput(value=self.columns)
        self.row = pn.FlexBox()

        # customization widgets
        self.cluster_type = pn.widgets.Select(name='cluster_type', options=['Relative Decision Tree', 'Decision Tree'],
                                              value='Relative Decision Tree')
        self.chart_type = pn.widgets.MultiChoice(name='chart_type', options=['scatter', 'line', 'band', 'contour'],
                                                 value=['line'])
        self.graph_type = pn.widgets.Select(name='graph_type', options=['Cluster', 'Dependency', 'Parallel'],
                                            value='Cluster')

        # update everything when the data changes
        self.file_widget.param.watch(
            lambda event: data_loader.data_changed(event,
                                                   [self.col, self.cur_feature, self.all_selected_cols,
                                                    self.get_widget, self.data_widget, self.nn_widget], self.file_widget.value,
                                                   self.nn_file_widget.value),
            parameter_names=['value'], onlychanged=False)
        self.nn_file_widget.param.watch(lambda event: data_loader.data_changed(event,
                                                                          [self.col, self.cur_feature,
                                                                           self.all_selected_cols, self.get_widget,
                                                                           self.data_widget, self.nn_widget], self.file_widget.value,
                                                                          self.nn_file_widget.value),
                                   parameter_names=['value'], onlychanged=False)

        column_functions.init_groups(self.get_widget())
        self.data_and_probabilities_widget = pn.widgets.LiteralInput(
            value=feature.combine_data_and_results(self.data, self.nn))
        self.data_and_probabilities = self.data_and_probabilities_widget.value

        self.data_widget.param.watch(lambda event: self.data_and_probabilities.param.set_param(
            value=feature.combine_data_and_results(self.data, self.nn)), parameter_names=['value'],
                                     onlychanged=False)

        # item
        self.item_prediction = pn.bind(item_functions.get_item_prediction, self.data_and_probabilities, self.x)
        self.item_shap = pn.bind(item_functions.get_item_shap_values, self.data, self.x, self.means, self.nn,
                                 self.columns, self.combined_columns)

        self.item_data = pn.bind(item_functions.get_item_data, self.data, self.x)
        self.prob_data = pn.bind(item_functions.get_item_probability_string, self.data_and_probabilities, self.x,
                                 self.item_prediction)
        self.prob_wo_selected_cols = pn.bind(item_functions.get_prob_wo_selected_cols, self.nn, self.all_selected_cols,
                                             self.means, self.item_data, self.item_prediction)

        # clustered data
        self.clustered_data_widget = pn.widgets.LiteralInput(value=pn.bind(similarity.get_clustering,self.cluster_type,
                                                                                             self.data_and_probabilities,
                                                                                             self.all_selected_cols,
                                                                                             self.cur_feature,
                                                                                             self.item_prediction, self.x,
                                                                                             exclude_col=False))
        self.clustered_data = self.clustered_data_widget.value

        self.data_and_probabilities_widget.param.watch(self.update_clustered_data, parameter_names=['value'],
                                                       onlychanged=False)
        self.cur_feature.param.watch(self.update_clustered_data, parameter_names=['value'], onlychanged=False)
        self.x_widget.param.watch(self.update_clustered_data, parameter_names=['value'], onlychanged=False)
        self.cluster_type.param.watch(self.update_clustered_data, parameter_names=['value'], onlychanged=False)

    def update_clustered_data(self, event):
        self.clustered_data_widget.param.set_param(
            value=pn.bind(similarity.get_clustering,self.cluster_type, self.data_and_probabilities, self.all_selected_cols,
                                            self.cur_feature, self.item_prediction, self.x, exclude_col=False))

    def get_all_data(self):
        return pn.bind(data_loader.load_data, self.file_widget.value, self.nn_widget.value)

    def get_data(self):
        return pn.bind(lambda data: data[0:200], self.get_all_data())

    def get_file_widgets(self):
        return pn.Row(self.file_widget, self.nn_file_widget, self.x_widget).servable()

    def get_widget(self):
        return [self.column_group, self.row, self.num_groups, self.remaining_options, self.combined_columns,
                self.columns]

    def get_customization_widgets(self):
        return pn.Row(self.cluster_type, self.graph_type, self.chart_type).servable()

    def get_row_widgets(self):
        return self.row.servable()
