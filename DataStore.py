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
        self.file = pn.widgets.FileInput(accept='.csv', name='Upload data')
        self.nn_file = pn.widgets.FileInput(accept='.pkl', name='Upload neural network')
        self.nn = pn.widgets.LiteralInput(value=data_loader.load_nn())
        self.item_index = pn.widgets.EditableIntSlider(name='item index', start=0, end=100, value=26)
        self.data = pn.widgets.LiteralInput(value=data_loader.load_data()[0:200])
        self.classes = pn.bind(lambda nn: nn.classes_, self.nn.value)
        self.means = pn.bind(feature.get_means, self.data.value)

        # columns
        self.columns = pn.bind(lambda data: [col for col in data.columns], self.data)
        self.col = pn.widgets.Select(name='column', options=self.columns)
        self.all_selected_cols = pn.widgets.LiteralInput(value=pn.bind(column_functions.return_col, self.col))

        # groups
        self.cur_feature = pn.widgets.Select(name='', options=self.all_selected_cols.value, align='center')
        self.all_selected_cols.param.watch(lambda event: self.cur_feature.param.update(
            options=event.new, value=event.new[0]), parameter_names=['value'], onlychanged=False)

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
        self.file.param.watch(self.update_data,
                              parameter_names=['value'], onlychanged=False)
        self.nn_file.param.watch(self.update_data,
                                 parameter_names=['value'], onlychanged=False)

        column_functions.init_groups(self.get_widget())
        self.data_and_probabilities = pn.widgets.LiteralInput(
            value=feature.combine_data_and_results(self.data.value, self.nn.value))

        self.data.param.watch(lambda event: self.data_and_probabilities.param.update(
            value=feature.combine_data_and_results(self.data.value, self.nn.value)), parameter_names=['value'],
                              onlychanged=False)

        # item
        self.item_prediction = pn.bind(item_functions.get_item_prediction, self.data_and_probabilities.value,
                                       self.item_index.value)
        self.item_shap = pn.bind(item_functions.get_item_shap_values, self.data, self.item_index, self.means,
                                 self.nn.value,
                                 self.columns, self.combined_columns)

        self.item_data = pn.bind(item_functions.get_item_data, self.data, self.item_index)
        self.prob_data = pn.bind(item_functions.get_item_probability_string, self.data_and_probabilities.value,
                                 self.item_index.value,
                                 self.item_prediction)
        self.prob_wo_selected_cols = pn.bind(item_functions.get_prob_wo_selected_cols, self.nn.value,
                                             self.all_selected_cols.value,
                                             self.means, self.item_data, self.item_prediction)

        # clustered data
        self.clustered_data = pn.widgets.LiteralInput(value=self._update_clustered_data(None))

        self.data_and_probabilities.param.watch(self.update_clustered_data, parameter_names=['value'],
                                                onlychanged=False)
        self.cur_feature.param.watch(self.update_clustered_data, parameter_names=['value'], onlychanged=False)
        self.item_index.param.watch(self.update_clustered_data, parameter_names=['value'], onlychanged=False)
        self.cluster_type.param.watch(self.update_clustered_data, parameter_names=['value'], onlychanged=False)

    def update_data(self, event):
        data_loader.data_changed(event,
                                 [self.col, self.cur_feature.value,
                                  self.all_selected_cols.value, self.get_widget,
                                  self.data, self.nn],
                                 self.file.value,
                                 self.nn_file.value)

    def _update_clustered_data(self, event):
        return pn.bind(similarity.get_clustering, self.cluster_type, self.data_and_probabilities.value,
                       self.all_selected_cols.value,
                       self.cur_feature.value, self.item_prediction, self.item_index.value, exclude_col=False)

    def update_clustered_data(self, event):
        self.clustered_data.param.update(
            value=self._update_clustered_data(event))

    def get_all_data(self):
        return pn.bind(data_loader.load_data, self.file.value, self.nn.value)

    def get_data(self):
        return pn.bind(lambda data: data[0:200], self.get_all_data())

    def get_file_widgets(self):
        return pn.Row(self.file, self.nn_file, self.item_index).servable()

    def get_widget(self):
        return [self.column_group, self.row, self.num_groups, self.remaining_options, self.combined_columns,
                self.columns]

    def get_customization_widgets(self):
        return pn.Row(self.cluster_type, self.graph_type, self.chart_type).servable()

    def get_row_widgets(self):
        return self.row.servable()
