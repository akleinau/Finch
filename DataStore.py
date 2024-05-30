import param
import panel as pn
import pandas as pd
from panel.viewable import Viewer
import calculations.data_loader as data_loader
import functions as feature
import calculations.item_functions as item_functions
from calculations import column_functions, similarity


class DataStore(Viewer):
    # data = param.DataFrame()
    file_widget = None
    nn_widget = None

    def __init__(self, **params):
        super().__init__(**params)
        self.file_widget = pn.widgets.FileInput(accept='.csv', name='Upload data')
        self.nn_widget = pn.widgets.FileInput(accept='.pkl', name='Upload neural network')
        # self.data = pn.bind(data_loader.load_data, self.file_widget.rx, self.nn_widget.rx)
        self.x = pn.widgets.EditableIntSlider(name='x', start=0, end=100, value=26)
        self.col = pn.widgets.Select(name='column', options=self.get_columns())
        # self.cur_feature = pn.widgets.Select(name='', options=self.get_all_selected_cols(), align='center')
        self.column_group = []
        self.combined_columns = pn.widgets.LiteralInput(value=[])
        self.num_groups = pn.widgets.LiteralInput(value=1)
        self.remaining_options = pn.widgets.LiteralInput(value=self.get_columns())
        self.row = pn.FlexBox().servable()

        # update everything when the data changes
        self.get_file_widget().param.watch(
            lambda event: data_loader.data_changed(event,
           [self.get_col(), self.get_cur_feature(), self.get_all_selected_cols(), self.get_widget]),
                    parameter_names=['value'], onlychanged=False)
        self.get_nn_widget().param.watch(lambda event: data_loader.data_changed(event,
            [self.get_col(), self.get_cur_feature(), self.get_all_selected_cols(), self.get_widget]),
                    parameter_names=['value'], onlychanged=False)


        self.cluster_type = pn.widgets.Select(name='cluster_type', options=['Relative Decision Tree', 'Decision Tree'],
                                         value='Relative Decision Tree')
        self.chart_type = pn.widgets.MultiChoice(name='chart_type', options=['scatter', 'line', 'band', 'contour'],
                                            value=['line'])
        self.graph_type = pn.widgets.Select(name='graph_type', options=['Cluster', 'Dependency', 'Parallel'],
                                       value='Cluster')

        column_functions.init_groups(self.get_widget())

    def get_all_data(self):
        return pn.bind(data_loader.load_data, self.file_widget.value, self.nn_widget.value)

    def get_data(self):
        return pn.bind(lambda data: data[0:200], self.get_all_data())

    def get_file_widgets(self):
        return pn.Row(self.file_widget, self.nn_widget, self.x).servable()

    def get_file_widget(self):
        return self.file_widget

    def get_nn_widget(self):
        return self.nn_widget

    def get_raw_data(self):
        return pn.bind(data_loader.load_data, self.file_widget, self.nn_widget)

    def get_nn(self):
        return pn.bind(data_loader.load_nn, self.nn_widget, self.file_widget)

    def get_means(self):
        return pn.bind(feature.get_means, self.get_raw_data())

    def get_classes(self):
        return pn.bind(lambda nn: nn.classes_, self.get_nn())

    def get_columns(self):
        return pn.bind(lambda data: [col for col in data.columns], self.get_raw_data())

    def get_data_and_probabilities(self):
        return pn.bind(feature.combine_data_and_results, self.get_data(), self.get_nn(), self.get_classes())

    def get_x(self):
        return self.x.value

    def get_item_prediction(self):
        return pn.bind(item_functions.get_item_prediction, self.get_data_and_probabilities(), self.get_x())

    def get_prob_data(self):
        return pn.bind(item_functions.get_item_probability_string, self.get_data_and_probabilities(), self.get_x(),
                       self.get_item_prediction())

    def get_item_data(self):
        return pn.bind(item_functions.get_item_data, self.get_data(), self.get_x())

    def get_col(self):
        return self.col.value

    def get_all_selected_cols(self):
        return pn.bind(column_functions.return_col, self.get_col())

    def get_cur_feature(self):
        return pn.bind(lambda a: a[0], self.get_all_selected_cols())  # self.cur_feature.value

    def get_column_group(self):
        return self.column_group

    def get_combined_columns(self):
        return self.combined_columns.value

    def get_num_groups(self):
        return self.num_groups.value

    def get_remaining_options(self):
        return self.remaining_options.value

    def get_row(self):
        return self.row

    def get_widget(self):
        return [self.column_group, self.row, self.num_groups, self.remaining_options, self.combined_columns,
                self.get_columns()]


    def get_item_shap(self):
        return pn.bind(item_functions.get_item_shap_values, self.get_data(), self.get_x(), self.get_means(), self.get_nn(),
                        self.get_columns(), self.get_combined_columns())


    def get_prob_wo_selected_cols(self):
        return pn.bind(item_functions.get_prob_wo_selected_cols, self.get_nn(), self.get_all_selected_cols(),
                        self.get_means(), self.get_item_data(), self.get_item_prediction())

    def get_clustered_data(self):
        return pn.bind(similarity.get_clustering, self.cluster_type, self.get_data_and_probabilities(),
                        self.get_all_selected_cols(), self.get_cur_feature(), self.get_item_prediction(), self.get_x(),
                        exclude_col=False)

    def get_customization_widgets(self):
        return pn.Row(self.cluster_type, self.graph_type, self.chart_type).servable()