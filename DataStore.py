import param
import panel as pn
import pandas as pd
from panel.viewable import Viewer, Viewable
import calculations.data_loader as data_loader
import functions as feature
import calculations.item_functions as item_functions
from calculations import column_functions, similarity
from plots.cluster_bar_plot import cluster_bar_plot
from plots.dependency_plot import dependency_scatterplot
from plots.parallel_plot import parallel_plot


class DataStore(Viewer):

    def __init__(self, **params):
        super().__init__(**params)
        self.active = True
        self.file = pn.widgets.FileInput(accept='.csv', name='Upload data')
        self.nn_file = pn.widgets.FileInput(accept='.pkl', name='Upload neural network')
        self.calculate = pn.widgets.Button(name='Calculate')
        self.calculate.on_click(self.update_data)
        self.data_loader = pn.widgets.LiteralInput(value=data_loader.DataLoader())
        self.item_index = pn.widgets.EditableIntSlider(name='item index', start=0, end=100, value=26)

        # columns
        self.col = pn.widgets.Select(name='column', options=self.data_loader.value.columns)
        self.all_selected_cols = pn.widgets.LiteralInput(value=column_functions.return_col(self.col.value))
        self.col.param.watch(lambda event: self.all_selected_cols.param.update(value=column_functions.return_col(event.new)),
                                parameter_names=['value'], onlychanged=False)

        # groups
        self.cur_feature = pn.widgets.Select(name='', options=self.all_selected_cols.value, align='center')
        self.all_selected_cols.param.watch(lambda event: self.cur_feature.param.update(options=event.new),
                                             parameter_names=['value'], onlychanged=False)

        self.column_grouping = column_functions.ColumnGrouping(self.data_loader.value.columns)

        # customization widgets
        self.cluster_type = pn.widgets.Select(name='cluster_type', options=['Relative Decision Tree', 'Decision Tree'],
                                              value='Relative Decision Tree')
        self.chart_type = pn.widgets.MultiChoice(name='chart_type', options=['scatter', 'line', 'band', 'contour'],
                                                 value=['line'])

        self.graph_type = pn.widgets.Select(name='graph_type', options=['Cluster', 'Dependency', 'Parallel'],
                                            value='Cluster')

        self.column_grouping.init_groups()
        self.data_and_probabilities = pn.widgets.LiteralInput(
            value=feature.combine_data_and_results(self.data_loader.value))

        # item
        self.item = pn.widgets.LiteralInput(
            value=item_functions.Item(self.data_loader.value, self.data_and_probabilities.value, self.item_index.value,
                                      self.column_grouping.combined_columns))
        self.item_index.param.watch(lambda event: self.item.param.update(
            value=item_functions.Item(self.data_loader.value, self.data_and_probabilities.value, event.new,
                                      self.column_grouping.combined_columns)), parameter_names=['value'], onlychanged=False)

        self.column_grouping.param.watch(self.column_grouping_changed, parameter_names=['combined_columns'], onlychanged=False)

        # clustered data
        self.clustered_data = pn.widgets.LiteralInput(value=self._update_clustered_data())
        self.clustered_data.param.watch(lambda event: self.render_plot.param.update(value=self.update_render_plot_self()), parameter_names=['value'], onlychanged=False)
        self.cur_feature.param.watch(self.update_clustered_data, parameter_names=['value'], onlychanged=False)
        self.item_index.param.watch(self.update_clustered_data, parameter_names=['value'], onlychanged=False)
        self.cluster_type.param.watch(self.update_clustered_data, parameter_names=['value'], onlychanged=False)

        #render
        self.render_plot = pn.widgets.LiteralInput(value=self.update_render_plot_self())
        self.graph_type.param.watch(lambda event: self.render_plot.param.update(value=self.update_render_plot_self()), parameter_names=['value'], onlychanged=False)
        self.render_plot_view = pn.bind(lambda x: x, self.render_plot)

    def prediction_string(self):
        return pn.bind(lambda x: x.item.value.prediction_string(), self)

    def column_grouping_changed(self, event):
        if self.active:
            self.item.param.update(value=item_functions.Item(self.data_loader.value, self.data_and_probabilities.value,
                                                            self.item_index.value, self.column_grouping.combined_columns))


    def update_data(self, event):
        self.active = False
        loader = data_loader.DataLoader(self.file.value, self.nn_file.value)
        data_and_probabilities = feature.combine_data_and_results(loader)
        all_selected_cols = column_functions.return_col(loader.columns[0])
        cur_feature = all_selected_cols[0]
        item = item_functions.Item(loader, data_and_probabilities, self.item_index.value, [])
        clustered_data = similarity.get_clustering(self.cluster_type.value, data_and_probabilities, all_selected_cols,
                                                    cur_feature, item.prediction, self.item_index.value, exclude_col=False)

        self.data_loader.param.update(value=loader)
        self.data_and_probabilities.param.update(
            value=data_and_probabilities)

        self.col.param.update(options=loader.columns)
        self.all_selected_cols.param.update(value=all_selected_cols)

        self.column_grouping.init_groups(loader.columns)

        self.item.param.update(value=item)

        self.clustered_data.param.update(value=clustered_data)

        self.render_plot.param.update(value=self.update_render_plot(self.graph_type.value, all_selected_cols,
                                        clustered_data, cur_feature, item, self.item_index.value, self.chart_type.value))

        self.cur_feature.param.update(options=self.all_selected_cols.value, value=cur_feature)

        self.active = True



    def _update_clustered_data(self):
        return similarity.get_clustering( self.cluster_type.value, self.data_and_probabilities.value,
                       self.all_selected_cols.value,
                       self.cur_feature.value, self.item.value.prediction, self.item_index.value, exclude_col=False)

    def update_clustered_data(self, event):
        if self.active:
            self.clustered_data.param.update(
                value=self._update_clustered_data())

    def get_all_data(self):
        return pn.bind(data_loader.load_data, self.file.value, self.data_loader.value.nn)

    def get_data(self):
        return pn.bind(lambda data: data[0:200], self.get_all_data())

    def get_file_widgets(self):
        return pn.Row(self.file, self.nn_file, self.calculate, self.item_index).servable()

    def get_customization_widgets(self):
        return pn.Row(self.cluster_type, self.graph_type, self.chart_type).servable()

    def get_row_widgets(self):
        return self.column_grouping.row.servable()

    def update_render_plot(self, graph_type, all_selected_cols, clustered_data, cur_feature, item, item_index, chart_type):
        if graph_type == 'Cluster':
            return cluster_bar_plot(clustered_data, item, item_index)
        elif graph_type == 'Dependency':
            dep_plot = dependency_scatterplot(clustered_data, cur_feature, all_selected_cols,
                               item.prediction, item_index, chart_type,
                               item.prob_wo_selected_cols)
            return pn.Column(dep_plot, all_selected_cols[0])
        else:
            return parallel_plot(clustered_data, cur_feature, all_selected_cols,
                           item.prediction, item.data, chart_type)

    def update_render_plot_self(self):
        return self.update_render_plot(self.graph_type.value, self.all_selected_cols.value,
                                self.clustered_data.value, self.cur_feature.value, self.item.value,
                                self.item_index.value,
                                self.chart_type.value)