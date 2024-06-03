import param
import panel as pn
import calculations.data_loader as data_loader
import calculations.item_functions as item_functions
import calculations.column_functions as column_functions
import calculations.clusters as clusters
import plots.render_plot as render_plot


class DataStore(param.Parameterized):
    item = param.ClassSelector(class_=item_functions.Item)
    columnGrouping = param.ClassSelector(class_=column_functions.ColumnGrouping)
    data_loader = param.ClassSelector(class_=data_loader.DataLoader)
    all_selected_cols = param.List()
    clustering = param.ClassSelector(class_=clusters.Clustering)
    render_plot = param.ClassSelector(class_=render_plot.RenderPlot)

    def __init__(self, **params):
        super().__init__(**params)
        self.active = True
        self.file = pn.widgets.FileInput(accept='.csv', name='Upload data')
        self.nn_file = pn.widgets.FileInput(accept='.pkl', name='Upload neural network')
        self.calculate = pn.widgets.Button(name='Calculate', button_type='primary')
        self.calculate.on_click(self.update_data)
        self.data_loader = data_loader.DataLoader()
        self.item_index = pn.widgets.EditableIntSlider(name='item index', start=0, end=100, value=26)
        self.predict_class = pn.widgets.Select(name='prediction', options=list(self.data_loader.classes))
        self.predict_class_label = pn.widgets.TextInput(name='prediction', value=self.predict_class.value)
        self.predict_class.param.watch(lambda event: self.predict_class_label.param.update(value=event.new),
                                        parameter_names=['value'], onlychanged=False)

        # columns
        self.col = pn.widgets.Select(name='column', options=self.data_loader.columns)
        self.all_selected_cols = column_functions.return_col(self.col.value)
        self.col.param.watch(
            lambda event: self.param.update(all_selected_cols=column_functions.return_col(event.new)),
            parameter_names=['value'], onlychanged=False)

        # groups
        self.cur_feature = pn.widgets.Select(name='', options=self.all_selected_cols,
                                             value=self.all_selected_cols[0], align='center')
        self.param.watch(lambda event: self.cur_feature.param.update(options=event.new),
                         parameter_names=['all_selected_cols'], onlychanged=False)
        self.column_grouping = column_functions.ColumnGrouping(self.data_loader.columns)
        self.column_grouping.param.watch(self.column_grouping_changed, parameter_names=['combined_columns'],
                                         onlychanged=False)

        # customization widgets
        self.cluster_type = pn.widgets.Select(name='cluster_type', options=['Relative Decision Tree', 'Decision Tree'],
                                              value='Relative Decision Tree')
        self.chart_type = pn.widgets.MultiChoice(name='chart_type', options=['scatter', 'line', 'band', 'contour'],
                                                 value=['line'])

        self.graph_type = pn.widgets.Select(name='graph_type', options=['Cluster', 'Dependency', 'Parallel'],
                                            value='Cluster')
        self.num_leafs = pn.widgets.EditableIntSlider(name='num_leafs', start=1, end=15, value=3)

        # item
        self.item = self._update_item_self()
        self.item_index.param.watch(self.update_item_self, parameter_names=['value'],
                                    onlychanged=False)
        self.predict_class_label.param.watch(self.update_item_self, parameter_names=['value'],
                                       onlychanged=False)

        # clustered data
        self.clustering = self._update_clustered_data()
        self.cur_feature.param.watch(self.update_clustered_data, parameter_names=['value'], onlychanged=False)
        self.item_index.param.watch(self.update_clustered_data, parameter_names=['value'], onlychanged=False)
        self.cluster_type.param.watch(self.update_clustered_data, parameter_names=['value'], onlychanged=False)
        self.num_leafs.param.watch(self.update_clustered_data, parameter_names=['value'], onlychanged=False)
        self.predict_class.param.watch(self.update_clustered_data, parameter_names=['value'], onlychanged=False)

        # render
        self.render_plot = self.update_render_plot()
        self.param.watch(
            lambda event: self.param.update(render_plot=self.update_render_plot()),
            parameter_names=['clustering'], onlychanged=False)
        self.graph_type.param.watch(lambda event: self.param.update(render_plot=self.update_render_plot()),
                                    parameter_names=['value'], onlychanged=False)
        self.chart_type.param.watch(lambda event: self.param.update(render_plot=self.update_render_plot()),
                                    parameter_names=['value'], onlychanged=False)

    def prediction_string(self):
        return pn.bind(lambda x: x.item.prediction_string(), self)

    def column_grouping_changed(self, event):
        if self.active:
            self.param.update(item=self.update_item_self())

    def update_data(self, event):
        self.active = False
        loader = data_loader.DataLoader(self.file.value, self.nn_file.value)
        predict_class = loader.classes[0]
        all_selected_cols = column_functions.return_col(loader.columns[0])
        cur_feature = all_selected_cols[0]
        item = item_functions.Item(loader, loader.data_and_probabilities, self.item_index.value, predict_class, predict_class, [])
        clustering = clusters.Clustering(self.cluster_type.value, loader.data_and_probabilities, all_selected_cols,
                                           cur_feature, item.prediction, self.item_index.value,
                                           exclude_col=False)
        self.predict_class.param.update(options=loader.classes, value=predict_class)

        self.param.update(data_loader=loader, item=item, clustering=clustering, all_selected_cols=all_selected_cols,
                          render_plot=render_plot.RenderPlot(self.graph_type.value, all_selected_cols,
                                                             clustering.data, cur_feature, item,
                                                             self.item_index.value, self.chart_type.value, self.predict_class.value))

        self.col.param.update(options=loader.columns)

        self.column_grouping.init_groups(loader.columns)

        self.cur_feature.param.update(options=self.all_selected_cols, value=cur_feature)

        self.active = True

    def _update_clustered_data(self):
        return clusters.Clustering(self.cluster_type.value, self.data_loader.data_and_probabilities,
                                     self.all_selected_cols,
                                     self.cur_feature.value, self.predict_class.value, self.item_index.value,
                                     exclude_col=False, num_leafs=self.num_leafs.value)

    def update_clustered_data(self, event):
        if self.active:
            self.param.update(
                clustering=self._update_clustered_data())

    def get_all_data(self):
        return pn.bind(data_loader.load_data, self.file.value, self.data_loader.nn)

    def get_file_widgets(self):
        return pn.Row(self.file, self.nn_file, self.calculate, self.item_index, self.predict_class, self.predict_class_label).servable()

    def get_customization_widgets(self):
        return pn.Row(self.cluster_type, self.graph_type, self.chart_type, self.cur_feature, self.num_leafs).servable()

    def get_row_widgets(self):
        return self.column_grouping.row.servable()

    def update_render_plot(self):
        return render_plot.RenderPlot(self.graph_type.value, self.all_selected_cols,
                                      self.clustering.data, self.cur_feature, self.item,
                                      self.item_index.value,
                                      self.chart_type.value, self.predict_class.value)

    def _update_item_self(self):
            return item_functions.Item(self.data_loader, self.data_loader.data_and_probabilities,
                                         self.item_index.value, self.predict_class.value, self.predict_class_label.value,
                                         self.column_grouping.combined_columns)

    def update_item_self(self, event):
        if self.active:
            self.param.update(item=self._update_item_self())