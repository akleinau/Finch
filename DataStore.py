import panel as pn
import param

import calculations.data_loader as data_loader
import calculations.feature_iter as feature_iter
import calculations.item_functions as item_functions
import plots.dependency_plot as dependency_plot
import plots.help as help_plot
import plots.overview_plot as overview_plot
import plots.similar_plot as similar_plot
import plots.tornado_plot as tornado_plot
from plots.styling import style_button, style_options, style_input


class DataStore(param.Parameterized):
    """
    main data store that manages everything
    """

    item = param.ClassSelector(class_=item_functions.Item)
    data_loader = param.ClassSelector(class_=data_loader.DataLoader)
    render_plot = param.ClassSelector(class_=dependency_plot.DependencyPlot)
    similar_plot = param.ClassSelector(class_=similar_plot.SimilarPlot)
    feature_iter = param.ClassSelector(class_=feature_iter.FeatureIter)
    item_widgets = param.ClassSelector(class_=pn.Column)
    tornado_plot = param.ClassSelector(class_=tornado_plot.TornadoPlot)
    add_feature_panel = param.ClassSelector(class_=pn.layout.FloatPanel)
    help_pane = param.ClassSelector(class_=help_plot.Help)
    overview_plot = param.ClassSelector(class_=overview_plot.OverviewPlot)

    def __init__(self, **params):
        super().__init__(**params)
        self.active = True
        white = {'background': 'white'}
        self.file = pn.widgets.FileInput(accept='.csv', name='Upload data', width=200, styles=white)
        self.nn_file = pn.widgets.FileInput(accept='.pkl', name='Upload neural network', width=200, styles=white)
        self.truth_file = pn.widgets.FileInput(accept='.csv', name='Upload truth', width=200, styles=white)
        self.calculate = pn.widgets.Button(name='Calculate', styles=dict(margin='auto'), stylesheets=[style_button])
        self.calculate.on_click(self.update_data)
        self.data_loader = data_loader.DataLoader()

        # item
        self.item_type = pn.widgets.RadioButtonGroup(name='item type', options=['predefined', 'custom'],
                                                     value='predefined', button_style='outline',
                                                     stylesheets=[style_options])
        self.item_index = pn.widgets.EditableIntSlider(name='Instance index', start=0, end=100, value=36, width=250)
        self.item_index.param.watch(lambda event: self.set_item_widgets(),
                                    parameter_names=['value_throttled'], onlychanged=False)
        self.item_custom_content = pn.Column()

        # predict class
        self.predict_class = pn.widgets.Select(name='prediction', options=list(self.data_loader.classes),
                                               value=self.data_loader.classes[-1], width=250, stylesheets=[style_input])
        self.predict_class_label = pn.widgets.TextInput(name='prediction label', value=self.predict_class.value,
                                                        width=250, stylesheets=[style_input])
        self.predict_class.param.watch(lambda event: self.predict_class_label.param.update(value=event.new),
                                       parameter_names=['value'], onlychanged=False)
        self.predict_class_label.param.watch(lambda event: self.set_item_widgets(),
                                             parameter_names=['value'], onlychanged=False)

        # columns
        self.feature_iter = feature_iter.FeatureIter(self.data_loader.columns)
        self.render_plot = dependency_plot.DependencyPlot()
        self.help_pane = help_plot.Help()
        self.overview_plot = overview_plot.OverviewPlot()

        # customization widgets
        self.cluster_type = pn.widgets.Select(name='cluster_type', options=['Relative Decision Tree', 'Decision Tree',
                                                                            'Similarity Decision Tree',
                                                                            'SimGroup Decision Tree'],
                                              value='Decision Tree')
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
        self.item_type.param.watch(self.update_item_self, parameter_names=['value'],
                                   onlychanged=False)
        self.item_widgets = self._set_item_widgets()
        self.param.watch(self.feature_iter.changed_item, parameter_names=['item'], onlychanged=False)
        self.init_item_custom_content()

        # render dependency plot
        self.feature_iter.param.watch(self.update_render_plot, parameter_names=['all_selected_cols', 'show_process'],
                                      onlychanged=False)
        self.param.watch(self.update_render_plot, parameter_names=['item'], onlychanged=False)
        self.predict_class.param.watch(self.update_render_plot, parameter_names=['value'], onlychanged=False)
        self.graph_type.param.watch(self.update_render_plot,
                                    parameter_names=['value'], onlychanged=False)
        self.chart_type.param.watch(self.update_render_plot,
                                    parameter_names=['value'], onlychanged=False)
        self.predict_class_label.param.watch(self.update_render_plot, parameter_names=['value'],
                                             onlychanged=False)

        # render similar plot
        self.update_similar_plot()
        self.feature_iter.param.watch(self.update_similar_plot,
                                      parameter_names=['all_selected_cols'], onlychanged=False)
        self.param.watch(self.update_similar_plot,
                         parameter_names=['item'], onlychanged=False)

        # help
        self.feature_iter.param.watch(self.update_help, parameter_names=['all_selected_cols'], onlychanged=False)
        self.param.watch(self.update_help,
                         parameter_names=['item'], onlychanged=False)

        # floatpanel
        self.add_feature_panel = None
        self.feature_iter.param.watch(self.set_feature_panel, parameter_names=['show_add'], onlychanged=False)

        # render tornado plot
        self.update_tornado_plot()
        self.feature_iter.param.watch(self.update_tornado_plot,
                                      parameter_names=['all_selected_cols'], onlychanged=False)
        self.param.watch(self.update_tornado_plot,
                         parameter_names=['item'], onlychanged=False)

        self.update_overview_plot()
        self.feature_iter.param.watch(self.update_overview_plot,
                                      parameter_names=['all_selected_cols'], onlychanged=False)
        self.param.watch(self.update_overview_plot,
                         parameter_names=['item'], onlychanged=False)

    def update_data(self, event):
        """
        updates everything when a new data set is loaded

        :param event
        """

        self.active = False
        loader = data_loader.DataLoader(self.file.value, self.nn_file.value, self.truth_file.value)
        predict_class = loader.classes[-1]
        item = item_functions.Item(loader, loader.data_and_probabilities, "predefined", self.item_index.value,
                                   pn.Column(),
                                   predict_class, predict_class)

        self.predict_class.param.update(options=loader.classes, value=predict_class)
        self.param.update(data_loader=loader, item=item)
        self.init_item_custom_content()
        self.item_widgets = self._set_item_widgets()
        self.active = True

        # intentionally trigger visualization updates, etc
        self.feature_iter.load_new_columns(loader.columns)

        self.update_tornado_plot()

    def init_item_custom_content(self):
        """
        initializes the custom content input fields
        """

        self.item_custom_content.clear()
        self.item_custom_content.append("(missing values will be imputed)")
        for col in self.data_loader.columns:
            widget = pn.widgets.LiteralInput(name=col, value=None, width=200, stylesheets=[style_input])
            widget.param.watch(self.update_item_self, parameter_names=['value'], onlychanged=False)
            self.item_custom_content.append(widget)

    def get_file_widgets(self) -> pn.Column:
        return pn.Column(pn.Row(
            pn.Column("Data*:", "Model*:", "Truth:"),
            pn.Column(self.file, self.nn_file, self.truth_file)),
            self.calculate,
            styles=dict(padding_bottom='10px', margin='0', align='end'))

    def get_title_widgets(self) -> pn.Column:
        return pn.Column(self.predict_class, self.predict_class_label, styles=dict(padding_top='10px'))

    def _set_item_widgets(self, data=None, y_class=None) -> pn.Column:
        # texts
        if data is None:
            data = self.data_loader.data_and_probabilities
        if y_class is None:
            y_class = self.predict_class.value
        str_dataset_nr = pn.pane.Markdown("Dataset size: " + str(data.shape[0]), height=10)
        mean = data[y_class].mean()
        str_mean_prediction = "Mean prediction: " + "{:.2f}".format(mean)

        # widgets
        second_item = pn.bind(
            lambda t: pn.Column(self.item_index,
                                self.item) if t == 'predefined' else self.item_custom_content if t == 'custom' else None,
            self.item_type)
        return pn.Column(str_dataset_nr, str_mean_prediction, self.item_type, second_item, pn.layout.Spacer(height=20))

    def set_item_widgets(self, data=None, y_class=None):
        if self.active:
            self.param.update(item_widgets=self._set_item_widgets(data, y_class))

    def update_render_plot(self, *params):
        if self.active:
            self.render_plot.update_plot(self.data_loader.data_and_probabilities, self.feature_iter.all_selected_cols,
                                         self.item,
                                         self.chart_type.value, self.data_loader,
                                         show_process=self.feature_iter.show_process,
                                         simple_next=self.feature_iter.simple_next)

    def update_similar_plot(self, *params):
        if self.active:
            self.param.update(
                similar_plot=similar_plot.SimilarPlot(self.data_loader, self.item, self.feature_iter.all_selected_cols))

    def update_tornado_plot(self, *params):
        if self.active:
            self.param.update(tornado_plot=tornado_plot.TornadoPlot(self.data_loader.data_and_probabilities, self.item,
                                                                    self.predict_class.value, self.data_loader.columns,
                                                                    self.feature_iter))

    def _update_item_self(self) -> item_functions.Item:
        return item_functions.Item(self.data_loader, self.data_loader.data_and_probabilities, self.item_type.value,
                                   self.item_index.value, self.item_custom_content,
                                   self.predict_class.value, self.predict_class_label.value)

    def update_item_self(self, *params):
        if self.active:
            self.param.update(item=self._update_item_self())

    def set_feature_panel(self, a):
        if a.new:
            self.add_feature_panel = pn.layout.FloatPanel(self.tornado_plot.panel_single, name='Add Feature', margin=20,
                                                          contained=False, height=800, status="normalized",
                                                          position="center")
        else:
            self.add_feature_panel = None

    def update_help(self, *params):
        self.help_pane.update(self.feature_iter.all_selected_cols, self.item)

    def update_overview_plot(self, *params):
        self.overview_plot.update(self.data_loader.data_and_probabilities, self.item, self.predict_class.value,
                                  self.data_loader.columns, self.feature_iter)
