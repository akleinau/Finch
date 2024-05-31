import panel as pn
import functions as feature
import calculations.item_functions as item_functions
import calculations.column_functions as column_functions
import calculations.data_loader as data_loader
import calculations.similarity as similarity
from plots.dependency_plot import dependency_scatterplot
from plots.tornado_plot import shap_tornado_plot
from plots.parallel_plot import parallel_plot
from plots.cluster_bar_plot import cluster_bar_plot
from DataStore import DataStore

pn.extension()

ds = DataStore()

# create widgets
ds.get_file_widgets()
ds.get_customization_widgets()
ds.get_row_widgets()

# create all the widgets and variables needed for the column group selection
pn.panel("<br>").servable()
pn.panel("### Grouped columns:").servable()

# shap plot
shap_plot = pn.bind(shap_tornado_plot, ds.item_shap, [ds.col])  # col is wrapped to be passed as reference


# remaining layout
pn.pane.Str(ds.prob_data, sizing_mode="stretch_width", align="center",
            styles={"font-size": "20px", "text-align": "center"}).servable()


def render_plot(graph_type, all_selected_cols):
    if graph_type == 'Cluster':
        return pn.bind(cluster_bar_plot, ds.clustered_data_widget, ds.item_prediction, ds.x_widget)
    elif graph_type == 'Dependency':
        dep_plot = pn.bind(dependency_scatterplot, ds.clustered_data, ds.cur_feature_widget, ds.all_selected_cols,
                           ds.item_prediction, ds.x, ds.chart_type, ds.prob_wo_selected_cols)
        return pn.Column(dep_plot, all_selected_cols[0])
    else:
        return pn.bind(parallel_plot, ds.clustered_data, ds.cur_feature_widget, ds.all_selected_cols,
                        ds.item_prediction, ds.item_data, ds.chart_type)

rendered_plot = pn.bind(render_plot, ds.graph_type, [ds.cur_feature_widget])
pn.Row(ds.item_data, shap_plot, rendered_plot).servable()
ds.cur_feature_widget.servable()

