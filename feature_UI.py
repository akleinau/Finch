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

# create all the widgets and variables needed for the column group selection
pn.panel("<br>").servable()
pn.panel("### Grouped columns:").servable()

# shap plot
shap_plot = pn.bind(shap_tornado_plot, ds.get_item_shap(), [ds.get_col()])  # col is wrapped to be passed as reference

# dependency plot

dep_plot = pn.bind(dependency_scatterplot, ds.get_clustered_data(), ds.get_cur_feature(), ds.get_all_selected_cols(),
                   ds.get_item_prediction(),ds.get_x(), ds.chart_type, ds.get_prob_wo_selected_cols())

# parallel plot
parallel_plot = pn.bind(parallel_plot, ds.get_clustered_data(), ds.get_cur_feature(), ds.get_all_selected_cols(),
                        ds.get_item_prediction(), ds.get_item_data(), ds.chart_type)

# cluster bar plot
cluster_plot = pn.bind(cluster_bar_plot, ds.get_clustered_data(), ds.get_cur_feature(), ds.get_all_selected_cols(),
                       ds.get_item_prediction(),ds.get_x(), ds.chart_type, ds.get_prob_wo_selected_cols())

# remaining layout
pn.pane.Str(ds.get_prob_data(), sizing_mode="stretch_width", align="center",
            styles={"font-size": "20px", "text-align": "center"}).servable()


def render_plot(graph_type, dep_plot, cluster_plot, parallel_plot, all_selected_cols):
    if graph_type == 'Cluster':
        return cluster_plot
    elif graph_type == 'Dependency':
        return pn.Column(dep_plot, all_selected_cols)
    else:
        return parallel_plot

rendered_plot = pn.bind(render_plot, ds.graph_type, dep_plot, cluster_plot, parallel_plot, [ds.get_cur_feature()])
pn.Row(ds.get_item_data(), shap_plot, rendered_plot).servable()
#ds.get_cur_feature().servable()

