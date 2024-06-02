from plots.cluster_bar_plot import cluster_bar_plot
from plots.dependency_plot import dependency_scatterplot
from plots.parallel_plot import parallel_plot
import panel as pn


class RenderPlot:

    def __init__(self, graph_type, all_selected_cols, clustered_data, cur_feature, item, item_index, chart_type):
        self.plot = self.get_render_plot(graph_type, all_selected_cols, clustered_data, cur_feature, item, item_index,
                                         chart_type)

    def get_render_plot(self, graph_type, all_selected_cols, clustered_data, cur_feature, item, item_index,
                        chart_type):
        if graph_type == 'Cluster':
            return cluster_bar_plot(clustered_data, item, item_index)
        elif graph_type == 'Dependency':
            dep_plot = dependency_scatterplot(clustered_data, cur_feature.value, all_selected_cols,
                                              item.prediction, item_index, chart_type,
                                              item.prob_wo_selected_cols)
            return pn.Column(dep_plot, cur_feature)
        else:
            return parallel_plot(clustered_data, cur_feature.value, all_selected_cols,
                                 item.prediction, item.data, chart_type)
