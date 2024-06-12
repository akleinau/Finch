import panel as pn
import param
from plots.cluster_bar_plot import cluster_bar_plot
from plots.dependency_plot import dependency_scatterplot
from plots.parallel_plot import parallel_plot
from plots.similar_bar_plot import similar_bar_plot


class RenderPlot(param.Parameterized):
    active_tab = param.Integer(default=0)

    def __init__(self, graph_type, all_selected_cols, clustered_data, cur_feature, item, item_index, chart_type,
                 predict_class, predict_label, active_tab=0, **params):
        super().__init__(**params)
        self.plot = self.render_plot_tabs(all_selected_cols, clustered_data, cur_feature, item, item_index,
                                         chart_type, predict_class, predict_label, active_tab)

    def get_render_plot(self, graph_type, all_selected_cols, clustered_data, cur_feature, item, item_index,
                        chart_type, predict_class, predict_label):
        if graph_type == 'Cluster':
            plot = cluster_bar_plot(clustered_data, item, all_selected_cols, predict_class, predict_label)
            plot = add_style(plot)
            return plot
        if graph_type == 'Similar':
            plot = similar_bar_plot(clustered_data, item, all_selected_cols, predict_class, predict_label)
            plot = add_style(plot)
            return plot
        elif graph_type == 'Dependency':
            dep_plot = dependency_scatterplot(clustered_data, cur_feature.value, all_selected_cols,
                                              item, chart_type.value)
            dep_plot = add_style(dep_plot)
            return pn.Column(dep_plot, cur_feature, chart_type)
        else:
            plot = parallel_plot(clustered_data, cur_feature.value, all_selected_cols,
                                 item.prediction, item.data, chart_type)
            plot = add_style(plot)
            return plot

    def render_plot_tabs(self, all_selected_cols, clustered_data, cur_feature, item, item_index,
                        chart_type, predict_class, predict_label, active_tab):
        p1 = self.get_render_plot('Cluster', all_selected_cols, clustered_data, cur_feature, item, item_index,
                                         chart_type, predict_class, predict_label)
        p2 = self.get_render_plot('Dependency', all_selected_cols, clustered_data, cur_feature, item, item_index,
                                            chart_type, predict_class, predict_label)
        p3 = self.get_render_plot('Parallel', all_selected_cols, clustered_data, cur_feature, item, item_index,
                                            chart_type, predict_class, predict_label)
        p4 = self.get_render_plot('Similar', all_selected_cols, clustered_data, cur_feature, item, item_index,
                                            chart_type, predict_class, predict_label)
        return pn.Tabs(('Cluster', p1), ('Dependency', p2), ('Parallel', p3), ('Similar', p4), dynamic=True,
                       active=active_tab)

def add_style(plot):
    plot.title.text_font_size = '20px'
    plot.title.align = 'center'
    plot.xaxis.major_label_text_font_size = '14px'
    plot.yaxis.major_label_text_font_size = '14px'
    plot.xaxis.axis_label_text_font_size = '14px'
    if len(plot.legend) >0:
        plot.legend.label_text_font_size = '14px'
    if len(plot.hover) > 0:
        # set the font size of the hover tooltip. I don't think this actually works?
        for h in plot.hover:
            h.tooltips.text_font_size = '14px'

    plot.grid.grid_line_color = "black"
    plot.grid.grid_line_alpha = 0.0

    return plot