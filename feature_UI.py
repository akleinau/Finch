import panel as pn
import DataStore as DataStore
from plots.styling import color

pn.extension('floatpanel', global_css=[f':root {{ --design-primary-color: {color}; }}'])

ds = DataStore.DataStore()

template = pn.template.MaterialTemplate(
    title="FINCH - Feature Interaction Charts",
    header_background =color,
)

item_widgets = pn.bind(lambda e: e, ds.param.item_widgets)

# sidebar content
template.sidebar.append(pn.Column("# Data set", ds.get_file_widgets(), pn.layout.Spacer(),
                                  "# Target", ds.get_title_widgets(), pn.layout.Spacer(),
                                  "# Item", item_widgets, styles=dict(margin='auto')))

# somehow necessary for reactivity
render_plot = pn.bind(lambda e: e, ds.param.render_plot)
sim_plot = pn.bind(lambda e: e, ds.param.similar_plot)
item_data = pn.bind(lambda e: e, ds.param.item)
tornado_plot_single = pn.bind(lambda e: e.get_panel_single, ds.param.tornado_plot)
tornado_plot_overview = pn.bind(lambda e: e, ds.param.tornado_plot)
ranked_buttons = pn.bind(lambda e: e.ranked_buttons, ds.param.tornado_plot)
floatpanel = pn.bind(lambda e: e, ds.param.add_feature_panel)

# main content
template.main.append(pn.Column(
    pn.Row(ds.feature_iter, styles=dict(margin="auto")),
    pn.Row(pn.bind(lambda a: a.prediction_string(), ds.param.item)),
    pn.Row(ds.render_plot,
           pn.Column(sim_plot, tornado_plot_single, styles=dict(margin_left='20px')), styles=dict(margin='auto')),
    tornado_plot_overview,
    floatpanel
))

template.servable()
