import panel as pn
import DataStore as DataStore
from plots.styling import color

pn.extension(global_css=[f':root {{ --design-primary-color: {color}; }}'])

ds = DataStore.DataStore()

template = pn.template.MaterialTemplate(
    title="Feature",
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

# main content
template.main.append(pn.Column(
    pn.Row(ds.feature_iter, styles=dict(margin="auto")),
    pn.Row(pn.bind(lambda a: a.prediction_string(), ds.param.item)),
    pn.Row(item_data, ds.render_plot, sim_plot)
))

template.servable()
