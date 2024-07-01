import panel as pn
from plots.tornado_plot import shap_tornado_plot
from plots.similar_plot import similar_plot
import DataStore as DataStore

pn.extension()

ds = DataStore.DataStore()

template = pn.template.MaterialTemplate(
    title="Feature",
)

# create widgets
template.sidebar.append(pn.Column("# Data set", ds.get_file_widgets(), pn.layout.Spacer(),
                                  "# Target", ds.get_title_widgets(), pn.layout.Spacer(),
                                  "# Item", ds.get_item_widgets(), styles=dict(margin='auto')))

# create all the widgets and variables needed for the column group selection
#pn.panel("<br>").servable()
#pn.panel("### Column groups:").servable()

# shap plot
#shap_plot = pn.bind(shap_tornado_plot, ds.param.item, [ds.col])  # col is wrapped to be passed as reference

# remaining layout

render_plot = pn.bind(lambda e: e.plot, ds.param.render_plot)
sim_plot = pn.bind(lambda e: e.plot, ds.param.similar_plot)
item_data = pn.bind(lambda e: e.data_reduced, ds.param.item)

template.main.append(pn.Column(
    ds.get_row_widgets(),
    pn.Row(pn.bind(lambda a: a.prediction_string(), ds.param.item)),
    pn.Row(item_data, render_plot, sim_plot, styles=dict(margin='auto'))
))

#ds.get_customization_widgets()

#template.main.append(pn.Row(ds.cur_feature, styles=dict(visibility='hidden')))# necessary for the column selection to work

template.servable()