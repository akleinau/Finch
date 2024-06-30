import panel as pn
from plots.tornado_plot import shap_tornado_plot
from plots.similar_plot import similar_plot
import DataStore as DataStore

pn.extension('floatpanel')

ds = DataStore.DataStore()

# create widgets
ds.get_file_widgets()
ds.get_title_widgets()
ds.get_item_widgets()

# create all the widgets and variables needed for the column group selection
#pn.panel("<br>").servable()
#pn.panel("### Column groups:").servable()
ds.get_row_widgets()

# shap plot
#shap_plot = pn.bind(shap_tornado_plot, ds.param.item, [ds.col])  # col is wrapped to be passed as reference

# remaining layout
pn.Row(pn.bind(lambda a: a.prediction_string(), ds.param.item)).servable()

render_plot = pn.bind(lambda e: e.plot, ds.param.render_plot)
sim_plot = pn.bind(lambda e: e.plot, ds.param.similar_plot)
item_data = pn.bind(lambda e: e.data_reduced, ds.param.item)

pn.Row(item_data, sim_plot, render_plot, styles=dict(margin='auto')).servable()

#ds.get_customization_widgets()

pn.Row(ds.cur_feature, styles=dict(visibility='hidden')).servable() # necessary for the column selection to work