import panel as pn
import functions as feature
import calculations.item as item_functions
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
shap_plot = pn.bind(shap_tornado_plot, ds.param.item, [ds.col])  # col is wrapped to be passed as reference

# remaining layout
pn.Row(pn.bind(lambda a: a.prediction_string(), ds.param.item)).servable()

table = pn.bind(lambda a: a.table(), ds.param.item)

render_plot = pn.bind(lambda e: e.plot, ds.param.render_plot)

pn.Row(table, shap_plot, render_plot).servable()
