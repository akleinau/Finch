from bokeh.plotting import figure
from bokeh.transform import jitter


def similar_plot(data_loader, item):
    data = data_loader.data[0:200]

    # for each column, create a bokeh plot with the distribution of the data

    y_range = list(data.columns)

    #first, pivot all columns to long format
    data = data.melt()

    # create a figure
    plot = figure(title="Similar items", y_range=y_range, width=400)
    plot.scatter(x='value', y=jitter('variable', width=0.6, range=plot.y_range), alpha=0.1, source=data)

    return plot



