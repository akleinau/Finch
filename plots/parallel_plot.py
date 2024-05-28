from bokeh.plotting import figure
from bokeh.models import Band, ColumnDataSource, HoverTool, Legend, LegendItem
import numpy as np
from scipy.stats import gaussian_kde
import pandas as pd
import bokeh.colors

def parallel_plot(data, col, all_selected_cols, prob, item, chart_type):
    #normalize data of all_selected_cols

    chart3 = figure(title="example", x_range=all_selected_cols, width=800)
    chart3.grid.level = "overlay"
    chart3.grid.grid_line_color = "black"
    chart3.grid.grid_line_alpha = 0.05
    chart3.add_layout(Legend(), 'right')


    means = {}
    stds = {}
    for col in all_selected_cols:
        means[col] = data[col].mean()
        stds[col] = data[col].std()

        chart3.line(x=[col, col], y=[0,1], line_width=1, color='black', alpha=0.5)


    chart3.line(x='feature', y='value', source=item, color='black')
    chart3.scatter(x='feature', y='value', source=item, color='black', size=5)

    return chart3