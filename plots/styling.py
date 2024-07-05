from bokeh.plotting import figure

color = "#381eaa"

style_options = f"""    
    :host(.outline) .bk-btn-group .bk-btn-default.bk-active {{
        background-color: {color};
        border-color: {color};
        color: white;
    }}
    
    :host(.outline) .bk-btn-group .bk-btn-default {{
        border-color: {color}
    }}
    
    :host(.solid) .bk-btn-group .bk-btn-default.bk-active {{
        background-color: {color};
        border-color: {color};
        color: white;
    }}
    
"""

style_button = f"""
    :host(.solid) .bk-btn.bk-btn-default {{
        background-color: {color};
        color: white;
    }}
"""

def add_style(plot: figure) -> figure:
    """
    add global styling to a plot

    :param plot: bokeh.plotting.figure
    :return:bokeh.plotting.figure
    """

    plot.xaxis.major_label_text_font_size = '14px'
    plot.yaxis.major_label_text_font_size = '14px'
    plot.xaxis.axis_label_text_font_size = '14px'
    if len(plot.legend) > 0:
        plot.legend.label_text_font_size = '14px'
    if len(plot.hover) > 0:
        # set the font size of the hover tooltip. I don't think this actually works?
        for h in plot.hover:
            h.tooltips.text_font_size = '14px'

    plot.grid.grid_line_color = "black"
    plot.grid.grid_line_alpha = 0.0

    return plot
