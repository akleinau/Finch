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

style_truth = f"""    
    :host(.solid) .bk-btn-group .bk-btn.bk-btn-default.bk-active {{
        border: 2px dotted #cc98e6;
        box-shadow: none;
    }}
"""

style_additive = f"""    
    :host(.solid) .bk-btn-group .bk-btn.bk-btn-default.bk-active {{
        border: 2px dashed #4B0082;
        box-shadow: none;
    }}
"""

style_sidebar = f"""
    :host(.accordion) {{
        box-shadow: none;
        outline: none;
        border: none;
        width: 100% !important;
    }}
    
    .accordion {{
        box-shadow: none;
        outline: none;
        border: none;
        width: 100% !important;
        margin: 0;
        border-top: 1px solid darkgrey;
        background-color: #EEEEEE
    }}
"""

style_input = f"""
    .bk-input {{
        background-color: white
    }}
"""

style_icon = f"""
    :host(.outline) .bk-btn-group .bk-btn-default {{
        border: none;
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
