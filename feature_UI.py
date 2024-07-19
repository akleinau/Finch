import panel as pn
import DataStore as DataStore
from plots.styling import color, style_sidebar

css = f"""
    :root {{
        --design-primary-color: {color};
    }}
    
    #sidebar {{
        background-color: #EEEEEE;
        padding-left: 0;
        padding-right: 0;
        margin-left: 0;
        margin-right: 0;
    }}
    
    #sidebar .mdc-list {{
        padding: 0 !important;
        margin-right: 0;
        margin-left: 0;
    }}
        
"""

pn.extension('floatpanel', global_css=[css])

ds = DataStore.DataStore()

template = pn.template.MaterialTemplate(
    title="FINCH - Feature Interaction Charts",
    header_background =color,
    sidebar_width=330,
)


item_widgets = pn.bind(lambda e: e, ds.param.item_widgets)

# sidebar content

s1 = ("Data Set", pn.Column(ds.get_file_widgets(), pn.layout.Spacer()))
s2 = ("Target", pn.Column(ds.get_title_widgets(), pn.layout.Spacer()))
s3 = ("Instance", pn.Column(item_widgets, pn.layout.Spacer()))
acc = pn.Accordion(s1, s2, s3, styles=dict(margin='0', width='100%', padding='0'), stylesheets=[style_sidebar],
                   header_background="lightgrey", active_header_background="lightgrey")
acc.active = [2]
acc.toggle = True

template.sidebar.append(acc)

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
