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
    logo="https://github.com/akleinau/Finch/blob/8e7f13c04c7c08c36fb823dfa57ffd057a77ce62/logo.png?raw=true",
    header_background=color,
    sidebar_width=300,
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
subset_widgets = pn.bind(lambda e: e, ds.param.subset_widgets)
item_data = pn.bind(lambda e: e, ds.param.item)
overview_plot = pn.bind(lambda e: e, ds.param.overview_plot)
floatpanel = pn.bind(lambda e: e, ds.param.add_feature_panel)
help = pn.bind(lambda e: e, ds.param.help_pane)


# main content
template.main.append(pn.Column(
    pn.Row(ds.feature_iter, styles=dict(margin="auto")),
    pn.Row(pn.bind(lambda a: a.prediction_string(), ds.param.item)),
    pn.Row(help, styles=dict(width='100%')),
    overview_plot,
    pn.Row(ds.render_plot,
           pn.Column(sim_plot, subset_widgets, styles=dict(margin_left='20px', align='center')),
           styles=dict(margin='auto', width='100%', padding='10px')),
    floatpanel,
    styles=dict(width='100%', margin='auto')
))

template.servable()
