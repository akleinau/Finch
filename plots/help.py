import panel as pn
import param
from panel.viewable import Viewer


class Help(Viewer):
    """
    Class to display the help text
    """

    neighborhood_text = param.ClassSelector(class_=pn.pane.Markdown)
    close_button_neighbor = param.ClassSelector(class_=pn.widgets.ButtonIcon)
    never_show_neighbor = param.Boolean()
    len_selected_cols = param.Integer()

    def __init__(self, **params):
        super().__init__(**params)

        self.str_neighborhood_start = ("**The purple line shows how instances in the *neighborhood* of the current instance behave** \n"
                    " - the neighborhood is defined based on the selected features and current instance \n")
        self.neighborhood_text = pn.pane.Markdown(self.str_neighborhood_start, sizing_mode='stretch_width', styles={'font-size': '11pt', 'color': '#A336B0'})

        self.str_base = ("**The grey line shows how the prediction changes based on the selected feature** \n"
                        " - centered around the mean prediction \n")
        self.base_text = pn.pane.Markdown(self.str_base, sizing_mode='stretch_width', styles={'font-size': '11pt', 'color': '#606060'})

        self.close_button_neighbor = pn.widgets.ButtonIcon(icon="x")
        self.never_show_neighbor = False
        self.close_button_neighbor.on_click(self.close_neighbor)

        self.close_button_base = pn.widgets.ButtonIcon(icon="x")
        self.never_show_base = False
        self.close_button_base.on_click(self.close_base)

        self.len_selected_cols = 0

    @param.depends('len_selected_cols')
    def __panel__(self):
        if self.len_selected_cols == 1 and not self.never_show_base:
            obj = pn.Card(pn.Row(self.base_text, self.close_button_base),
                          styles=dict(margin='auto', width='100%', background='#EEEEEE',
                                   border='1px solid darkgrey', margin_bottom='5px'),
                          hide_header=True, sizing_mode='stretch_width')

        elif self.len_selected_cols > 1 and not self.never_show_neighbor:
            obj = pn.Card(pn.Row(self.neighborhood_text, self.close_button_neighbor),
                          styles=dict(margin='auto', width='100%', background='#FFF2FF',
                                   border='1px solid #A336B0', margin_bottom='5px'),
                          hide_header=True, sizing_mode='stretch_width')
        else:
            obj = None

        return pn.Row(obj, styles=dict(width='100%'), sizing_mode='stretch_width')

    def close_neighbor(self, event):
        self.never_show_neighbor = True
        self.len_selected_cols = 0  # trigger update

    def close_base(self, event):
        self.never_show_base = True
        self.len_selected_cols = 0 #trigger update

    def update(self, all_selected_cols, item):
        if len(all_selected_cols) > 1:
            str_here = " - here: instances with "
            for col in all_selected_cols[1:]:
                item_value = item.data_raw[col].values[0]
                str_here += f"{col} ~ {item_value}, "
            str_here = str_here[:-2]
            self.neighborhood_text.object = self.str_neighborhood_start + "\n" + str_here

        self.len_selected_cols = len(all_selected_cols)
