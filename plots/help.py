import panel as pn
import param
from panel.viewable import Viewer


class Help(Viewer):
    """
    Class to display the help text
    """

    info_text = param.ClassSelector(class_=pn.pane.Markdown)
    close_button = param.ClassSelector(class_=pn.widgets.ButtonIcon)
    never_show = param.Boolean()
    show = param.Boolean()

    def __init__(self, **params):
        super().__init__(**params)

        self.str = "**The purple line represents instances similar in the selected features (*neighborhood*)**"
        self.str_here = " - here: instances with a similar value of X"
        self.info_text = pn.pane.Markdown(self.str, sizing_mode='stretch_width', styles={'font-size': '11pt', 'color': '#A336B0'})
        self.close_button = pn.widgets.ButtonIcon(icon="x")
        self.never_show = False
        self.close_button.on_click(self.close)
        self.show = False

    @param.depends('show')
    def __panel__(self):
        if self.show and not self.never_show:
            obj = pn.Card(pn.Row(self.info_text, self.close_button),
                       styles=dict(margin='auto', width='100%', background='white', color="#A336B0",
                                   border='1px solid #A336B0', margin_bottom='5px'),
                       hide_header=True, sizing_mode='stretch_width')
        else:
            obj = None

        return pn.Row(obj, styles=dict(width='100%'), sizing_mode='stretch_width')

    def close(self, event):
        self.never_show = True
        self.show = False

    def update(self, all_selected_cols, item):
        self.show = len(all_selected_cols) > 1
        if len(all_selected_cols) > 1:
            str_here = " - here: instances with "
            for col in all_selected_cols[1:]:
                item_value = item.data_raw[col].values[0]
                str_here += f"{col} ~ {item_value}, "
            str_here = str_here[:-2]
            self.str_here = str_here
            self.info_text.object = self.str + "\n" + self.str_here