import panel as pn
import param
from panel.viewable import Viewer


class Help(Viewer):
    """
    Class to display the help text
    """

    info_text = param.ClassSelector(class_=pn.pane.Markdown)
    close_button = param.ClassSelector(class_=pn.widgets.ButtonIcon)

    def __init__(self, **params):
        super().__init__(**params)

        text = "The purple line represents the mean prediction for similar items in the chosen features"
        self.info_text = pn.pane.Markdown(text, sizing_mode='stretch_width', styles={'font-size': '11pt'})
        self.close_button = pn.widgets.ButtonIcon(icon="x")


    def __panel__(self):
        return pn.Card(pn.Row(self.info_text, self.close_button),
                       styles=dict(margin='auto', width='100%', background='lightpink'),
                       hide_header=True, sizing_mode='stretch_width')