import panel as pn
import param
from panel.viewable import Viewer

class FeatureIter(Viewer):
    """
    Class to coordinate the selection of features
    """


    all_selected_cols = param.List()
    all_selected_cols_final = param.List()
    show_process = param.Boolean()
    widgets = param.ClassSelector(class_=pn.Row)

    def __init__(self, columns: list, **params):
        super().__init__(**params)

        self.columns = columns
        self.active = True

        self.col_widget = pn.widgets.Select(name='add', options=["", *columns], value="")
        self.col_widget.param.watch(self.add_col, parameter_names=['value'], onlychanged=False)

        self.col_display = pn.widgets.RadioButtonGroup(button_type='primary', button_style='outline', align="center")
        self.col_display.param.watch(lambda event: self.col_selected(event.new), parameter_names=['value'], onlychanged=False)

        self.col_type = 'singular'
        self.all_selected_cols = []
        self.all_selected_cols_final = []
        self.col_whitelist = columns

        # create minus button
        self.minus_button = pn.widgets.ButtonIcon(icon='trash', size="2em", align="center")
        self.minus_button.on_click(self.remove_col)
        self.minus_button.visible = False

        # create final toggle
        self.show_process = True
        self.final_toggle = pn.widgets.Toggle(name='Final', value=False, align="center")
        self.final_toggle.param.watch(self.final_toggle_changed, parameter_names=['value'], onlychanged=False)

        self.widgets = pn.Row(self.col_display, self.minus_button, self.col_widget,  self.final_toggle)

    @param.depends('widgets')
    def __panel__(self) -> pn.Row:
        return self.widgets

    def add_col(self, event):
        if event.new != "" and self.active:
            self.all_selected_cols_final.append(event.new)
            self.update_widgets()

    def col_selected(self, col: str):
        """
        handles the selection of a feature for display

        :param col: str
        """

        if self.active:
            self.active = False
            if col is not None:
                index = self.all_selected_cols_final.index(col)
                self.all_selected_cols = self.all_selected_cols_final[:index+1]

            self.show_process = True
            self.final_toggle.value = False
            self.active = True

    def update_widgets(self):
        """
        updates all widgets based on the current state

        """

        self.active = False
        self.col_whitelist = [col for col in self.columns if col not in self.all_selected_cols_final]
        self.col_widget.options = ["", *self.col_whitelist]
        self.col_widget.value = ""
        self.minus_button.visible = len(self.all_selected_cols_final) > 0
        self.widgets = pn.Row(self.col_display, self.minus_button, self.col_widget, self.final_toggle)

        self.active = True

        # this intentionally triggers col_selected
        if len(self.all_selected_cols_final) > 0:
            self.col_display.visible = True
            self.col_display.options = self.all_selected_cols_final
            self.col_display.value = self.all_selected_cols_final[-1]

        else:
            self.col_display.visible = False
            self.all_selected_cols = []



    def remove_col(self, event):
        if len(self.all_selected_cols_final) > 0:
            self.all_selected_cols_final = self.all_selected_cols_final[:-1]
            self.update_widgets()

    def load_new_columns(self, columns: list):
        """
        loads new columns when a new data set is loaded

        :param columns: list
        """
        self.columns = columns
        self.all_selected_cols = []
        self.all_selected_cols_final = []
        self.update_widgets()

    def final_toggle_changed(self, event):
        if self.active:
            self.active = False
            self.show_process = not event.new
            self.all_selected_cols = self.all_selected_cols_final
            self.col_display.value = None
            self.active = True


def get_first(event):
    if len(event) > 0:
        return event[0]
    return None

