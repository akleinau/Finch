import panel as pn
import param
from panel.viewable import Viewer

class FeatureIter(Viewer):
    all_selected_cols = param.List()
    all_selected_cols_final = param.List()
    show_process = param.Boolean()
    widgets = param.ClassSelector(class_=pn.Row)

    def __init__(self, columns, **params):
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

        # create final toggle
        self.show_process = True
        self.final_toggle = pn.widgets.Toggle(name='Final', value=False, align="center")
        self.final_toggle.param.watch(self.final_toggle_changed, parameter_names=['value'], onlychanged=False)

        self.widgets = pn.Row(self.col_display, self.minus_button, self.col_widget,  self.final_toggle)

    @param.depends('widgets')
    def __panel__(self):
        return self.widgets


    def add_col(self, event):
        if event.new != "" and self.active:
            self.all_selected_cols_final.append(event.new)
            self.update_col_whitelist()


    def col_selected(self, col):
        if self.active:
            self.active = False
            if col is not None:
                index = self.all_selected_cols_final.index(col)
                self.all_selected_cols = self.all_selected_cols_final[:index+1]

            self.show_process = True
            self.final_toggle.value = False
            self.active = True

    def update_col_whitelist(self):
        self.active = False
        self.col_whitelist = [col for col in self.columns if col not in self.all_selected_cols_final]
        self.col_widget.options = ["", *self.col_whitelist]
        self.col_widget.value = ""
        self.widgets = pn.Row(self.col_display, self.minus_button, self.col_widget, self.final_toggle)

        self.active = True


        if len(self.all_selected_cols_final) > 0:
            self.col_display.visible = True
            self.col_display.options = self.all_selected_cols_final
            self.col_display.value = self.all_selected_cols_final[-1]

        else:
            self.col_display.visible = False



    def remove_col(self, event):
        if len(self.all_selected_cols_final) > 0:
            self.all_selected_cols_final = self.all_selected_cols_final[:-1]
            self.update_col_whitelist()

    def load_new_columns(self, columns):
        self.columns = columns
        self.all_selected_cols = []
        self.update_col_whitelist()

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

