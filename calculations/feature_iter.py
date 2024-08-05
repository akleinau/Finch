import panel as pn
import param
from panel.viewable import Viewer

from plots.styling import style_options, style_icon


class FeatureIter(Viewer):
    """
    Class to coordinate the selection of features
    """

    all_selected_cols = param.List()
    all_selected_cols_final = param.List()
    show_process = param.Boolean()
    simple_next = param.Boolean()
    widgets = param.ClassSelector(class_=pn.Row)
    show_add = param.Boolean()

    def __init__(self, columns: list, **params):
        super().__init__(**params)

        self.columns = columns
        self.active = True

        icon_style = {'border': '2px solid black', 'border-radius': '50px', 'padding': '5px', 'background-color': 'white'}

        self.col_widget = pn.widgets.Select(name='add', options=["", *columns], value="")
        self.col_widget.param.watch(lambda event: self.add_col(event.new), parameter_names=['value'], onlychanged=False)
        self.add_button = pn.widgets.Button(align="center", icon='plus', name="Add Feature", button_style='outline',
                                            styles=icon_style, stylesheets=[style_icon])
        self.add_button.on_click(self.show_add_panel)


        self.col_display = pn.widgets.RadioButtonGroup(button_style='outline', align="center", stylesheets=[style_options])
        self.col_display.param.watch(lambda event: self.col_selected(event), parameter_names=['value'],
                                     onlychanged=False)

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
        self.final_toggle = pn.widgets.Toggle(name='Final', value=False, align="center", stylesheets=[style_options],
                                              button_style="outline")
        self.final_toggle.param.watch(self.final_toggle_changed, parameter_names=['value'], onlychanged=False)

        self.widgets = pn.Row(self.col_display, self.minus_button, self.add_button, self.final_toggle)

    @param.depends('widgets')
    def __panel__(self) -> pn.Row:
        return self.widgets

    def set_all_selected_cols(self, cols: list):
        self.all_selected_cols_final = cols
        self.update_widgets(final=True)

    def changed_item(self, *params):
        self.simple_next = False

    def add_col(self, col):
        if col != "" and self.active:
            self.show_add = False
            self.all_selected_cols_final.append(col)
            self.update_widgets()

    def show_add_panel(self, event):
        self.show_add = True

    def col_selected(self, event):
        """
        handles the selection of a feature for display

        """

        col = event.new


        if self.active:
            self.active = False

            if col is not None:

                # make animations smoother
                if event.old is not None:
                    old_index = self.all_selected_cols_final.index(event.old)
                    new_index = self.all_selected_cols_final.index(col)
                    self.simple_next = new_index == old_index + 1
                else:
                    self.simple_next = False

                index = self.all_selected_cols_final.index(col)
                self.all_selected_cols = self.all_selected_cols_final[:index + 1]

            if not self.show_process:
                self.show_process = True
                self.final_toggle.value = False
            self.active = True

    def update_widgets(self, final=False):
        """
        updates all widgets based on the current state

        """

        self.active = False
        self.col_whitelist = [col for col in self.columns if col not in self.all_selected_cols_final]
        self.col_widget.options = ["", *self.col_whitelist]
        self.col_widget.value = ""
        self.minus_button.visible = len(self.all_selected_cols_final) > 0
        self.widgets = pn.Row(self.col_display, self.minus_button, self.add_button, self.final_toggle)

        if final:
            self.col_display.visible = True
            self.col_display.options = self.all_selected_cols_final

        self.active = True

        if final:
            self.final_toggle.value = True
        # this intentionally triggers col_selected
        elif len(self.all_selected_cols_final) > 0:
            self.col_display.visible = True
            self.col_display.options = self.all_selected_cols_final
            self.col_display.value = self.all_selected_cols_final[-1]

        else:
            self.col_display.visible = False
            self.all_selected_cols = []

    def remove_col(self, event):
        if len(self.all_selected_cols_final) > 0:
            self.simple_next = False
            self.all_selected_cols_final = self.all_selected_cols_final[:-1]
            self.update_widgets()

    def load_new_columns(self, columns: list):
        """
        loads new columns when a new data set is loaded

        :param columns: list
        """

        self.simple_next = False
        self.columns = columns
        self.all_selected_cols = []
        self.all_selected_cols_final = []
        self.update_widgets()

    def final_toggle_changed(self, event):
        if self.active:
            self.active = False
            self.simple_next = False
            self.show_process = not event.new
            self.all_selected_cols = self.all_selected_cols_final
            self.col_display.value = None
            self.active = True


def get_first(event):
    if len(event) > 0:
        return event[0]
    return None
