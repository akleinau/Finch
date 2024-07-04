import panel as pn
import param


class FeatureIter(param.Parameterized):
    all_selected_cols = param.List()
    show_process = param.Boolean()

    def __init__(self, columns, **params):
        super().__init__(**params)

        self.columns = columns
        self.all_selected_cols = []

        self.col_widget = pn.widgets.Select(name='add', options=["", *columns], value="")
        self.col_widget.param.watch(self.changed, parameter_names=['value'], onlychanged=False)

        self.col_display = pn.Row()
        self.col_type = 'singular'
        self.all_selected_cols = []
        self.col_whitelist = columns

        # create plus button
        self.plus_button = pn.widgets.Button(name='+', button_type='primary')
        self.plus_button.on_click(self.add_widget)

        # create minus button
        self.minus_button = pn.widgets.Button(name='-', button_type='primary')
        self.minus_button.on_click(self.remove_widget)

        # create final toggle
        self.show_process = True
        self.final_toggle = pn.widgets.Toggle(name='Final', value=False)
        self.final_toggle.param.watch(self.final_toggle_changed, parameter_names=['value'], onlychanged=False)

        self.update_widgets()


    def changed(self, event):

        new_cols = self.all_selected_cols.copy()
        if (event.old == ""):
            if (event.new != ""):
                new_cols.append(event.new)
        elif (event.new == ""):
            new_cols = new_cols[:-1]
        else:
            new_cols[-1] = event.new
        self.param.update(all_selected_cols=new_cols)
        self.update_col_whitelist()

    def update_col_whitelist(self):
        list = self.all_selected_cols if self.col_widget.value == "" else self.all_selected_cols[:-1]

        self.col_whitelist = [col for col in self.columns if col not in list]

    def add_widget(self, event):
        if (self.col_widget.value != ""):
            self.all_selected_cols.append(self.col_widget.value)
            self.update_col_whitelist()
            self.col_widget.value = ""
            self.col_widget.options = ["", *self.col_whitelist]
            self.update_widgets()

    def remove_widget(self, event):
        self.all_selected_cols = self.all_selected_cols[:-1]
        self.update_col_whitelist()
        self.col_widget.options = ["", *self.col_whitelist]
        self.col_widget.value = self.all_selected_cols[-1]

        self.update_widgets()

    def update_col_display(self):
        self.col_display.clear()
        list = self.all_selected_cols if self.col_widget.value == "" else self.all_selected_cols[:-1]
        for col in list:
            self.col_display.append("## " + col + ", ")

    def update_widgets(self):
        self.update_col_display()

        self.widgets = pn.Row(self.col_display, self.col_widget, self.plus_button, self.minus_button, self.final_toggle)

    def load_new_columns(self, columns):
        self.columns = columns
        self.all_selected_cols = []
        self.update_col_whitelist()
        self.col_widget.options = ["", *self.col_whitelist]
        self.col_widget.value = ""
        self.update_widgets()

    def final_toggle_changed(self, event):
        self.show_process = not event.new


def get_first(event):
    if len(event) > 0:
        return event[0]
    return None

