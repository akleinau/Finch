import panel as pn
import param


class ColumnGrouping(param.Parameterized):
    combined_columns = param.List()

    def __init__(self, columns, **params):
        super().__init__(**params)
        self.column_group = []
        self.row = pn.FlexBox()
        self.num_groups = 1
        self.remaining_options = columns.copy()
        self.all_options = columns.copy()


    def clean_column_groups(self, group):
        columns = []
        for col in group:
            if (len(col.value) > 1):
                columns.append(col.value)

        return columns


    def get_group_options(self, index):
        # combine lists remaining_options with items in columngroup[index]
        return self.remaining_options.copy() + self.column_group[index].value.copy()


    def column_group_changed(self, event):
        index = int(event.obj.name)
        self.combined_columns = self.clean_column_groups(self.column_group)
        length = len(self.column_group[index].value)
        if length == 0 and index != (self.num_groups - 1):
            # remove old widget
            self.column_group.pop(index)
            self.row.pop(index)
            self.update_names()
            self.num_groups -= 1
        if length > 1 and (index == (self.num_groups - 1)):
            # add new widget
            self.column_group.append(
                pn.widgets.MultiChoice(name=str(self.num_groups), value=[], options=self.remaining_options.copy()))
            self.column_group[self.num_groups].param.watch(lambda event: self.column_group_changed(event),
                                                       parameter_names=['value'], onlychanged=False)
            self.row.append(self.column_group[self.num_groups])
            self.num_groups += 1

        # update remaining options
        columns = []
        for col in self.column_group:
            for name in col.value:
                columns.append(name)
        self.remaining_options =[name for name in self.all_options if name not in columns]
        for col in self.column_group:
            col.options = self.get_group_options(int(col.name))


    def update_names(self):
        for i in range(self.num_groups - 1):
            self.column_group[i].name = str(i)


    def init_groups(self, columns=None):
        if columns is not None:
            self.all_options = columns.copy()
            self.remaining_options = columns.copy()


        self.combined_columns = []
        self.num_groups = 1
        self.column_group.clear()
        self.row.clear()
        self.column_group.append(
            pn.widgets.MultiChoice(name=str(0), value=[], options=self.remaining_options.copy()))
        self.column_group[0].param.watch(lambda event: self.column_group_changed(event),
                                    parameter_names=['value'], onlychanged=False)
        self.row.append(self.column_group[0])
        self.column_group[0].param.trigger('value')  # trigger event to update remaining options

def return_col(combined_col):
    col = combined_col.split(", ")
    return [c for c in col]