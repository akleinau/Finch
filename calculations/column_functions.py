import panel as pn

def clean_column_groups(group):
    columns = []
    for col in group:
        if (len(col.value) > 1):
            columns.append(col.value)

    return columns


def return_col(combined_col):
    col = combined_col.split(", ")
    return [c for c in col]

def get_group_options(index, widgets):
    column_group = widgets[0]
    remaining_options = widgets[3]
    # combine lists remaining_options with items in columngroup[index]
    return remaining_options.value.copy() + column_group[index].value.copy()


def column_group_changed(event, widgets):
    column_group = widgets[0]
    row = widgets[1]
    num_groups = widgets[2]
    remaining_options = widgets[3]
    combined_columns = widgets[4]
    all_options = widgets[5]

    index = int(event.obj.name)
    combined_columns.value = clean_column_groups(column_group)
    length = len(column_group[index].value)
    if length == 0 and index != (num_groups.value - 1):
        # remove old widget
        column_group.pop(index)
        row.pop(index)
        update_names(widgets)
        num_groups.value -= 1
    if length > 1 and (index == (num_groups.value - 1)):
        # add new widget
        column_group.append(
            pn.widgets.MultiChoice(name=str(num_groups.value), value=[], options=remaining_options.value.copy()))
        column_group[num_groups.value].param.watch(lambda event: column_group_changed(event, widgets), parameter_names=['value'], onlychanged=False)
        row.append(column_group[num_groups.value])
        num_groups.value += 1

    # update remaining options
    columns = []
    for col in column_group:
        for name in col.value:
            columns.append(name)
    remaining_options.value = pn.bind(lambda all: [name for name in all if name not in columns], all_options)
    for col in column_group:
        col.options = get_group_options(int(col.name), widgets)


def update_names(widgets):
    column_group = widgets[0]
    num_groups = widgets[2]
    for i in range(num_groups.value - 1):
        column_group[i].name = str(i)
