def clean_column_groups(group):
    columns = []
    for col in group:
        if (len(col.value) > 1):
            columns.append(col.value)

    return columns


def return_col(combined_col):
    col = combined_col.split(", ")
    return [c for c in col]
