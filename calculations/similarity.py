

def get_similar_items(data, item, exclude_cols):
    # standardize the data
    data = data.copy()
    item_data = item.data_raw.copy()
    columns = list(data.columns)
    excluded_columns = ['prob_', 'scatter', 'prediction', 'group']
    if len(exclude_cols) > 0:
        excluded_columns.extend(exclude_cols)
    columns = [col for col in columns if not any([excluded in col for excluded in excluded_columns])]

    for col in columns:
        mean = data[col].mean()
        std = data[col].std()
        data[col] = (data[col] - mean) / std
        item_data[col] = (item_data[col] - mean) / std

    # calculate distance to item
    data['distance'] = ((data[columns] - item_data[columns])**2).sum(axis=1)

    # get the 10% closest items
    data = data.sort_values(by='distance')
    data = data.head(int(len(data) * 0.05))

    return data