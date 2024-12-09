#https://github.com/BokwaiHo/DEGC

def month_transfer(month):
    if month == 'Jan':
        month = '01'
    elif month == 'Feb':
        month = '02'
    elif month == 'Mar':
        month = '03'
    elif month == 'Apr':
        month = '04'
    elif month == 'May':
        month = '05'
    elif month == 'Jun':
        month = '06'
    elif month == 'Jul':
        month = '07'
    elif month == 'Aug':
        month = '08'
    elif month == 'Sep':
        month = '09'
    elif month == 'Oct':
        month = '10'
    elif month == 'Nov':
        month = '11'
    elif month == 'Dec':
        month = '12'
    else:
        #print(month)
        pass
    return month

def year_transfer(year):
    return str(str(year)[:4])

def day_transfer(day):
    if day<10:
        return '0'+str(int(day))
    else:
        return str(int(day))

def date_create(year, month, day):
    return year + month + day

def date_new_create(year, month, day):
    return year + '-' + month + '-' + day

def remove_infrequent_node(df, node_type, min_counts=5):
    n_node_type = len(df[node_type].unique())
    counts = df[node_type].value_counts()
    df = df[df[node_type].isin(counts[counts >= min_counts].index)]
    n_removed = n_node_type - len(df[node_type].unique())
    return df, n_removed

def k_core_foursquare(df, u_thre, i_thre):
    filtered_data = df.copy()
    filtered_data, u_removed = remove_infrequent_node(filtered_data, 'user_id', u_thre)
    filtered_data, i_removed = remove_infrequent_node(filtered_data, 'item_id', i_thre)

    while(u_removed != 0 or i_removed != 0):
        filtered_data, u_removed = remove_infrequent_node(filtered_data, 'user_id', u_thre)
        filtered_data, i_removed = remove_infrequent_node(filtered_data, 'item_id', i_thre)

    filtered_data = filtered_data.sort_values('timestamp')
    return filtered_data