import pandas as pd

def last_touch_attribution(events: pd.DataFrame) -> pd.DataFrame:
    # Filter only events where a purchase was made
    purchased_events = events[events['is_purchased'] == 1]

    # Create a DataFrame to store information about the last touch before purchase
    last_touch_data = []

    # For each purchase, identify the last touch
    for index, row in purchased_events.iterrows():
        # Get all touches for this user up to the week of the purchase
        user_touches = events[(events['user_id'] == row['user_id']) & (events['week'] <= row['week'])]

        # Find the last touch before the purchase
        last_touch = user_touches[user_touches['week'] == user_touches['week'].max()].iloc[-1]

        # Add information about the last touch and the GMV of the purchase
        last_touch_data.append([row['week'], row['user_id'], last_touch['channel'], row['gmv']])

    # Create a DataFrame from the collected information
    last_touch_df = pd.DataFrame(last_touch_data, columns=['week', 'user_id', 'channel', 'gmv'])

    # Create columns for each channel
    for channel in events['channel'].unique():
        last_touch_df[channel] = 0

    # Distribute GMV by channels
    for index, row in last_touch_df.iterrows():
        last_touch_df.at[index, row['channel']] = row['gmv']

    # Group by user_id and week, summing GMV by channels
    attribution_summary = last_touch_df.groupby(['week', 'user_id']).sum().reset_index()

    # Add total GMV column
    attribution_summary['total_gmv'] = attribution_summary[events['channel'].unique()].sum(axis=1)
    attribution_summary.drop(columns=['gmv'], inplace=True)

    return attribution_summary

def first_touch_attribution(events: pd.DataFrame) -> pd.DataFrame:
    # Filter only events where a purchase was made
    purchased_events = events[events['is_purchased'] == 1]

    # Create a DataFrame for storing information about the first touch before a purchase
    first_touch_data = []

    # For each purchase, determine the first touch after the previous purchase
    for index, row in purchased_events.iterrows():
        # Get all touches for this user up to the week of the purchase
        user_touches = events[(events['user_id'] == row['user_id']) & (events['week'] <= row['week'])]

        # If the user had previous purchases, consider touches after the last purchase
        if not purchased_events[(purchased_events['user_id'] == row['user_id']) & (purchased_events['week'] < row['week'])].empty:
            last_purchase_week = purchased_events[(purchased_events['user_id'] == row['user_id']) & (purchased_events['week'] < row['week'])]['week'].max()
            user_touches = user_touches[user_touches['week'] > last_purchase_week]

        # Find the first touch before the purchase
        if not user_touches.empty:
            first_touch = user_touches[user_touches['week'] == user_touches['week'].min()].iloc[0]
            first_touch_data.append([row['week'], row['user_id'], first_touch['channel'], row['gmv']])

    # Create a DataFrame from the collected information
    first_touch_df = pd.DataFrame(first_touch_data, columns=['week', 'user_id', 'channel', 'gmv'])

    # Create columns for each channel
    for channel in events['channel'].unique():
        first_touch_df[channel] = 0

    # Distribute GMV by channels
    for index, row in first_touch_df.iterrows():
        first_touch_df.at[index, row['channel']] = row['gmv']

    # Group by user_id and week, summing GMV by channels
    attribution_summary = first_touch_df.groupby(['week', 'user_id']).sum().reset_index()

    # Add total GMV column
    attribution_summary['total_gmv'] = attribution_summary[events['channel'].unique()].sum(axis=1)
    attribution_summary.drop(columns=['gmv'], inplace=True)

    return attribution_summary

def linear_attribution(events: pd.DataFrame) -> pd.DataFrame:
    # Create a DataFrame for storing information about linear attribution
    linear_attribution_data = []

    # Iterate through each purchase event
    for _, purchase_row in events[events['is_purchased'] == 1].iterrows():
        user_id = purchase_row['user_id']
        purchase_week = purchase_row['week']
        gmv = purchase_row['gmv']

        # Get all touches for this user up to the purchase (inclusive)
        user_touches_before_purchase = events[(events['user_id'] == user_id) & (events['week'] <= purchase_week)]

        # Exclude touches after the previous purchase
        previous_purchases = events[(events['user_id'] == user_id) & (events['week'] < purchase_week) & (events['is_purchased'] == 1)]
        if not previous_purchases.empty:
            last_purchase_week = previous_purchases['week'].max()
            user_touches_before_purchase = user_touches_before_purchase[user_touches_before_purchase['week'] > last_purchase_week]

        # Distribute GMV evenly across touches, if any
        if not user_touches_before_purchase.empty:
            gmv_per_touch = gmv / len(user_touches_before_purchase)
            for _, touch_row in user_touches_before_purchase.iterrows():
                linear_attribution_data.append([purchase_week, user_id, touch_row['channel'], gmv_per_touch])

    # Create a DataFrame from the collected information
    linear_attribution_df = pd.DataFrame(linear_attribution_data, columns=['week', 'user_id', 'channel', 'gmv'])

    # Group by user_id and week, sum GMV by channels with rounding
    attribution_summary = linear_attribution_df.pivot_table(index=['week', 'user_id'], columns='channel', values='gmv', aggfunc='sum', fill_value=0)

    # Add total GMV column
    attribution_summary['total_gmv'] = attribution_summary.sum(axis=1)

    # Round final GMV values by channels to two decimal places
    attribution_summary = attribution_summary.round(2)

    # Reset index for better data structure
    attribution_summary.reset_index(inplace=True)

    return attribution_summary

def u_shaped_attribution(events: pd.DataFrame) -> pd.DataFrame:
    # Filter only the rows where a purchase was made
    attribution = events.pivot_table(index='user_id', columns='channel', values='gmv', fill_value=0)
    return attribution
