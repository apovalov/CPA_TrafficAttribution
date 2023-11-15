import pandas as pd


def last_touch_attribution(events: pd.DataFrame) -> pd.DataFrame:
    # Отфильтровываем только те события, где была совершена покупка
    purchased_events = events[events['is_purchased'] == 1]

    # Создаем DataFrame для хранения информации о последнем касании перед покупкой
    last_touch_data = []

    # Для каждой покупки определяем последнее касание
    for index, row in purchased_events.iterrows():
        # Получаем все касания для данного пользователя до недели покупки
        user_touches = events[(events['user_id'] == row['user_id']) & (events['week'] <= row['week'])]

        # Находим последнее касание перед покупкой
        last_touch = user_touches[user_touches['week'] == user_touches['week'].max()].iloc[-1]

        # Добавляем информацию о последнем касании и GMV покупки
        last_touch_data.append([row['week'], row['user_id'], last_touch['channel'], row['gmv']])

    # Создаем DataFrame из собранной информации
    last_touch_df = pd.DataFrame(last_touch_data, columns=['week', 'user_id', 'channel', 'gmv'])

    # Создание столбцов для каждого канала
    for channel in events['channel'].unique():
        last_touch_df[channel] = 0

    # Распределение GMV по каналам
    for index, row in last_touch_df.iterrows():
        last_touch_df.at[index, row['channel']] = row['gmv']

    # Группировка по user_id и неделе, суммирование GMV по каналам
    attribution_summary = last_touch_df.groupby(['week', 'user_id']).sum().reset_index()

    # Добавление общего GMV
    attribution_summary['total_gmv'] = attribution_summary[events['channel'].unique()].sum(axis=1)
    attribution_summary.drop(columns=['gmv'], inplace=True)

    return attribution_summary

#
#
# events = pd.read_csv('data/events.csv')
# # print(events.head())
#
# print('Last touch attribution')
# print(last_touch_attribution(events))


def first_touch_attribution(events: pd.DataFrame) -> pd.DataFrame:
    # Отфильтровываем только те события, где была совершена покупка
    purchased_events = events[events['is_purchased'] == 1]

    # Создаем DataFrame для хранения информации о первом касании перед покупкой
    first_touch_data = []

    # Для каждой покупки определяем первое касание после предыдущей покупки
    for index, row in purchased_events.iterrows():
        # Получаем все касания для данного пользователя до недели покупки
        user_touches = events[(events['user_id'] == row['user_id']) & (events['week'] <= row['week'])]

        # Если у пользователя были предыдущие покупки, учитываем касания после последней покупки
        if not purchased_events[
            (purchased_events['user_id'] == row['user_id']) & (purchased_events['week'] < row['week'])].empty:
            last_purchase_week = purchased_events[
                (purchased_events['user_id'] == row['user_id']) & (purchased_events['week'] < row['week'])][
                'week'].max()
            user_touches = user_touches[user_touches['week'] > last_purchase_week]

        # Находим первое касание перед покупкой
        if not user_touches.empty:
            first_touch = user_touches[user_touches['week'] == user_touches['week'].min()].iloc[0]
            first_touch_data.append([row['week'], row['user_id'], first_touch['channel'], row['gmv']])

    # Создаем DataFrame из собранной информации
    first_touch_df = pd.DataFrame(first_touch_data, columns=['week', 'user_id', 'channel', 'gmv'])

    # Создание столбцов для каждого канала
    for channel in events['channel'].unique():
        first_touch_df[channel] = 0

    # Распределение GMV по каналам
    for index, row in first_touch_df.iterrows():
        first_touch_df.at[index, row['channel']] = row['gmv']

    # Группировка по user_id и неделе, суммирование GMV по каналам
    attribution_summary = first_touch_df.groupby(['week', 'user_id']).sum().reset_index()

    # Добавление общего GMV
    attribution_summary['total_gmv'] = attribution_summary[events['channel'].unique()].sum(axis=1)

    attribution_summary.drop(columns=['gmv'], inplace=True)

    return attribution_summary


def linear_attribution(events: pd.DataFrame) -> pd.DataFrame:
    # Создаем DataFrame для хранения информации о линейной атрибуции
    linear_attribution_data = []

    # Перебираем каждое событие покупки
    for _, purchase_row in events[events['is_purchased'] == 1].iterrows():
        user_id = purchase_row['user_id']
        purchase_week = purchase_row['week']
        gmv = purchase_row['gmv']

        # Получаем все касания для данного пользователя до момента покупки (включительно)
        user_touches_before_purchase = events[(events['user_id'] == user_id) & (events['week'] <= purchase_week)]

        # Удаляем касания после предыдущей покупки
        previous_purchases = events[
            (events['user_id'] == user_id) & (events['week'] < purchase_week) & (events['is_purchased'] == 1)]
        if not previous_purchases.empty:
            last_purchase_week = previous_purchases['week'].max()
            user_touches_before_purchase = user_touches_before_purchase[
                user_touches_before_purchase['week'] > last_purchase_week]

        # Если есть касания, распределяем GMV равномерно без округления
        if not user_touches_before_purchase.empty:
            gmv_per_touch = gmv / len(user_touches_before_purchase)
            for _, touch_row in user_touches_before_purchase.iterrows():
                linear_attribution_data.append([purchase_week, user_id, touch_row['channel'], gmv_per_touch])

    # Создаем DataFrame из собранной информации
    linear_attribution_df = pd.DataFrame(linear_attribution_data, columns=['week', 'user_id', 'channel', 'gmv'])

    # Группировка по user_id и неделе, суммирование GMV по каналам с последующим округлением
    attribution_summary = linear_attribution_df.pivot_table(index=['week', 'user_id'], columns='channel', values='gmv',
                                                            aggfunc='sum', fill_value=0)
    # Добавление общего GMV
    attribution_summary['total_gmv'] = attribution_summary.sum(axis=1)

    # Округляем итоговые значения GMV по каналам до двух знаков после запятой
    attribution_summary = attribution_summary.round(2)

    # Сброс индекса для лучшей структуры данных
    attribution_summary.reset_index(inplace=True)

    return attribution_summary


def u_shaped_attribution(events: pd.DataFrame) -> pd.DataFrame:
    """Calculate U-Shaped attribution"""
    # YOUR CODE HERE
    # Filter only the rows where a purchase was made
    attribution = events.pivot_table(index='user_id', columns='channel', values='gmv', fill_value=0)
    return attribution


def roi(attribution: pd.DataFrame, ad_costs: pd.DataFrame) -> pd.DataFrame:
    # Calculate total GMV per channel
    total_gmv_per_channel = attribution.drop(columns=['total_gmv']).sum().reset_index()
    total_gmv_per_channel.columns = ['channel', 'gmv']

    # Merge with ad costs
    merged_df = pd.merge(total_gmv_per_channel, ad_costs, on='channel')

    # Calculate and round ROI
    merged_df['roi%'] = (((merged_df['gmv'] - merged_df['costs']) / merged_df['costs']) * 100).round()
    return merged_df


# Events
#    week  user_id       channel  is_purchased  gmv
# 0     1       93   context_ads             0    0
# 1     1       38   context_ads             0    0
# 2     1       60   context_ads             0    0
# 3     1       34  social_media             1   50
# 4     1       29    mobile_ads             1   10
# 5     1        5    mobile_ads             0    0
# 6     1       26    mobile_ads             1  100
# 7     1       13  social_media             1  200
# 8     1       96  social_media             1   10
# 9     1       22    mobile_ads             0    0

# Costs
# channel       gmv   costs  roi%
# social_media  4000  1000   300
# mobile_ads    4000  800    400
# bloggers      1800  900    100
# context_ads   6600  1100   500


# week          user_id social_media  mobile_ads  bloggers  context_ads  total_gmv
# 1             1       100           0           0         0            100
# 1             2       0             0           50        0            50
# 1             3       0             0           150       0            150



events = pd.read_csv('data/events.csv')
# print(events.head())

print('Last touch attribution')
print(last_touch_attribution(events))



def first_touch_attribution(events: pd.DataFrame) -> pd.DataFrame:
    """Calculate first touch attribution"""
    # YOUR CODE HERE
    ...
    return attribution


def linear_attribution(events: pd.DataFrame) -> pd.DataFrame:
    """Calculate linear attribution"""
    # YOUR CODE HERE
    ...
    return attribution


def u_shaped_attribution(events: pd.DataFrame) -> pd.DataFrame:
    """Calculate U-Shaped attribution"""
    # YOUR CODE HERE
    ...
    return attribution
