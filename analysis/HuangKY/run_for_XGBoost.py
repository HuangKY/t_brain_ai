# impirt important packages
import pandas as pd
import numpy as np
import xgboost as xgb
pd.options.display.max_columns = None  # show all columns



# read data
airline_data = pd.read_csv('../../data/dataset/airline.csv')
# cache_map_data = pd.read_csv('../../data/dataset/cache_map.csv')
group_data = pd.read_csv('../../data/dataset/group.csv')
order_data = pd.read_csv('../../data/dataset/order.csv')
day_schedule_data = pd.read_csv('../../data/dataset/day_schedule.csv')

#
column_names = ['AirportID', 'Name', 'City', 'Country', 'IATA', 'ICAO', 'Latitude', 'Longitude',
              'Altitude', 'Timezone', 'DST', 'TZ', 'Type','Source']
df_airport = pd.read_csv('../../data/airport.csv', names=column_names) #ref: https://openflights.org/data.html#airport

# test and train data
train_data = pd.read_csv('../../data/training-set.csv')
test_data = pd.read_csv('../../data/testing-set.csv')


# orders

order_data['order_date_dt'] = pd.to_datetime(order_data.order_date, format='%d-%b-%y') # ref: http://strftime.org/
order_data['order_year'] = order_data.order_date_dt.dt.year
order_data['order_month'] = order_data.order_date_dt.dt.month
order_data['order_day'] = order_data.order_date_dt.dt.day
order_data['order_dayofyear'] = order_data.order_date_dt.dt.dayofyear
order_data['order_weekday'] = order_data.order_date_dt.dt.dayofweek # 0=Mon, 6=Sun
order_data['unit']=order_data.unit.str.replace('unit_value_', '').astype(int)
order_data['source_1']=order_data.source_1.str.replace('src1_value_', '')
order_data['source_2']=order_data.source_2.str.replace('src2_value_', '')


group_data['sub_line'] = group_data.sub_line.str.replace('subline_value_','')
group_data['area'] = group_data.area.str.replace('area_value_','')
group_data['begin_date_dt'] = pd.to_datetime(group_data.begin_date, format='%d-%b-%y')
group_data['begin_year'] = group_data.begin_date_dt.dt.year
group_data['begin_month'] = group_data.begin_date_dt.dt.month
group_data['begin_day'] = group_data.begin_date_dt.dt.day
group_data['begin_dayofyear'] = group_data.begin_date_dt.dt.dayofyear
group_data['begin_weekday'] = group_data.begin_date_dt.dt.dayofweek # 0=Mon, 6=Sun
group_data['end_date_dt'] = group_data.begin_date_dt + group_data['days'].apply(np.ceil).apply(lambda x: pd.Timedelta(x, unit='D'))
group_data['end_year'] = group_data.end_date_dt.dt.year
group_data['end_month'] = group_data.end_date_dt.dt.month
group_data['end_day'] = group_data.end_date_dt.dt.day
group_data['end_dayofyear'] = group_data.end_date_dt.dt.dayofyear
group_data['end_weekday'] = group_data.end_date_dt.dt.dayofweek # 0=Mon, 6=Sun


#
airline_data['dst_airport'] = airline_data.dst_airport.str.replace('([A-Z]+) ([^A-Z]+)','\\1')
airline_data['src_airport'] = airline_data.src_airport.str.replace('([A-Z]+) ([^A-Z]+)','\\1')
airline_data['go_back'] = airline_data.go_back.str.replace('去程', 'go')
airline_data['go_back'] = airline_data.go_back.str.replace('回程', 'back')

# unify time, compute the travel time for each routes
airline_data['src_IATA'] = airline_data.src_airport.str.replace('([A-Z]+) ([^A-Z])+', '\\1')
airline_data['dst_IATA'] = airline_data.dst_airport.str.replace('([A-Z]+) ([^A-Z])+', '\\1')
tmp = pd.merge(airline_data, df_airport[['IATA','TZ']], left_on='src_IATA', right_on='IATA', how='left')
airline_data = tmp.rename(index=str, columns={'TZ':'src_TZ'}).drop(['IATA'], axis=1)
tmp = pd.merge(airline_data, df_airport[['IATA','TZ']], left_on='dst_IATA', right_on='IATA', how='left')
airline_data = tmp.rename(index=str, columns={'TZ':'dst_TZ'}).drop(['IATA'], axis=1)
airline_data['fly_time'] = pd.to_datetime(airline_data.fly_time, format='%Y/%m/%d %H:%M')
airline_data['arrive_time'] = pd.to_datetime(airline_data.arrive_time, format='%Y/%m/%d %H:%M')

def src_time_utc(x):
    try:
        res = x['fly_time'].tz_localize(x['src_TZ']).tz_convert('UTC')
    except:
        res = None # 因為有些機場沒有抓到對應time-zone (可後續用手動補)
    return res

def dst_time_utc(x):
    try:
        res = x['arrive_time'].tz_localize(x['dst_TZ']).tz_convert('UTC')
    except:
        res = None
    return res

airline_data['arrive_time_utc'] = airline_data.apply(dst_time_utc, axis=1)
airline_data['fly_time_utc'] = airline_data.apply(src_time_utc, axis=1)
airline_data['traval_time'] = airline_data['arrive_time_utc'] - airline_data['fly_time_utc']

# derive number of routes for each group
go_route_num=airline_data[airline_data['go_back']=='go'].groupby('group_id')['dst_airport'].nunique()
back_route_num=airline_data[airline_data['go_back']=='back'].groupby('group_id')['dst_airport'].nunique()
df_route_num = pd.merge(go_route_num.to_frame('go_route_num'), back_route_num.to_frame('back_route_num'), left_index=True, right_index=True,how='outer')

# derive total travel time for each group
def mean(x):
    return x.sum()/x.count()

#
tmp=airline_data[airline_data.go_back=='go'].groupby(['group_id','dst_IATA'])['traval_time'].apply(mean)
#print(tmp.head())
go_time=tmp.to_frame().groupby(['group_id'])['traval_time'].sum()

#
tmp=airline_data[airline_data.go_back=='back'].groupby(['group_id','dst_IATA'])['traval_time'].apply(mean)
#print(tmp.head())
back_time=tmp.to_frame().groupby(['group_id'])['traval_time'].sum()

#
df_total_travel_time = pd.merge(go_time.to_frame('go_time'), back_time.to_frame('back_time'), left_index=True, right_index=True,how='outer')


tmp = airline_data[airline_data['go_back']=='go'].sort_values(by='fly_time')
df_airline_fly_time = tmp.groupby('group_id').first().reset_index()[['group_id', 'fly_time']]




tmp = pd.merge(order_data, df_total_travel_time.reset_index(), on='group_id', how='left')
tmp = pd.merge(tmp, df_route_num.reset_index(), on='group_id', how='left')
tmp = pd.merge(tmp, df_airline_fly_time, on='group_id', how='left')
df_predictors = pd.merge(tmp, group_data.drop(['product_name','promotion_prog'], axis=1), on='group_id', how='left').drop('order_date', axis=1)
df_predictors['date_diff_order_begin'] = df_predictors.fly_time - df_predictors.order_date_dt



# 把時間轉單位轉成總共幾分
df_predictors.go_time = df_predictors.go_time.apply(lambda x: x.total_seconds()/60)
df_predictors.back_time = df_predictors.back_time.apply(lambda x: x.total_seconds()/60)
df_predictors.date_diff_order_begin = df_predictors.date_diff_order_begin.apply(lambda x: x.total_seconds()/60)


df_predictors = df_predictors.drop(['order_date_dt','begin_date_dt','end_date_dt'], axis=1)





######################################################  XGBoost
# dummy化
df_predictors_dummy = pd.get_dummies(df_predictors)


# seperate into train and test features
train_features_dummy = pd.merge(df_predictors_dummy, train_data, on='order_id', how='inner')
test_features_dummy = pd.merge(df_predictors_dummy, test_data, on='order_id', how='inner')


# train features: drop nulls
train_features_dummy = train_features_dummy.dropna()  # TODO:  try to save more data !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

train_features_y_dummy = train_features_dummy['deal_or_not']
train_features_X_dummy = train_features_dummy.drop(['order_id', 'group_id', 'begin_date', 'deal_or_not'], axis=1)


# test features: fill nulls with 0
test_features_X_dummy = test_features_dummy.fillna(0)   ## TODO:  fillna考慮 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
test_features_X_dummy = test_features_dummy.drop(['order_id', 'group_id', 'begin_date', ' deal_or_not'], axis=1)


X_train_dummy, X_test_dummy, y_train_dummy, y_test_dummy = train_test_split(train_features_dummy,
                                                    train_features_y_dummy,
                                                    test_size = 0.33,
                                                    random_state = 0)



xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train_dummy, y_train_dummy)




xgb_predictions = xgb_model.predict(X_test_dummy)


# Report the before-and-afterscores
print("\nOptimized Model\n------")
print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test_dummy, xgb_predictions)))
print("Final precision score on the testing data: {:.4f}".format(precision_score(y_test_dummy, xgb_predictions)))
print("Final recall score on the testing data: {:.4f}".format(recall_score(y_test_dummy, xgb_predictions)))
print("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test_dummy, xgb_predictions, beta = 0.5)))




xgb_results = pd.DataFrame(data={'order_id': test_features['order_id'], 'deal_or_not' : xgb_model.predict(test_features_X_dummy)})
xgb_results.to_csv('xgb_results.csv', index=False)
