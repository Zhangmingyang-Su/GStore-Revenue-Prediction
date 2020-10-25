import numpy as np
import pandas as pd
import datatime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# feature engineering

def feature_engineering(train):
    train['trafficSource.adwordsClickInfo.isVideoAd'].fillna(True, inplace=True)
    train['trafficSource.isTrueDirect'].fillna(False, inplace=True)

    # remove columns with only one distinct value
    cols_to_drop = [col for col in train.columns if train[col].nunique(dropna=False) == 1]
    train.drop(cols_to_drop, axis=1, inplace=True)

    #only one not null value
    train.drop(['trafficSource.campaignCode'], axis=1, inplace=True)
# 对数值特征采用np.log1p方法进行平滑操作

    num_cols = ['visitNumber', 'totals.hits', 'totals.pageviews', 'totals.bounces', 'totals.newVisits', 'totals.transactionRevenue']

    for col in num_cols:
        train[col] = train[col].fillna(0)
        train[col] = train[col].astype(float)
        train[col] = np.log1p(train[col])

# 其他补0就好
    train['trafficSource.adContent'] = train['trafficSource.adContent'].fillna(0)
    train['trafficSource.keyword'] = train['trafficSource.keyword'].fillna(0)
    train['trafficSource.adwordsClickInfo.adNetworkType'] = train['trafficSource.adwordsClickInfo.adNetworkType'].fillna(0)
    train['trafficSource.adwordsClickInfo.gclId'] = train['trafficSource.adwordsClickInfo.gclId'].fillna(0)
    train['trafficSource.adwordsClickInfo.page'] = train['trafficSource.adwordsClickInfo.page'].fillna(0)
    train['trafficSource.adwordsClickInfo.slot'] = train['trafficSource.adwordsClickInfo.slot'].fillna(0)


# 新特征构建
    train['browser_category'] = train['device.browser'] + '_' + train['device.deviceCategory']
    train['browser_operatingSystem'] = train['device.browser'] + '_' + train['device.operatingSystem']
    train['source_country'] = train['trafficSource.source'] + '_' + train['geoNetwork.country']

# 移除没有用的columns：
    no_use = ["date", "fullVisitorId", "sessionId", "visitId", "visitStartTime", 'totals.transactionRevenue', 'trafficSource.referralPath']
    cat_cols = [col for col in train.columns if col not in num_cols and col not in no_use]

# categorical encoding
    max_values = {}
    for col in cat_cols:
        print(col)
        lbl = LabelEncoder()
        lbl.fit(list(train[col].values.astype('str')))
        train[col] = lbl.transform(list(train[col].values.astype('str')))
        max_values[col] = train[col].max() + 2  # 根据经验，比真实值大一点，效果较好
    return train, max_values


# cat_col_labels1 = ["channelGrouping", "device.deviceCategory", "device.operatingSystem", "geoNetwork.continent",
#                    "geoNetwork.subContinent", "trafficSource.adContent", "trafficSource.adwordsClickInfo.adNetworkType",
#                    "trafficSource.adwordsClickInfo.isVideoAd", "trafficSource.adwordsClickInfo.page", "trafficSource.adwordsClickInfo.slot",
#                    "trafficSource.campaign", "trafficSource.medium", "geoNetwork.region"]
#
# cat_col_labels2 = ["browser_category", "browser_operatingSystem", "source_country", "device.browser", "geoNetwork.city",
#                    "trafficSource.source", "trafficSource.keyword", "trafficSource.adwordsClickInfo.gclId", "geoNetwork.networkDomain",
#                    "geoNetwork.country", "geoNetwork.metro", "geoNetwork.region"]


def data_preprocessing(train):
    train = train.sort_values('date')

    x_train = train[train["date"] <= pd.Timestamp(2017,5,31)]
    x_val = train[train["date"] > pd.Timestamp(2017,5,31)]

    y_train = x_train['totals.transactionRevenue']
    y_val = x_val['totals.transactionRevenue']

    x_train = x_train.drop(no_use, axis=1)
    x_val = x_val.drop(no_use, axis=1)

    return x_train, x_val, y_train, y_val
