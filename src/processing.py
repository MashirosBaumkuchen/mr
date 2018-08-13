# coding=utf-8
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import gc

class DataSet:
    def __init__(self):
        self.path = '../input/train.csv'
        self.posi_f = [
            'voice_connection', 'wifi_connection',
            'voice_convert_1', 'convert_rate',
            'rrc_connection', 'erab_connection',
            'esrvcc_convert'
            ]
        self.navg_f = [
            'voice_disconnection', 'wifi_disconnection',
            'wifi_disconnection_1',
            'erab_trash', 'prb_pull', 'prb_push'
            ]
        self.count_f = [
            'voice_pull_delay','voice_count', 'data_count', 
            'rrc_max', 'csgb_rrc', 'rrc_2g', 'rrc_3g', 'rrc_num',
            'voice_push_miss', 'voice_pull_miss'
            ]
        self.drop_f = [
            'video_connection', 'video_disconnection', 'voice_convert_2', 'voice_convert_2', 'pdcch_cce'
            ]
        self.cat_f = None
        self.train_x = None
        self.train_y_low = None
        self.train_y_high = None
        self.test_x = None
        self.test_y_low = None
        self.test_y_high = None

        
    def load(self):
        print('start loading')
        train = pd.read_csv(self.path)
        train = train[train['mr_low']>0]
        for feature in self.posi_f:
            print('format', self.posi_f)
            train = self.format_(train, feature, fillna=train[feature].mean())
        for feature in self.navg_f:
            print('format', self.navg_f)
            train = self.format_(train, feature, fillna=train[feature].mean())
        for feature in self.count_f:
            print('format', self.count_f)
            train = self.format_(train, feature, fillna=train[feature].mean(), normalize=False)
            train[feature] = self.normalized_feature(train, feature)
        train = train.drop(self.drop_f, axis=1)
        
        train.mr_low = train.mr_low.astype(np.float32)
        train.mr_high = train.mr_high.astype(np.float32)
        
        print('cgi')
        # train['MCC'] = train['cgi'].apply(lambda x: str(x).split('-')[0])
        # train['MNC'] = train['cgi'].apply(lambda x: str(x).split('-')[1])
        train['ENODEB_ID'] = train['cgi'].apply(lambda x: str(x).split('-')[2])
        train['CID'] = train['cgi'].apply(lambda x: str(x).split('-')[3])
        train['ENODEB_ID'] = train['ENODEB_ID'].astype(np.int32)
        train['CID'] = train['CID'].astype(np.int32)
        
        print('time')
        train['datetime'] = train['time'].apply(lambda x: str(x).split(' ')[0])
        # train['month'] = pd.to_datetime(train['time']).dt.month
        # train['day'] = pd.to_datetime(train['time']).dt.day
        train['hour'] = pd.to_datetime(train['time']).dt.hour
        
        print('region')
        train['region'] = LabelEncoder().fit_transform(train['region'])
        # train['type'] = LabelEncoder().fit_transform(train['type'])
        train = train.drop(['city', 'cgi', 'time'], axis=1)
        num_f = self.posi_f + self.navg_f + self.count_f
        self.cat_f = [
            'region', 'ENODEB_ID', 'CID', 'hour'
        ]
        
        print('split by date')
        test_set = train[train.loc[:,('datetime')]=='2018-05-06']
        train_set = train[train.loc[:,('datetime')]=='2018-05-05']
        train_set = pd.concat([train_set, train[train.loc[:,('datetime')]=='2018-05-04']])
        train_set = pd.concat([train_set, train[train.loc[:,('datetime')]=='2018-05-03']])
        train_set = pd.concat([train_set, train[train.loc[:,('datetime')]=='2018-05-02']])
        train_set = pd.concat([train_set, train[train.loc[:,('datetime')]=='2018-05-01']])
        train_set = pd.concat([train_set, train[train.loc[:,('datetime')]=='2018-04-30']])
        test_set = test_set.drop(['datetime'], axis=1)
        train_set = train_set.drop(['datetime'], axis=1)
        
        self.train_y_low = train_set['mr_low']
        self.train_y_high = train_set['mr_high']
        self.train_x = train_set.drop(['mr_low', 'mr_high'], axis=1)

        self.test_y_low = test_set['mr_low']
        self.test_y_high = test_set['mr_high']
        self.test_x = test_set.drop(['mr_low', 'mr_high'], axis=1)
        
        del train
        del train_set
        del test_set
        gc.collect()
        return

    def format_(self, dataframe, feature, fillna='0.0', astype=np.float32, normalize=True):
        print('format', feature)
        dataframe[feature] = dataframe[feature].fillna(fillna)
        dataframe[feature] = dataframe[feature].astype(np.float32)
        if normalize:
            dataframe.loc[dataframe[feature]>1, feature] = 1
            dataframe.loc[dataframe[feature]<0, feature] = 0
        return dataframe
    
    def normalized_feature(self, dataframe, feature):
        print('start scale', feature)
        mms = MinMaxScaler()
        return mms.fit_transform(dataframe[feature].values.reshape(-1, 1))
