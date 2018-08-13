# coding=utf-8
import pandas as pd
import numpy as np
import gc

date = ['0430', '0501', '0502', '0503', '0504', '0505', '0506']

# mr data processing
mr = pd.DataFrame()
for i in date:
    print('read for ', i)
    d = pd.read_csv('../input/knowyou_mr_2018'+i+'.csv', encoding='gbk',
                      usecols=['时间', 'CGI', 'MR下行弱覆盖比例', 'MR下行良好覆盖比例'])
    mr = pd.concat([mr, d])
mr = mr.rename(columns={'时间':'time', 'CGI':'cgi', 'MR下行弱覆盖比例':'mr_low', 'MR下行良好覆盖比例':'mr_high'})
mr = mr.replace('null', np.nan)
mr = mr.dropna(how='any')
print('to csv')
mr.to_csv('../input/mr.csv', index=False)

print('mr to csv success!')
# pm data processing
col_names = [
    'city', 'region', 'time', 'cgi', 
    'voice_connection', 'wifi_connection', 'video_connection', 
    'voice_disconnection', 'wifi_disconnection', 'video_disconnection',
    'esrvcc_convert', 'voice_convert_1', 'convert_rate', 'voice_convert_2',
    'voice_push_miss', 'voice_pull_miss', 'voice_pull_delay', 'voice_count', 'data_count',
    'rrc_connection', 'erab_connection', 'erab_trash', 
    'wifi_disconnection_1', 'prb_push', 'prb_pull', 'pdcch_cce', 'rrc_max', 
    'csgb_rrc', 'rrc_2g', 'rrc_3g', 'rrc_num'
]
pm = pd.DataFrame()
for i in date:
    print('read', i)
    d = pd.read_csv('../input/knowyou_pm_2018'+i+'.csv', encoding='gbk', names=col_names, skiprows=1)
    pm = pd.concat([pm, d])
pm = pm.replace('null', np.nan)
pm.to_csv('../input/pm.csv', index=False)

print('to csv')
# merge 
train = pd.merge(pm, mr, on=['time', 'cgi'], how='inner')
train.to_csv('../input/train.csv', index=False)

del mr
del pm
del train
gc.collect()
