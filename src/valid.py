# coding=utf-8
import lightgbm as lgb
from sklearn import metrics
from sklearn.model_selection import train_test_split
import gc
from processing import DataSet

dataset = DataSet()
dataset.load()

print('train test split')
trainx, validx, trainy, validy = train_test_split(
    dataset.train_x, 
    dataset.train_y_low, 
    test_size=0.1, 
    random_state=432423
)

print('make dataset')
train_data = lgb.Dataset(trainx, trainy, categorical_feature=dataset.cat_f)
valid_data = lgb.Dataset(validx, validy, categorical_feature=dataset.cat_f)

params = {
#     'application': 'binary',
    'boosting': 'gbdt',
    'num_leaves': 80,
    'min_data_in_leaf': 50,
    'learning_rate': 0.05,
    'zero_as_missing': True,
    'lambda_l1': 0,
    'lambda_l2': 0,
    'metric':{'mse'}
}

print('start train')
model = lgb.train(
    params, train_data, 10000,
    valid_sets=[train_data, valid_data],
    valid_names=['train', 'valid'],
    verbose_eval=50,
    early_stopping_rounds=100
)

print('make predict')
pre = model.predict(dataset.test_x)

valid_auc = metrics.mean_squared_error(dataset.test_y_low, pre)
print(valid_auc)

'''
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (15.0, 8.0) # 显示大小
lgb.plot_importance(booster=model)


def draw(real, pre, skip=0, n=10):
    al = []
    for i in range(n):
        al.append(i)

    ax1 = plt.subplot(221)
    ax1.set_title('pre')   
    ax1.plot(al, pre[skip:skip+n], 'r', label='pre')
    ax1.legend(bbox_to_anchor=[1, 1])  
    ax1.grid() 

    ax2 = plt.subplot(222)
    ax2.set_title('real')   
    ax2.plot(al, real[skip:skip+n].tolist(), 'b', label='real')
    ax2.legend(bbox_to_anchor=[1, 1])  
    ax2.grid() 

    ax3 = plt.subplot(212)
    ax3.set_title('pre & real')   
    ax3.plot(al, pre[skip:skip+n], 'r', label='pre')
    ax3.plot(al, real[skip:skip+n].tolist(), 'b', label='real')
    ax3.legend(bbox_to_anchor=[1, 1])  
    ax3.grid() 

    plt.show()
    
draw(dataset.test_y_low, pre, skip=226, n=300)
'''
