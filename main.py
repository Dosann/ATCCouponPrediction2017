# -*- coding: utf-8 -*-
'''
create time: 2017/9/22 18:24
author: duxin
site: 
email: duxin_be@outlook.com
'''

import numpy as np, pandas as pd
import copy
import time
from tools import dataload, prefix, generateCustomDetails, generateMerchtDetails, train, prediction, saveResult


[tbOflOri, tbOflpredOri] = dataload(folderpath = u'D://Workfiles/ATCBigData/201709CouponPrediction/data', filename = ['ccf_offline_stage1_train.csv', 'ccf_offline_stage1_test_revised.csv'])
#[tbOflOri, tbOnlOri, tbOflpredOri] = dataload(folderpath = u'../data/ATCCouponPrediction', ['ccf_offline_stage1_train.csv', 'ccf_online_stage1_train.csv', 'ccf_offline_stage1_test_revised.csv'])




# offline train集数据预处理
tbOfl = copy.copy(tbOflOri)
tbOfl = prefix(data = tbOfl, mode = 'train')
#print('tbOfl: ', tbOfl)

# offline test集数据预处理
tbOflpred = copy.copy(tbOflpredOri)
tbOflpred = prefix(data = tbOflpred, mode = 'test')
#print('tbOflpred', tbOflpred)

# 顾客特征DataFrame
customs = generateCustomDetails(tbOfl)
# 店铺特征DataFrame
merchts = generateMerchtDetails(tbOfl)
# 联合特征DataFrame
cusMer = generateCusMerDetails(tbOfl)

# 训练模型
model = train(customs, merchts, tbOfl)

# offline test集结果预测
[trainResult, testResult] = prediction(model, trainData = tbOfl, testData = tbOflpred)
# 保存结果
saveResult(result = result, filefolder = 'result/')
"""

