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

##### 数据读取
def dataload(folderpath, filename = ['ccf_offline_stage1_train.csv',  'ccf_online_stage1_train.csv', 'ccf_offline_stage1_test_revised.csv']):
    tsDataload = time.ctime()
    print('数据读取开始: %s'%tsDataload)

    if type(filename) == str:
        filename = [filename]
    data = []
    for filen in filename:
        data.append( pd.read_csv(filepath_or_buffer=folderpath + '/' + filen) )
        print('fileload finished: %s'%filen)

    teDataload = time.ctime()
    print('数据读取结束: %s'%teDataload)
    return data

##### 预处理
def prefixNull2nan(data, columns, dateColumns):
    for columnName in columns:
        data.loc[data[columnName] == 'null', columnName] = np.nan
    for columnName in dateColumns:
        data[columnName] = pd.to_datetime(data[columnName])

def prefixHasCoupon(data):
    data = data.loc[~pd.isnull(data['Coupon_id']), :]
    return data

def prefixNullOfDistance(data, method = 'delete'):
# 处理Distance字段为null的情形
    boolIsNull = pd.isnull(data['Distance'])

    if method == 'delete':
        data = data.loc[~boolIsNull, :]
    elif method == 'average':
        averDistance = np.mean(pd.to_numeric(data['Distance']))
        data.loc[boolIsNull, 'Distance'] = averDistance
    elif method == 'setZero':
        data.loc[boolIsNull, 'Distance'] = 0

    return data

def featureDiscountRate(data, delDiscountRate = 'yes'):
# 处理Discount_rate字段数据
    dataLength = data.shape[0]
    columns = ['DiscThreshd', 'DiscVolume', 'DiscRate']
    discTemp = np.zeros([data.shape[0],3])
    ioDR = list(data.columns).index('Discount_rate')

    for i in range(data.shape[0]):
        temp = data.iloc[i, ioDR].split(':')
        if len(temp) == 1:
            discTemp[i, 0] = 0.0
            discTemp[i, 1] = 0.0
            discTemp[i, 2] = 1 - float(temp[0])
        else:
            discTemp[i, 0] = float(temp[0])
            discTemp[i, 1] = float(temp[1])
            discTemp[i, 2] = float(temp[1])/(float(temp[0]) + 1e-3)
    for i in range(len(columns)):
        data[columns[i]] = discTemp[:, i]

    if delDiscountRate =='yes':
        del data['Discount_rate']
    return data

def featureCouponUsage(data):
# 添加每张优惠券的使用情况描述
    data['UsedCouponWI15days'] = pd.to_timedelta(np.array((data['Date'] - data['Date_received']))).days <=15
    return data

def prefix(data, mode = 'train'):
    ######################################################
    tsPrefix = time.ctime()
    print('数据清洗开始: %s'%tsPrefix)

    # 将所有'null'变为np.nan
    if mode == 'train':
        columns_PN2N = ['Coupon_id', 'Discount_rate', 'Distance', 'Date_received','Date']
        dateColumns_PN2N = ['Date_received','Date']
    else:
        columns_PN2N = ['Discount_rate', 'Distance']
        dateColumns_PN2N = ['Date_received']
    prefixNull2nan(data = data, columns = columns_PN2N, dateColumns = dateColumns_PN2N)
    # 筛选出获得了优惠券的顾客
    data = prefixHasCoupon(data = data)
    # 对Distance为null的顾客进行预处理
    if mode == 'train':
        method_PNOD = 'delete'
    else:
        method_PNOD = 'average'
    data = prefixNullOfDistance(data = data, method = method_PNOD)
    # 对Discount_rate进行预处理
    data = featureDiscountRate(data = data)
    # 添加每张优惠券的使用情况描述
    if mode == 'train':
        data = featureCouponUsage(data = data)
    #print(tbOfl)

    tePrefix = time.ctime()
    print('数据清洗结束: %s'%tePrefix)
    #####################################################

    return data

##### 生成特征

def generateCustomDetails(data):
    gcustomOfl = data.groupby('User_id')
    gcustomOflIndSet = gcustomOfl.indices
    gcustomOflInd = list(gcustomOflIndSet)
    gcustomOflCount = gcustomOfl.ngroups

    # 顾客收优惠券频次
    cusGetCouponCount = np.array(gcustomOfl.count().iloc[:, 0], dtype = int)

    # 顾客使用优惠券频次
    cusUseCouponCount = np.zeros([gcustomOflCount], dtype = int)
    i = 0
    arrOfl_UsedCouponWI15days = np.array(data['UsedCouponWI15days'])

    for key,value in gcustomOflIndSet.items():
        cusUseCouponCount[i] = arrOfl_UsedCouponWI15days[value].sum()
        i += 1
    del(arrOfl_UsedCouponWI15days)
    cusUseCouponAL1T = cusUseCouponCount > 0
    nCusUseCouponAL1T = np.sum(cusUseCouponAL1T)

    # 顾客使用优惠券频率
    cusUseCouponRate = np.zeros([gcustomOflCount])
    cusUseCouponRate = map(lambda x,y:y == 0 and 0 or x/y, cusUseCouponCount.reshape(-1).tolist(), cusGetCouponCount.reshape(-1).tolist())
    cusUseCouponRate = np.array(list(cusUseCouponRate))

    # 顾客使用优惠券折扣比例 - 最低 平均 最高
    cusLowestDiscRate = np.zeros([gcustomOflCount])
    cusAverDiscRate = np.zeros([gcustomOflCount])
    cusHighestDiscRate = np.zeros([gcustomOflCount])
    i = 0
    arrOfl_DiscRate = np.array(data['DiscRate'], dtype = float)
    for key,value in gcustomOflIndSet.items():
        discRate = arrOfl_DiscRate[value]
        cusLowestDiscRate[i] = np.min(discRate)
        cusAverDiscRate[i] = np.mean(discRate)
        cusHighestDiscRate[i] = np.max(discRate)
        i += 1
    cusLowestDiscRate_aver = np.sum(cusUseCouponAL1T * cusLowestDiscRate) / nCusUseCouponAL1T
    cusAverDiscRate_aver = np.sum(cusUseCouponAL1T * cusAverDiscRate) / nCusUseCouponAL1T
    cusHighestDiscRate_aver = np.sum(cusUseCouponAL1T * cusHighestDiscRate) / nCusUseCouponAL1T
    cusLowestDiscRate[~cusUseCouponAL1T] = cusLowestDiscRate_aver
    cusAverDiscRate[~cusUseCouponAL1T] = cusAverDiscRate_aver
    cusHighestDiscRate[~cusUseCouponAL1T] = cusHighestDiscRate_aver
    del(arrOfl_DiscRate)

    # 顾客使用优惠券起步消费额 - 最低 平均 最高
    cusLowestDiscThreshd = np.zeros([gcustomOflCount])
    cusAverDiscThreshd = np.zeros([gcustomOflCount])
    cusHighestDiscThreshd = np.zeros([gcustomOflCount])
    i = 0
    arrOfl_DiscThreshd = np.array(data['DiscThreshd'], dtype = float)
    for key,value in gcustomOflIndSet.items():
        discThreshd = arrOfl_DiscThreshd[value]
        cusLowestDiscThreshd[i] = np.min(discThreshd)
        cusAverDiscThreshd[i] = np.mean(discThreshd)
        cusHighestDiscThreshd[i] = np.max(discThreshd)
        i += 1
    cusLowestDiscThreshd_aver = np.sum(cusUseCouponAL1T * cusLowestDiscThreshd) / nCusUseCouponAL1T
    cusAverDiscThreshd_aver = np.sum(cusUseCouponAL1T * cusAverDiscThreshd) / nCusUseCouponAL1T
    cusHighestDiscThreshd_aver = np.sum(cusUseCouponAL1T * cusHighestDiscThreshd) / nCusUseCouponAL1T
    cusLowestDiscThreshd[~cusUseCouponAL1T] = cusLowestDiscThreshd_aver
    cusAverDiscThreshd[~cusUseCouponAL1T] = cusAverDiscThreshd_aver
    cusHighestDiscThreshd[~cusUseCouponAL1T] = cusHighestDiscThreshd_aver
    del(arrOfl_DiscThreshd)

    # 特征合成为一个DataFrame
    customs = pd.DataFrame(index = gcustomOflInd,
                           data = {'cusGetCouponCount': cusGetCouponCount,
                                   'cusUseCouponCount': cusUseCouponCount,
                                   'cusUseCouponRate': cusUseCouponRate,
                                   'cusLowestDiscRate': cusLowestDiscRate,

                                   'cusAverDiscRate': cusAverDiscRate,
                                   'cusHighestDiscRate': cusHighestDiscRate,
                                   'cusLowestDiscThreshd': cusLowestDiscThreshd,
                                   'cusAverDiscThreshd': cusAverDiscThreshd,
                                   'cusHighestDiscThreshd': cusHighestDiscThreshd})
    return customs

def generateMerchtDetails(data):
    gmerchtOfl = data.groupby('Merchant_id')
    gmerchtOflIndSet = gmerchtOfl.indices
    gmerchtOflInd = list(gmerchtOflIndSet)
    gmerchtOflCount = gmerchtOfl.ngroups

    # 店铺发出优惠券频次
    merCouponCount = np.array(gmerchtOfl.count().iloc[:, 0], dtype = int)

    # 店铺发出优惠券的使用频次
    merCouponUsedCount = np.zeros([gmerchtOflCount], dtype = int)
    i = 0
    arrOfl_UsedCouponWI15days = np.array(data['UsedCouponWI15days'])
    for key,value in gmerchtOflIndSet.items():
        merCouponUsedCount[i] = arrOfl_UsedCouponWI15days[value].sum()
        i += 1
    del(arrOfl_UsedCouponWI15days)
    merCouponUsedAL1T = merCouponUsedCount > 0
    nMerCouponUsedAL1T = np.sum(merCouponUsedAL1T)

    # 店铺发出优惠券的使用比例
    merCouponUsedRate = np.zeros([gmerchtOflCount])
    merCouponUsedRate = map(lambda x,y:y == 0 and 0 or x/y, merCouponUsedCount.reshape(-1).tolist(), merCouponCount.reshape(-1).tolist())
    merCouponUsedRate = np.array(list(merCouponUsedRate))

    # 店铺发出优惠券(使用过的)折扣: 最低 平均 最高
    merLowestDiscRate = np.zeros([gmerchtOflCount])
    merAverDiscRate = np.zeros([gmerchtOflCount])
    merHighestDiscRate = np.zeros([gmerchtOflCount])
    i = 0
    arrOfl_DiscRate = np.array(data['DiscRate'], dtype = float)
    for key,value in gmerchtOflIndSet.items():
        discRate = arrOfl_DiscRate[value]
        merLowestDiscRate[i] = np.min(discRate)
        merAverDiscRate[i] = np.mean(discRate)
        merHighestDiscRate[i] = np.max(discRate)
        i += 1
    merLowestDiscRate_aver = np.sum(merCouponUsedAL1T * merLowestDiscRate) / nMerCouponUsedAL1T
    merAverDiscRate_aver = np.sum(merCouponUsedAL1T * merAverDiscRate) / nMerCouponUsedAL1T
    merHighestDiscRate_aver = np.sum(merCouponUsedAL1T * merHighestDiscRate) / nMerCouponUsedAL1T
    merLowestDiscRate[~merCouponUsedAL1T] = merLowestDiscRate_aver
    merAverDiscRate[~merCouponUsedAL1T] = merAverDiscRate_aver
    merHighestDiscRate[~merCouponUsedAL1T] = merHighestDiscRate_aver
    del(arrOfl_DiscRate)

    # 店铺发出优惠券(使用过的)起步消费额: 最低 平均 最高
    merLowestDiscThreshd = np.zeros([gmerchtOflCount])
    merAverDiscThreshd = np.zeros([gmerchtOflCount])
    merHighestDiscThreshd = np.zeros([gmerchtOflCount])
    i = 0
    arrOfl_DiscThreshd = np.array(data['DiscThreshd'], dtype = float)
    for key,value in gmerchtOflIndSet.items():
        discThreshd = arrOfl_DiscThreshd[value]
        merLowestDiscThreshd[i] = np.min(discThreshd)
        merAverDiscThreshd[i] = np.mean(discThreshd)
        merHighestDiscThreshd[i] = np.max(discThreshd)
        i += 1
    merLowestDiscThreshd_aver = np.sum(merCouponUsedAL1T * merLowestDiscThreshd) / nMerCouponUsedAL1T
    merAverDiscThreshd_aver = np.sum(merCouponUsedAL1T * merAverDiscThreshd) / nMerCouponUsedAL1T
    merHighestDiscThreshd_aver = np.sum(merCouponUsedAL1T * merHighestDiscThreshd) / nMerCouponUsedAL1T
    merLowestDiscThreshd[~merCouponUsedAL1T] = merLowestDiscThreshd_aver
    merAverDiscThreshd[~merCouponUsedAL1T] = merAverDiscThreshd_aver
    merHighestDiscThreshd[~merCouponUsedAL1T] = merHighestDiscThreshd_aver
    del(arrOfl_DiscThreshd)

        # 特征合成为一个DataFrame
    merchts = pd.DataFrame(index = gmerchtOflInd,
                           data = {'merCouponCount': merCouponCount,
                                   'merCouponUsedCount': merCouponUsedCount,
                                   'merCouponUsedRate': merCouponUsedRate,
                                   'merLowestDiscRate': merLowestDiscRate,
                                   'merAverDiscRate': merAverDiscRate,
                                   'merHighestDiscRate': merHighestDiscRate,
                                   'merLowestDiscThreshd': merLowestDiscThreshd,
                                   'merAverDiscThreshd': merAverDiscThreshd,
                                   'merHighestDiscThreshd': merHighestDiscThreshd})
    return merchts