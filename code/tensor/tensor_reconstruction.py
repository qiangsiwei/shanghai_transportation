# -*- coding: UTF-8 -*-

import os
import sys
import gzip
import json
import math
import pymongo
import fileinput
import numpy as np
sys.path.append('./scikit-tensor')
from sktensor import dtensor
from _ncp import *
from pylab import *
from scipy import interpolate
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

_grid_width, _grid_height = 28, 24
_lng_bound = [121.308,121.598]; _lat_bound = [31.146,31.306]


def get_lnglat_dict(src):
	lnglat_dict = {}
	for idx, line in enumerate(fileinput.input('../../data/data_exp/{0}Spatial.csv'.format(src))):
		station, lng, lat = line.strip().split(',')
		lnglat_dict[station] = (idx, float(lng), float(lat))
	fileinput.close()
	return lnglat_dict


def get_tensor(src='metro', daytype='workdays', n_component=5):
	lnglat_dict = get_lnglat_dict(src=src); travel = {}; test = {}
	locations = {ind:name for name,(ind,_,_) in lnglat_dict.iteritems()}
	for line in gzip.open('../../data/statistics/intermediate/MAE_{0}_{1}.txt.gz'.format(src,daytype)):
		_time, _src, _dst, _len, _mean = line.split()[:5]
		_lng_src, _lat_src = lnglat_dict[_src][1:]; _lng_dst, _lat_dst = lnglat_dict[_dst][1:]
		travel[_src] = travel.get(_src,{}); travel[_src][_dst] = travel[_src].get(_dst,[0]*24)
		travel[_src][_dst][int(_time)] = float(_mean)
		test[_src] = test[_dst] = 1
	tensor = [[[travel.get(locations[i],{}).get(locations[j],[0]*24)[k]\
				 for k in xrange(24)] for j in xrange(len(locations))] for i in xrange(len(locations))]
	np.save('../../data/tensor_data3/tensor_{0}_{1}.npy'.format(src,daytype),np.array(tensor))
	tensor = [[[100.0*tensor[i][j][k]/sum(tensor[i][j]) if sum(tensor[i][j])!=0 else 0 \
				for k in range(24)] for j in range(len(tensor[0]))] for i in range(len(tensor))]
	np.save('../../data/tensor_data3/tensor_normalized_{0}_{1}.npy'.format(src,daytype),np.array(tensor))
	T = dtensor(tensor)
	# tensor = np.array(tensor); approx = nonnegative_tensor_factorization(T, n_component, method='anls_asgroup', min_iter=20, max_iter=50).totensor()
	# print abs(tensor-approx).sum()/tensor.sum()
	matrix_src, matrix_dst, matrix_time = nonnegative_tensor_factorization(T, n_component, method='anls_asgroup', min_iter=20, max_iter=50).U
	matrix_time = np.transpose(matrix_time)
	matrix_file = '../../data/tensor_data3/matrix_{0}_{1}/{2}.npy'.format(src,daytype,'{0}')
	for _matrix, _name in [(matrix_src,'src'),(matrix_dst,'dst'),(matrix_time,'time')]: np.save(matrix_file.format(_name),_matrix)


def get_geo_feature(src='metro'):
	db = pymongo.MongoClient('localhost', 27017)['Shanghai']
	db_poi, db_busline, db_xiaoqu, db_loupan = db['poi'], db['busline'], db['anjuke_xiaoqu_households'], db['anjuke_loupan_details']
	query_poi = {
		'position':{'$near':{
				'$geometry':{'type':'Point','coordinates':[]},
				'$maxDistance': 0}},
		'detail_info.tag':{'$regex':''}}
	query_busline = {
		'position':{'$near':{
				'$geometry':{'type':'Point','coordinates':[]},
				'$maxDistance': 0}}}
	geo_feature = []
	for ind, line in enumerate(fileinput.input('../../data/data_exp/{}Spatial.csv'.format(src))):
		count_list = []; print ind
		station, lng, lat = line.strip().split(','); lng, lat = float(lng), float(lat)
		for distance in (200, 500, 1000):
			for key_word in ['房地产;住宅区','房地产;写字楼','公司企业','美食','购物','金融','生活服务','休闲娱乐','旅游景点','自然地物','政府机构']:
				query_poi['detail_info.tag']['$regex'] = r'.*{0}.*'.format(key_word)
				query_poi['position']['$near']['$geometry']['coordinates'] = [lng,lat]
				query_poi['position']['$near']['$maxDistance'] = distance
				count_list.append(db_poi.find(query_poi).count())
			query_busline['position']['$near']['$geometry']['coordinates'] = [lng,lat]
			query_busline['position']['$near']['$maxDistance'] = distance
			count_list.append(db_busline.find(query_busline).count())
		geo_feature.append(count_list)
	fileinput.close()
	np.save('../../data/tensor_data3/geo_feature_{0}.npy'.format(src),np.array(geo_feature))


def poi_feature(dumpfile='../../data/tensor_data3/poi_distribution.npy'):
	x_num, y_num = 100, 80
	tags = ['房地产;住宅区','公司企业','购物','生活服务','休闲娱乐','旅游景点']
	if not os.path.isfile(dumpfile):
		statistics = []
		for tag in tags:
			grid = [[0 for y in range(y_num)] for x in range(x_num)]
			for entry in pymongo.MongoClient('localhost', 27017)['Shanghai']['poi'].find({'detail_info.tag':{'$regex':r'{0}(;.*)?'.format(tag)}}): 
				x = int(x_num*(float(entry['location']['lng'])-121.2)/(121.7-121.2))
				y = int(y_num*(float(entry['location']['lat'])-31.0)/(31.4-31.0))
				if 0<=x<x_num and 0<=y<y_num: grid[x][y] += 1
			statistics.append(grid)
		statistics = np.array(statistics); np.save(dumpfile,statistics)
	else:
		statistics = np.load(dumpfile)

	tags = map(lambda x:x.split(';')[-1],tags)
	fig = plt.figure(figsize=(16,8))
	fig.subplots_adjust(left=0.05,right=0.98,top=0.95,bottom=0.05,wspace=0.12,hspace=0.24)
	avg = 1.*statistics.sum(axis=0)/statistics.sum()
	for c, i in enumerate(range(len(statistics))):
		diff = 1.*statistics[i]/statistics.sum(axis=(1,2))[i]-avg
		levels = arange(-diff.max(),diff.max(),0.1*diff.max()); norm = cm.colors.Normalize(vmin=-diff.max(),vmax=diff.max())
		subplot(2,3,c+1)
		(X, Y), C = meshgrid(np.arange(x_num), np.arange(y_num)), diff
		cset = pcolormesh(X,Y,C.T,cmap=cm.get_cmap('bwr',len(levels)),norm=norm); colorbar(cset)
		if c == 0:plt.xlabel('Longitude grid index /500m');plt.ylabel('Latitude grid index /500m')
		plt.title(tags[c].decode('utf-8')); plt.axis([0,x_num-1,0,y_num-1])
	for postfix in ('eps','png'): plt.savefig('../../figure/{0}/09.{0}'.format(postfix))


def distance_feature():
	from collections import defaultdict
	from sklearn import linear_model
	for travel_mode in ['metro', 'taxi']:
		src_volumn, dst_volumn, travel_volumn, travel_dist = {}, {}, defaultdict(dict), defaultdict(dict)
		for line in fileinput.input('../../data/statistics/intermediate/statistics_{0}_OD.csv'.format(travel_mode)):
			if fileinput.isfirstline(): continue
			src, dst, day, dist, count = line.strip().split(',')
			if int(day) == 1:
				src_volumn[src] = src_volumn.get(src,0)+int(count)
				dst_volumn[dst] = dst_volumn.get(dst,0)+int(count)
				travel_volumn[src][dst] = int(count); travel_dist[src][dst] = float(dist)
		fileinput.close()
		X, Y = [], []
		for src, volumn_src in src_volumn.iteritems():
			for dst, volumn_trave in travel_volumn[src].iteritems():
				if travel_dist[src][dst] > 0:
					X.append([math.log(src_volumn[src]),math.log(dst_volumn[dst]),math.log(travel_dist[src][dst])])
					Y.append(math.log(volumn_trave))
		clf = linear_model.LinearRegression()
		clf.fit(X, Y)
		print travel_mode, len(Y), clf.coef_


def get_distance(lnglat_dict, locations, i, j, R=6400):
	(lng1,lat1), (lng2,lat2) = lnglat_dict[locations[i]][:2], lnglat_dict[locations[j]][:2]
	return R*math.acos(min(math.sin(lat1)*math.sin(lat2)*math.cos(lng1-lng2)+math.cos(lat1)*math.cos(lat2),1.0))


def travel_volumn_prediction(src='metro', daytype='workdays', valmodel='model1', min_volumn=30):
	'''
		Flow Reconstruction
	'''
	lnglat_dict = get_lnglat_dict(src=src)
	locations = {ind:name for name,(ind,_,_) in lnglat_dict.iteritems()}
	tensor = np.load('../../data/tensor_data3/tensor_{0}_{1}.npy'.format(src,daytype))
	tensor_normalized = np.load('../../data/tensor_data3/tensor_normalized_{0}_{1}.npy'.format(src,daytype))
	matrix_src = np.load('../../data/tensor_data3/matrix_{0}_{1}/src.npy'.format(src,daytype))
	matrix_dst = np.load('../../data/tensor_data3/matrix_{0}_{1}/dst.npy'.format(src,daytype))
	geo_feature = np.load('../../data/tensor_data3/geo_feature_{0}.npy'.format(src))
	geo_feature = ((1.*geo_feature/geo_feature.sum(axis=0)).swapaxes(0,1)-1.*geo_feature.sum(axis=1)/geo_feature.sum()).swapaxes(0,1)

	X1, X2, X3, X4, X5, Y = [], [], [], [], [], []
	for i in xrange(len(locations)):
		for j in xrange(len(locations)):
			if i != j and tensor[i,j].sum() >= min_volumn: 
				X1.append(matrix_src[i]); X2.append(matrix_dst[j]); X3.append(geo_feature[i]); X4.append(geo_feature[j]); X5.append([get_distance(lnglat_dict,locations,i,j)/10**3])
				Y.append(tensor_normalized[i,j].reshape(-1,1))
	X1, X2, X3, X4, X5, Y = np.array(X1), np.array(X2), np.array(X3), np.array(X4), np.array(X5), np.array(Y)

	from keras.models import Sequential
	from keras.layers.core import RepeatVector, TimeDistributedDense
	from keras.layers import Dense, recurrent
	from keras.layers import Merge
	# from keras.utils.visualize_util import plot
	from sklearn.cross_validation import KFold

	RNN = recurrent.GRU
	input1 = Sequential(); input1.add(Dense(32, activation='relu', input_dim=5))
	input2 = Sequential(); input2.add(Dense(32, activation='relu', input_dim=5))
	input3 = Sequential(); input3.add(Dense(32, activation='relu', input_dim=36))
	input4 = Sequential(); input4.add(Dense(32, activation='relu', input_dim=36))
	input5 = Sequential(); input5.add(Dense(4, activation='relu', input_dim=1))
	merged1 = Merge([input1,input2], mode='mul')
	merged2 = Merge([input3,input4], mode='concat')
	
	def model1(): # tensor
		model = Sequential()
		model.add(merged1)
		model.add(Dense(64, activation='relu'))
		model.add(Dense(64, activation='relu'))
		model.add(RepeatVector(24))
		model.add(RNN(64, activation='relu', return_sequences=True))
		model.add(RNN(64, activation='relu', return_sequences=True, go_backwards=True))
		model.add(TimeDistributedDense(1))
		return model

	def model2(): # poi + distance
		model = Sequential()
		model.add(Merge([merged2,input5], mode='concat'))
		model.add(Dense(64, activation='relu'))
		model.add(Dense(64, activation='relu'))
		model.add(RepeatVector(24))
		model.add(RNN(64, activation='relu', return_sequences=True))
		model.add(RNN(64, activation='relu', return_sequences=True, go_backwards=True))
		model.add(TimeDistributedDense(1))
		return model

	def model3(): # tensor + poi + distance
		model = Sequential()
		model.add(Merge([merged1,merged2,input5], mode='concat'))
		model.add(Dense(64, activation='relu'))
		model.add(Dense(64, activation='relu'))
		model.add(RepeatVector(24))
		model.add(RNN(64, activation='relu', return_sequences=True))
		model.add(RNN(64, activation='relu', return_sequences=True, go_backwards=True))
		model.add(TimeDistributedDense(1))
		return model
	# plot(model, to_file='../../data/tensor_data3/model.png')
	if valmodel == 'model1':
		error = 0
		for train, test in KFold(len(Y), n_folds=2):
			model = model1()
			model.compile(loss='mean_absolute_error', optimizer='adam')
			model.fit([X1[train],X2[train]], Y[train], nb_epoch=10, show_accuracy=True)
			Yp = model.predict([X1[test],X2[test]])
			error += abs(Y[test]-Yp).sum()
		print error/Y.sum()
	if valmodel == 'model2':
		error = 0
		for train, test in KFold(len(Y), n_folds=2):
			model = model2()
			model.compile(loss='mean_absolute_error', optimizer='adam')
			model.fit([X3[train],X4[train],X5[train]], Y[train], nb_epoch=10, show_accuracy=True)
			Yp = model.predict([X3[test],X4[test],X5[test]])
			error += abs(Y[test]-Yp).sum()
		print error/Y.sum()
	if valmodel == 'model3':
		error = 0
		for train, test in KFold(len(Y), n_folds=2):
			model = model3()
			model.compile(loss='mean_absolute_error', optimizer='adam')
			model.fit([X1[train],X2[train],X3[train],X4[train],X5[train]], Y[train], nb_epoch=10, show_accuracy=True)
			Yp = model.predict([X1[test],X2[test],X3[test],X4[test],X5[test]])
			error += abs(Y[test]-Yp).sum()
		print error/Y.sum()


if __name__ == '__main__':
	# lnglat_dict_taxi = get_lnglat_dict(src='taxi')
	# get_tensor(src='metro', daytype='workdays')
	# get_tensor(src='metro', daytype='holidays')
	# get_tensor(src='taxi', daytype='workdays')
	# get_tensor(src='taxi', daytype='holidays')
	# get_geo_feature(src='metro')
	# get_geo_feature(src='taxi')
	# poi_feature()
	# distance_feature()
	# travel_volumn_prediction(src='metro', daytype='workdays', valmodel='model1', min_volumn=100)
	# travel_volumn_prediction(src='metro', daytype='workdays', valmodel='model2', min_volumn=100)
	# travel_volumn_prediction(src='metro', daytype='workdays', valmodel='model3', min_volumn=100)
	# travel_volumn_prediction(src='metro', daytype='holidays', valmodel='model1', min_volumn=50)
	# travel_volumn_prediction(src='metro', daytype='holidays', valmodel='model2', min_volumn=50)
	# travel_volumn_prediction(src='metro', daytype='holidays', valmodel='model3', min_volumn=50)
	# travel_volumn_prediction(src='taxi', daytype='workdays', valmodel='model1', min_volumn=15)
	# travel_volumn_prediction(src='taxi', daytype='workdays', valmodel='model2', min_volumn=15)
	# travel_volumn_prediction(src='taxi', daytype='workdays', valmodel='model3', min_volumn=15)
	# travel_volumn_prediction(src='taxi', daytype='holidays', valmodel='model1', min_volumn=20)
	# travel_volumn_prediction(src='taxi', daytype='holidays', valmodel='model2', min_volumn=20)
	# travel_volumn_prediction(src='taxi', daytype='holidays', valmodel='model3', min_volumn=20)
	# travel_volumn_prediction(src='taxi', daytype='workdays')

