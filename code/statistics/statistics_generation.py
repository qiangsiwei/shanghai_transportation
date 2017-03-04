# -*- coding: UTF-8 -*-

import sys
sys.path.append('../_utils')
from _const import _lng_bound, _lat_bound, _grid_width, _grid_height, _cube_width, _cube_height, inbound, XY2cord, cord2XY, cord2gridXY, gridXY2cord, gridXY2mapXY
from _colors import _color_red_list, _color_orange_list, get_color

import json
import math
import time
import pymongo
import fileinput
import itertools
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
from sklearn import linear_model

# # 统计天气信息（降雨量）
# def plot_weather():
# 	day_dict = {}
# 	for line in fileinput.input("../../data/data_exp/weather.csv"):
# 		datetime, _, _, temperature, wind_diraction, wind_speed, rain_drop = line.strip().split(',')
# 		datetime = time.strptime(datetime, "%Y%m%d%H%M")
# 		if datetime.tm_mon == 4:
# 			if not datetime.tm_mday in day_dict:
# 				day_dict[datetime.tm_mday] = {'rain_drop':[]}
# 			if rain_drop != '////':
# 				day_dict[datetime.tm_mday]['rain_drop'].append(float(rain_drop))	
# 	fileinput.close()
# 	for day in sorted(day_dict.keys()):
# 		rain_drop_avg = sum(day_dict[day]['rain_drop'])/len(day_dict[day]['rain_drop'])
# 		print day, rain_drop_avg

# # 生成栅格名称和经纬度对照表
# def generate_gridmap():
# 	with open("../../data/data_exp/taxiSpatial.csv",'w') as _file:
# 		for gx in xrange(_grid_width):
# 			for gy in xrange(_grid_height): 
# 				_lng, _lat = gridXY2cord(gx, gy)
# 				print _lng, _lat
# 				_file.write("{}_{},{},{}\n".format(gx,gy,_lng,_lat))

# 计算经纬度之间欧氏距离
def distance(loc1, loc2):
	import math
	lat1, lng1, lat2, lng2 = loc1[0]/180*math.pi, loc1[1]/180*math.pi, loc2[0]/180*math.pi, loc2[1]/180*math.pi
	R, C = 6400, min(math.sin(lat1)*math.sin(lat2)*math.cos(lng1-lng2)+math.cos(lat1)*math.cos(lat2),1.0)
	distance = R*math.acos(C)
	return "{0:.2f}".format(distance)

# # 为POI建立索引
# def poi_geo_index():
# 	collection = pymongo.MongoClient('localhost', 27017)['Shanghai']['poi']
# 	for doc in collection.find():
# 		doc.update({'position':{
# 						'type': "Point",
# 						'coordinates':[doc['location']['lng'],doc['location']['lat']]
# 					}})
# 		collection.update({'_id':doc['_id']}, doc, upsert=True)

# # 为公交站点建立索引
# def busline_geo_index():
# 	collection = pymongo.MongoClient('localhost', 27017)['Shanghai']['busline']
# 	for line in fileinput.input("../../data/data_exp/buslineSpatial.txt"):
# 		if fileinput.isfirstline() or len(line.strip().split('\t'))!=7:
# 			continue
# 		fullname, name, starttime, endtime, company, stationnum, stations = line.strip().split('\t')
# 		for station in stations.split(' '):
# 			station_name, _location = station.split(':')
# 			doc = {"fullname":fullname,
# 					"name":name,
# 					"starttime":starttime,
# 					"endtime":endtime,
# 					"company":company,
# 					"stationnum":int(stationnum),
# 					"stations":stations,
# 					"station_name":station_name,
# 					'position':{
# 						'type': "Point",
# 						'coordinates':[float(cord) for cord in _location.split(',')]
# 				}}
# 			collection.update({'fullname':fullname,'station_name':station_name},doc,upsert=True)
# 	fileinput.close()

# # 统计经纬度附近POI信息
# def statistic_poi(source, distance):
# 	baidu_tags = ["房地产;住宅区","房地产;写字楼","公司企业","美食","购物","金融","生活服务","休闲娱乐","旅游景点","自然地物","政府机构"]
# 	collection_poi = pymongo.MongoClient('localhost', 27017)['Shanghai']['poi']
# 	collection_busline = pymongo.MongoClient('localhost', 27017)['Shanghai']['busline']
# 	collection_anjuke_xiaoqu = pymongo.MongoClient('localhost', 27017)['Shanghai']['anjuke_xiaoqu_households']
# 	collection_anjuke_loupan = pymongo.MongoClient('localhost', 27017)['Shanghai']['anjuke_loupan_details']
# 	query_poi = {
# 		"position":{
# 			"$near":{
# 				"$geometry":{
# 						"type": "Point",
# 						"coordinates": []
# 					},
# 				"$maxDistance": distance
# 			}
# 		},
# 		"detail_info.tag":{
# 			"$regex":""
# 		}
# 	}
# 	query_anjuke = {
# 		"position":{
# 			"$near":{
# 				"$geometry":{
# 						"type": "Point",
# 						"coordinates": []
# 					},
# 				"$maxDistance": distance
# 			}
# 		},
# 	}
# 	query_busline = query_anjuke
# 	with open('../../data/statistics/intermediate/nearby_poi_statistics_{}.csv'.format(source),'w') as _file:
# 		_file.write('name,count\n')
# 		_index = 0
# 		for line in fileinput.input("../../data/data_exp/{}Spatial.csv".format(source)):
# 			_index += 1
# 			print _index
# 			station, lng, lat = line.strip().split(',')
# 			lng, lat = float(lng), float(lat)
# 			count_list = []
# 			for key_word in baidu_tags:
# 				query_poi['position']['$near']['$geometry']['coordinates'] = [lng, lat]
# 				query_poi['detail_info.tag']['$regex'] = r'.*'+key_word+r'.*'
# 				count_poi = collection_poi.find(query_poi).count()
# 				count_list.append(str(count_poi))
# 			query_anjuke['position']['$near']['$geometry']['coordinates'] = [lng, lat]
# 			count_households = 0
# 			for _item in collection_anjuke_xiaoqu.find(query_anjuke):
# 				try:
# 					count_households += int(_item[u'总户数'].replace(u'暂无数据',u'0户').split(u'户')[0])
# 				except:
# 					continue
# 			count_list.append(str(count_households))
# 			count_loupanarea = 0
# 			for _item in collection_anjuke_loupan.find(query_anjuke):
# 				count_loupanarea += _item['area']
# 			count_list.append(str(int(count_loupanarea)))
# 			query_busline = query_anjuke
# 			count_list.append(str(collection_busline.find(query_busline).count()))
# 			print count_list
# 			_file.write("{},{}\n".format(station.strip(),",".join(count_list)))
# 		fileinput.close()

def get_lnglat_dict(source, get_str=False):
	lnglat_dict = {}
	for line in fileinput.input("../../data/data_exp/{}Spatial.csv".format(source)):
		station, lng, lat = line.strip().split(',')
		if not get_str:
			lnglat_dict[station] = (float(lng), float(lat))
		else:
			lnglat_dict[station] = "{},{}".format(lng,lat)
	fileinput.close()
	return lnglat_dict

# 统计经纬度对应单日出入交通总量
def statistic_preprocessing(source):
	lnglat_dict = get_lnglat_dict(source)
	_src_dict, _dst_dict, _OD_dict = {}, {}, {}
	for line in fileinput.input("../../data/statistics/total_count_by_day_hour_src_dst_{}.txt".format(source)):
		day, hour, location_begin, location_end, count = line.strip().split(',')
		location_begin, location_end = location_begin.strip(), location_end.strip()
		_key_src = (location_begin,day)
		_key_dst = (location_end,day)
		_key_OD = (location_begin,location_end,day)
		_src_dict[_key_src] = _src_dict.get(_key_src,0)+int(count)
		_dst_dict[_key_dst] = _dst_dict.get(_key_dst,0)+int(count)
		_OD_dict[_key_OD] = _OD_dict.get(_key_OD,0)+int(count)
	fileinput.close()
	for _name, _dict in [('statistics_{}_src.csv'.format(source), _src_dict),
						 ('statistics_{}_dst.csv'.format(source), _dst_dict)]:
		with open('../../data/statistics/intermediate/{}'.format(_name),'w') as _file:
			_file.write('name,day,lng,lat,count\n')
			for (_location, _day), _count in _dict.iteritems():
				_lng, _lat = lnglat_dict[_location]
				_file.write("{},{},{},{},{}\n".format(_location,_day,_lng,_lat,_count))
	with open('../../data/statistics/intermediate/statistics_{}_OD.csv'.format(source),'w') as _file:
		_file.write('src,dst,day,dist,count\n')
		for (_src, _dst, _day), _count in _OD_dict.iteritems():
			_dist = distance(lnglat_dict[_src], lnglat_dict[_dst])
			_file.write("{},{},{},{},{}\n".format(_src,_dst,_day,_dist,_count))

# # 统计经纬度对应单日出入交通总量与POI数量分布的关系
# def draw_statistic_generation_poi_correlation(source, filename, date=1):
# 	lnglat_dict = get_lnglat_dict(source)
# 	station_dict = {}
# 	for line in fileinput.input("../../data/statistics/intermediate/{}.csv".format(filename)):
# 		if fileinput.isfirstline():
# 			continue
# 		station, day, _, _, count = line.strip().split(',')
# 		if int(day) == date:
# 			station_dict[station] = {'travel_count':int(count),'poi_distribution':[]}
# 	fileinput.close()
# 	for line in fileinput.input("../../data/statistics/intermediate/nearby_poi_statistics_{}.csv".format(source)):
# 		if fileinput.isfirstline():
# 			continue
# 		station = line.strip().split(',')[0]
# 		poi_distribution = [int(count) for count in line.strip().split(',')[1:]]
# 		if station in station_dict:
# 			station_dict[station]['poi_distribution'] = poi_distribution
# 	fileinput.close()
# 	X, Y, LNG, LAT = [], [], [], []
# 	for _station, _value in station_dict.iteritems():
# 		_lng, _lat = lnglat_dict[_station][0], lnglat_dict[_station][1]
# 		if _lng_bound[0]<_lng<_lng_bound[1] and _lat_bound[0]<_lat<_lat_bound[1]:
# 			LNG.append(_lng)
# 			LAT.append(_lat)
# 			X.append(_value['poi_distribution'])
# 			Y.append(_value['travel_count'])
# 	clf = linear_model.LinearRegression()
# 	clf.fit(X, Y)
# 	print clf.score(X, Y), clf.coef_
# 	_Y = clf.predict(X)
# 	ERR = [abs(_Y[i]-Y[i])/Y[i] for i in xrange(len(Y))]
# 	plt.scatter(LNG, LAT, s=ERR, c='red', alpha=0.5)
# 	plt.plot([_lng_bound[0],_lng_bound[0]], [_lat_bound[0],_lat_bound[1]], 'r--') #左
# 	plt.plot([_lng_bound[1],_lng_bound[1]], [_lat_bound[0],_lat_bound[1]], 'r--') #右
# 	plt.plot([_lng_bound[0],_lng_bound[1]], [_lat_bound[0],_lat_bound[0]], 'r--') #下
# 	plt.plot([_lng_bound[0],_lng_bound[1]], [_lat_bound[1],_lat_bound[1]], 'r--') #上
# 	plt.show()
# 	plt.scatter(_Y, Y, c='black', alpha=0.5)
# 	plt.show()

# 统计经纬度之间对应单日出入交通总量与距离的关系(重力模型)
def draw_statistic_distribution_distance_correlation(source, weight_normalize=100, xlim=[0,1], ylim=[0,1], date=1):
	lnglat_dict = get_lnglat_dict(source,get_str=True)
	_src_dict, _dst_dict = {}, {}
	for _name, _dict in [('src',_src_dict),('dst',_dst_dict)]:
		for line in fileinput.input("../../data/statistics/intermediate/statistics_{}_{}.csv".format(source,_name)):
			if fileinput.isfirstline():
				continue
			station, day, _, _, count = line.strip().split(',')
			if int(day) == date:
				_dict[station] = {'travel_count':float(count)}
		fileinput.close()
	_station_dict = {}
	for line in fileinput.input("../../data/statistics/intermediate/statistics_{}_OD.csv".format(source)):
		if fileinput.isfirstline():
			continue
		src, dst, day, dist, count = line.strip().split(',')
		if int(day) == 1 and float(dist) != 0:
			volumn_src, volumn_dst = _src_dict[src]['travel_count'], _dst_dict[dst]['travel_count']
			if not src in _station_dict:
				_station_dict[src] = {'src':{'X':[],'Y':[],'V':volumn_src/weight_normalize}, 'dst':{'X':[],'Y':[],'V':volumn_src/weight_normalize}}
			if not dst in _station_dict:
				_station_dict[dst] = {'src':{'X':[],'Y':[],'V':volumn_dst/weight_normalize}, 'dst':{'X':[],'Y':[],'V':volumn_dst/weight_normalize}}
			_station_dict[src]['src']['X'].append([math.log(volumn_dst), math.log(float(dist))])
			_station_dict[src]['src']['Y'].append(math.log(float(count)))
			_station_dict[dst]['dst']['X'].append([math.log(volumn_src), math.log(float(dist))])
			_station_dict[dst]['dst']['Y'].append(math.log(float(count)))
	fileinput.close()

	subplot_i = 0
	plt.figure(figsize=(12,5))
	with open("../../data/statistics/intermediate/gravity_model_coef_{}.txt".format(source),'w') as _file:
		alpha, beta, weight = {'src':[],'dst':[]}, {'src':[],'dst':[]}, {'src':[],'dst':[]}
		for _station, _value in _station_dict.iteritems():
			for _role in ['src','dst']:
				X, Y = _value[_role]['X'], _value[_role]['Y']
				if len(X) == 0 or len(Y) == 0:
					continue
				clf = linear_model.LinearRegression()
				clf.fit(X, Y)
				_Y = clf.predict(X)
				alpha[_role].append(clf.coef_[0])
				beta[_role].append(clf.coef_[1])
				weight[_role].append(_value[_role]['V'])
				_file.write("{},{},{}\n".format(_station,lnglat_dict[_station],','.join([str(_coef) for _coef in clf.coef_])))
		_file.close()
		for _role, _title in [('src','{0} (Source)'.format(source.capitalize())),('dst','{0} (Destination)'.format(source.capitalize()))]:
			subplot_i += 1
			subplot(1,2,subplot_i)
			clf = linear_model.LinearRegression()
			clf.fit([[c] for c in alpha[_role]], [[c] for c in beta[_role]], sample_weight=weight[_role])
			plt.scatter(alpha[_role], beta[_role], s=weight[_role], c='black', edgecolors='white', alpha=0.3)
			# plt.plot(np.reshape([[xlim[0]],[xlim[1]]],(2,)),np.reshape(clf.predict([[xlim[0]],[xlim[1]]]),(2,)),'r--',linewidth=2,label="fitting")
			plt.axis([xlim[0], xlim[1], ylim[0], ylim[1]])
			plt.title(_title)
			if subplot_i == 1:
				plt.xlabel('Volumn coeffient')
				plt.ylabel('Distance coeffient')
	plt.show()
	# for postfix in ('eps','png'):
	# 	plt.savefig('../../figure/{0}/02.{0}'.format(postfix))

# # 为交通数据建立索引
# def source_geo_index():
# 	# 'metro',
# 	for _source in ['taxi']:
# 		collection = pymongo.MongoClient('localhost', 27017)['Shanghai']['traffic_{}'.format(_source)]
# 		lnglat_dict = get_lnglat_dict(_source)
# 		_src_dict, _dst_dict, _OD_dict = {}, {}, {}
# 		for line in fileinput.input("../../data/statistics/total_count_by_day_hour_src_dst_{}.txt".format(_source)):
# 			day, hour, location_begin, location_end, count = line.strip().split(',')
# 			if int(day) <= 15:
# 				location_begin, location_end = location_begin.strip(), location_end.strip()
# 				position_begin, position_end = lnglat_dict[location_begin], lnglat_dict[location_end]
# 				_doc = {
# 					"location_begin":location_begin,
# 					"location_end":location_end,
# 					"position_begin":position_begin,
# 					"position_end":position_end,
# 					"day":int(day),
# 					"hour":int(hour),
# 					"count":int(count)
# 				}
# 				collection.insert(_doc)
# 		fileinput.close()

# 统计经纬度交通量随时间分布
def get_temporal_traffic_by_location(source, station, date=1):
	collection = pymongo.MongoClient('localhost', 27017)['Shanghai']['traffic_{}'.format(source)]
	for _location_key, _color in [('location_begin','r'),('location_end','b')]:
		print itertools.groupby(collection.find({_location_key:station,'day':date}), lambda x:x['hour'])

		_result = [(_hour, sum([_entry['count'] for _entry in _group])) 
						for _hour, _group in itertools.groupby(
								sorted(collection.find({_location_key:station,'day':date}),key=lambda x:x['hour']), 
							lambda x:x['hour'])]
		print _result
		line, = plt.plot([_e[0] for _e in _result], [_e[1] for _e in _result], '{}-'.format(_color), linewidth=2)
	plt.show()

# 统计经纬度之间交通量随时间分布
def get_temporal_traffic_by_OD(source, src, dst, date=1):
	collection = pymongo.MongoClient('localhost', 27017)['Shanghai']['traffic_{}'.format(source)]
	for _src, _dst, _color in [(src,dst,'r'),(dst,src,'b')]:
		_result = [(_hour, sum([_entry['count'] for _entry in _group])) 
						for _hour, _group in itertools.groupby(
							collection.find({'location_begin':_src,'location_end':_dst,'day':date}), 
							lambda x:x['hour'])]
		line, = plt.plot([_e[0] for _e in _result], [_e[1] for _e in _result], '{}-'.format(_color), linewidth=2)
	plt.show()

# 统计经纬度之间交通量随时间分布的平均绝对误差
def get_temporal_traffic_by_OD_compute_MAE(filename):
	import gzip
	_dict_hour = {}
	for line in gzip.open("../../data/statistics/intermediate/{}".format(filename)):
		_hour, _, _, _len, _mean, _mae, _, _ = line.strip().split('\t')
		if not _hour in _dict_hour:
			_dict_hour[_hour] = {'total_volumn':0,'total_error':0}
		_dict_hour[_hour]['total_volumn'] += int(_len)*float(_mean)
		_dict_hour[_hour]['total_error'] += int(_len)*float(_mae)
	for _hour in xrange(24):
		if str(_hour) in _dict_hour:
			total_volumn = _dict_hour[str(_hour)]['total_volumn']
			total_error = _dict_hour[str(_hour)]['total_error']
			print _hour, total_volumn, total_error, round(total_error/total_volumn,4)


if __name__ == "__main__":
	# plot_weather()
	# generate_gridmap()
	# poi_geo_index()
	# busline_geo_index()
	# statistic_poi('metro',1000)
	# statistic_poi('taxi',500)

	# statistic_preprocessing('metro')
	# statistic_preprocessing('taxi')

	# draw_statistic_generation_poi_correlation('metro','statistics_metro_src', 1)
	# draw_statistic_generation_poi_correlation('taxi','statistics_taxi_src', 1)
	
	# draw_statistic_distribution_distance_correlation('metro',weight_normalize=200,xlim=[0,1.5],ylim=[-2.0,1.0])
	draw_statistic_distribution_distance_correlation('taxi',weight_normalize=20,xlim=[0,0.8],ylim=[-1.6,0.0])
	
	# source_geo_index()
	# get_temporal_traffic_by_location('metro','鞍山新村',1)
	# get_temporal_traffic_by_OD('metro','鞍山新村','江湾体育场',1)
	# get_temporal_traffic_by_location('taxi',cord2gridXY(121.510258,31.282839,get_str=True),1)
	# get_temporal_traffic_by_OD_compute_MAE("MAE_metro_workdays.txt.gz")
	# get_temporal_traffic_by_OD_compute_MAE("MAE_metro_holidays.txt.gz")
	# get_temporal_traffic_by_OD_compute_MAE("MAE_taxi_workdays.txt.gz")
	# get_temporal_traffic_by_OD_compute_MAE("MAE_taxi_holidays.txt.gz")

