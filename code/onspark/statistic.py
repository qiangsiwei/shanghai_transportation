# -*- coding: utf-8 -*- 

import sys
import time
from operator import add
from pyspark import SparkConf,SparkContext

def extract_raw_metro(line):
	uid, time_begin, location_begin, time_end, location_end = line.strip().split(',')
	time_begin = time.strptime(time_begin,"%Y-%m-%d %H:%M:%S")
	time_end = time.strptime(time_end,"%Y-%m-%d %H:%M:%S")
	location_begin = location_begin.split("号线")[1]
	location_end = location_end.split("号线")[1]
	day, hour = time_begin.tm_mday, time_begin.tm_hour
	return ((day, hour, location_begin, location_end), 1)

def extract_raw_taxi(line):
	_lng_bound = [121.308,121.598]
	_lat_bound = [31.146,31.306]
	_grid_width, _grid_height = 28, 24
	uid, time_begin, location_begin, time_end, location_end, _, _ = line.strip().split(',')
	time_begin = time.strptime(time_begin,"%Y-%m-%d %H:%M:%S")
	time_end = time.strptime(time_end,"%Y-%m-%d %H:%M:%S")
	lng_start, lat_start, lng_finish, lat_finish = float(location_begin.split(' ')[0]), float(location_begin.split(' ')[1]), float(location_end.split(' ')[0]), float(location_end.split(' ')[1])
	if _lng_bound[0]<lng_start<_lng_bound[1] and _lat_bound[0]<lat_start<_lat_bound[1]\
	and _lng_bound[0]<lng_finish<_lng_bound[1] and _lat_bound[0]<lat_finish<_lat_bound[1]:
		gx_start, gy_start = int(_grid_width*(lng_start-_lng_bound[0])/(_lng_bound[1]-_lng_bound[0])), int(_grid_height*(lat_start-_lat_bound[0])/(_lat_bound[1]-_lat_bound[0]))
		gx_finish, gy_finish = int(_grid_width*(lng_finish-_lng_bound[0])/(_lng_bound[1]-_lng_bound[0])), int(_grid_height*(lat_finish-_lat_bound[0])/(_lat_bound[1]-_lat_bound[0]))
		location_begin = "{0}_{1}".format(gx_start, gy_start)
		location_end = "{0}_{1}".format(gx_finish, gy_finish)
		day, hour = time_begin.tm_mday, time_begin.tm_hour
		return ((day, hour, location_begin, location_end), 1)
	else:
		return ""

def extract_statistics(line):
	holidays = [4,5,6,11,12,18,19,25,26]
	day, hour, location_begin, location_end, count = line.strip().split(',')
	# if not int(day) in holidays:
	if int(day) in holidays:
		return ((hour, location_begin, location_end), int(count))
	else:
		return ""

def compute_MAE(_key, _list):
	_key = '\t'.join(_key)
	_len = len(_list)
	_mean = round(1.*sum(_list)/_len,2)
	_list = [abs(_entry-_mean) for _entry in _list]
	_mae = round(1.*sum(_list)/_len,2)
	_err = round(_mae/_mean,4)
	return "{0}\t{1}\t{2}\t{3}\t{4}\t{5}".format(_key,_len,_mean,_mae,_err,','.join([str(_e) for _e in _list]))

if __name__ == "__main__":
	conf = SparkConf().setMaster('yarn-client') \
					  .setAppName('Shanghai_metro') \
					  .set('spark.driver.maxResultSize', "10g")
	sc = SparkContext(conf=conf)
	def f(x): return x

	lines = sc.textFile('hdfs://namenode.omnilab.sjtu.edu.cn/user/qiangsiwei/Shanghai/data/metro/', minPartitions=100, use_unicode=False)
	counts = lines.map(lambda line : extract_raw_metro(line))\
				  .reduceByKey(add)\
				  .sortByKey()\
				  .map(lambda ((day, hour, location_begin, location_end), count) : "{0},{1},{2},{3},{4}".format(day,hour,location_begin,location_end,count))\
				  .coalesce(1)
	output = counts.saveAsTextFile("./Shanghai/statistics/total_count_by_day_hour_src_dst/metro")

	lines = sc.textFile('hdfs://namenode.omnilab.sjtu.edu.cn/user/qiangsiwei/Shanghai/data/taxi/', minPartitions=100, use_unicode=False)
	counts = lines.map(lambda line : extract_raw_taxi(line))\
				  .filter(lambda x: x!="")\
				  .reduceByKey(add)\
				  .sortByKey()\
				  .map(lambda ((day, hour, location_begin, location_end), count) : "{0},{1},{2},{3},{4}".format(day,hour,location_begin,location_end,count))\
				  .coalesce(1)
	output = counts.saveAsTextFile("./Shanghai/statistics/total_count_by_day_hour_src_dst/taxi")

	lines = sc.textFile('hdfs://namenode.omnilab.sjtu.edu.cn/user/qiangsiwei/Shanghai/statistics/total_count_by_day_hour_src_dst/metro/', minPartitions=100, use_unicode=False)
	counts = lines.map(lambda line : extract_statistics(line))\
				  .filter(lambda x: x!="")\
				  .groupByKey()\
				  .map(lambda (_key, _list) : compute_MAE(_key, _list))\
				  .coalesce(1)
	# output = counts.saveAsTextFile("./Shanghai/statistics/MAE/metro_workdays")
	output = counts.saveAsTextFile("./Shanghai/statistics/MAE/metro_holidays")

	lines = sc.textFile('hdfs://namenode.omnilab.sjtu.edu.cn/user/qiangsiwei/Shanghai/statistics/total_count_by_day_hour_src_dst/taxi/', minPartitions=100, use_unicode=False)
	counts = lines.map(lambda line : extract_statistics(line))\
				  .filter(lambda x: x!="")\
				  .groupByKey()\
				  .map(lambda (_key, _list) : compute_MAE(_key, _list))\
				  .coalesce(1)
	# output = counts.saveAsTextFile("./Shanghai/statistics/MAE/taxi_workdays")
	output = counts.saveAsTextFile("./Shanghai/statistics/MAE/taxi_holidays")



