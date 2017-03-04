# -*- coding: utf-8 -*- 

import sys
from operator import add
from pyspark import SparkConf,SparkContext

def distance(loc1, loc2):
	import math
	lat1, lng1, lat2, lng2 = loc1[0]/180*math.pi, loc1[1]/180*math.pi, loc2[0]/180*math.pi, loc2[1]/180*math.pi
	R, C = 6400, min(math.sin(lat1)*math.sin(lat2)*math.cos(lng1-lng2)+math.cos(lat1)*math.cos(lat2),1.0)
	distance = R*math.acos(C)
	return "{0:.2f}".format(distance)

def extract_raw(line):
	parts = line.split(',')
	tid, status, timestamp, location = parts[0], parts[2], parts[7], "{0} {1}".format(parts[8],parts[9])
	return (tid, (status, timestamp, location))

def generate_trip(tid, sequence):
	import time
	sequence = sorted(list(sequence), key=lambda item: item[1])
	records, time_start, time_finish, loc_start, loc_finish = [], None, None, None, None
	for (status, timestamp, location) in sequence:
		if status == "0":
			if time_start == None and loc_start == None:
				time_start, loc_start = timestamp, location
			else:
				time_finish, loc_finish = timestamp, location
		elif status == "1":
			if time_start != None and loc_start != None and time_finish != None and loc_finish != None:
				try:
					delt = time.mktime(time.strptime(time_finish,"%Y-%m-%d %H:%M:%S")) - time.mktime(time.strptime(time_start,"%Y-%m-%d %H:%M:%S"))
					dist = distance([float(i) for i in loc_start.split(' ')], [float(i) for i in loc_finish.split(' ')])
					records.append(','.join([time_start, loc_start, time_finish, loc_finish, str(delt), dist]))
				except:
					pass
				time_start, time_finish, loc_start, loc_finish = None, None, None, None
			else:
				continue
	if time_start != None and loc_start != None and time_finish != None and loc_finish != None:
		try:
			delt = time.mktime(time.strptime(time_finish,"%Y-%m-%d %H:%M:%S")) - time.mktime(time.strptime(time_start,"%Y-%m-%d %H:%M:%S"))
			dist = distance([float(i) for i in loc_start.split(' ')], [float(i) for i in loc_finish.split(' ')])
			records.append(','.join([time_start, loc_start, time_finish, loc_finish, str(delt), dist]))
		except:
			pass
	return (tid, records)

if __name__ == "__main__":
	conf = SparkConf().setMaster('yarn-client') \
					  .setAppName('Shanghai_taxi') \
					  .set('spark.driver.maxResultSize', "10g")
	sc = SparkContext(conf=conf)
	def f(x): return x
	lines = sc.textFile('hdfs://namenode.omnilab.sjtu.edu.cn/user/soda/强生出租车行车数据/30', minPartitions=100)
	# .sample(False, 0.1, 50)\
	counts = lines.map(lambda line : extract_raw(line))\
				  .groupByKey()\
				  .map(lambda (tid,sequence) : generate_trip(tid,sequence))\
				  .flatMapValues(f)\
				  .map(lambda tp : "{0},{1}".format(tp[0],tp[1]))
	output = counts.saveAsTextFile("./Shanghai/taxi/2015_04_30")
