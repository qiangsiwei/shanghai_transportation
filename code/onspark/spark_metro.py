# -*- coding: utf-8 -*- 

import sys
from operator import add
from pyspark import SparkConf,SparkContext

def extract_raw(line):
	parts = line.split(',')
	uid, timestamp, location, mode, status = parts[0], "{0} {1}".format(parts[1],parts[2]), parts[3], parts[4], parts[5]
	if mode == u"地铁":
		return (uid, (status, timestamp, location))
	else:
		return ""

def generate_trip(uid, sequence):
	sequence = sorted(list(sequence), key=lambda item: item[1])
	records, time_start, time_finish, loc_start, loc_finish = [], None, None, None, None
	for (status, timestamp, location) in sequence:
		if status == "0.00" and time_start == None and loc_start == None:
			time_start, loc_start = timestamp, location
		if status != "0.00" and time_start != None and loc_start != None:
			time_finish, loc_finish = timestamp, location
			records.append(','.join([time_start, loc_start, time_finish, loc_finish]).encode('utf-8'))
			time_start, time_finish, loc_start, loc_finish = None, None, None, None
	return (uid, records)

if __name__ == "__main__":
	conf = SparkConf().setMaster('yarn-client') \
					  .setAppName('Shanghai_metro') \
					  .set('spark.driver.maxResultSize', "10g")
	sc = SparkContext(conf=conf)
	def f(x): return x
	lines = sc.textFile('hdfs://namenode.omnilab.sjtu.edu.cn/user/soda/一卡通乘客刷卡数据/SPTCC-20150430.csv', minPartitions=100)
	# .sample(False, 0.1, 50)\
	counts = lines.map(lambda line : extract_raw(line))\
				  .filter(lambda x : x!="")\
				  .groupByKey()\
				  .map(lambda (uid,sequence) : generate_trip(uid,sequence))\
				  .flatMapValues(f)\
				  .map(lambda tp : "{0},{1}".format(tp[0],tp[1]))
	output = counts.saveAsTextFile("./Shanghai/metro/2015_04_30")
