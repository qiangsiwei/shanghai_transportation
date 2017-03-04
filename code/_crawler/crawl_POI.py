# -*- coding: UTF-8 -*-

import json
import pymongo
import urllib2
import numpy as np
from pylab import *

'''
百度标签分类:
	"房地产","公司企业","美食","购物","金融","生活服务","休闲娱乐","旅游景点","自然地物","政府机构"
	"丽人","医疗","教育","运动健身"
	"宾馆","结婚","汽车服务","培训机构"
	"门址","道路","交通线","交通设施","行政区划"
'''

collection = pymongo.MongoClient('localhost', 27017)['Shanghai']['poi']

def crawl_POI():
	baidu_tags = ["房地产","公司企业","美食","购物","金融","生活服务","休闲娱乐","旅游景点","自然地物","政府机构"]
	apikey = "hIlimGuvEfHV41Aw885gONzB"
	# 经纬度范围:121.0~122.0, 30.7~31.5
	lng_min, lat_min, lng_gap, lat_gap = 121.0, 30.7, 0.005, 0.005
	for tag in baidu_tags:
		for gx in xrange(200):
			for gy in xrange(160):
				page_num, page_num_max = 0, 20
				while page_num < page_num_max:
					lng_begin, lat_begin, lng_end, lat_end = lng_min+gx*lng_gap, lat_min+gy*lat_gap, lng_min+(gx+1)*lng_gap, lat_min+(gy+1)*lat_gap
					bounds = str(lat_begin)+","+str(lng_begin)+","+str(lat_end)+","+str(lng_end)
					url = "http://api.map.baidu.com/place/v2/search?&query="+tag+"&bounds="+bounds+"&scope=2&output=json&page_size=20&page_num="+str(page_num)+"&ak="+apikey
					data = json.load(urllib2.urlopen(url, timeout=30))
					if data["status"] != 0:
						exit()
					if page_num == 0:
						total = data["total"]
						page_num_max = (total-1)/20+1
					for result in data["results"]:
						if 'uid' in result:
							collection.update({'uid':result['uid']},result,upsert=True)
					print data["status"], tag, gx, gy, page_num, data["total"]
					page_num += 1

def draw_POI_distribution():
	gx_num, gy_num,  = 100, 80
	baidu_tags = ["房地产;住宅区","房地产;写字楼","公司企业","美食","购物","金融","生活服务","休闲娱乐","旅游景点","自然地物","政府机构"]
	for key_word in baidu_tags:
		tag_dict, total = {}, 0
		grid = [[0 for y in range(gy_num)] for x in range(gx_num)]
		grid_price = [[[] for y in range(gy_num)] for x in range(gx_num)]
		for result in collection.find({'detail_info.tag':{'$regex':r'.*'+key_word+'.*'}}): 
			for tag in result['detail_info']['tag'].strip().split(';'):
				tag_dict.update({tag:tag_dict.get(tag,0)+1})
			total += 1
			lng, lat = result['location']['lng'], result['location']['lat']
			gx, gy = int(gx_num*(float(lng)-121.2)/(121.7-121.2)), int(gy_num*(float(lat)-31.0)/(31.4-31.0))
			if 0<=gx<gx_num and 0<=gy<gy_num:
				grid[gx][gy] += 1
				if 'price' in result['detail_info']:
					try:
						if int(float(result['detail_info']['price'])) != 0:
							grid_price[gx][gy].append(float(result['detail_info']['price']))
					except:
						pass
		print "{}:{}".format(key_word, total)
		for (k,v) in sorted([(k,v) for k,v in tag_dict.items() if k.encode('utf-8')!=key_word], key=lambda x:x[1], reverse=True):
			print '\t{}:{}'.format(k.encode('utf-8'),v)
		(X, Y), C = meshgrid(np.arange(gx_num), np.arange(gy_num)), np.array(grid)
		cset = pcolormesh(X, Y, C.T, cmap=cm.get_cmap("OrRd"))
		colorbar(cset)
		axis('off')
		show()
		grid_price = [[sum(grid_price[x][y])/len(grid_price[x][y]) if len(grid_price[x][y])!=0 else 0 for y in range(gy_num)] for x in range(gx_num)]
		(X, Y), C = meshgrid(np.arange(gx_num), np.arange(gy_num)), np.array(grid_price)
		cset = pcolormesh(X, Y, C.T, cmap=cm.get_cmap("OrRd"))
		colorbar(cset)
		axis('off')
		show()


if __name__ == "__main__":
	crawl_POI()
	draw_POI_distribution()

