# -*- coding: UTF-8 -*-

import re
import pymongo
import urllib2
import fileinput

def crawl_anjuke_xiaoqu():
	_file = open("crawl_anjuke_xiaoqu_xiaoqu.txt",'w')
	for _page in xrange(1944):
		print _page
		url = "http://shanghai.anjuke.com/community/W0QQpZ"+str(_page)
		data = urllib2.urlopen(urllib2.Request(url), timeout=60).read().replace('\r\n','').replace('\n','')
		for _url, _name, _location in re.findall(r'<li class="list_item">.*?<a href="(.+?)" title="(.+?)".+?<a href="http://shanghai.anjuke.com/map/sale/#(.*?)".*?</li>', data):
			_location = ','.join([_para.split('=')[1] for _para in _location.split('&')[:2]])
			_file.write("{}\t{}\t{}\n".format(_name, _url, _location))
	_file.close()

def crawl_anjuke_xiaoqu_households(skip=0):
	collection = pymongo.MongoClient('localhost', 27017)['Shanghai']['anjuke_xiaoqu_households']
	_index = -1
	for line in fileinput.input("crawl_anjuke_xiaoqu_xiaoqu.txt"):
		_index += 1
		if _index < skip:
			continue
		print _index
		_name, _url, _location = line.strip().split('\t')
		data = urllib2.urlopen(urllib2.Request(_url), timeout=60).read().replace('\r\n','').replace('\n','')
		try:
			for _content in re.findall(r'<dl class="comm-r-detail float-r">(.+?)</dl>', data):
				doc = {"name":_name,
					"url":_url,
					'position':{
						'type': "Point",
						'coordinates':[float(cord) for cord in _location.split(',')][::-1]
					}	
				}
				doc.update({_key:_value for _key, _value in \
							re.findall(r'<dt>(.+?)</dt><dd.*?>(.+?)</dd>', _content)})
				collection.update({'url':_url},doc,upsert=True)
		except:
			continue
	fileinput.close()

def crawl_anjuke_loupan():
	_file = open("crawl_anjuke_xiaoqu_loupan.txt",'w')
	for _page in xrange(382):
		print _page
		url = "http://sh.xzl.anjuke.com/loupan/p"+str(_page)+'/'
		data = urllib2.urlopen(urllib2.Request(url), timeout=60).read().replace('\r\n','').replace('\n','')
		for _url, _name in re.findall(r'<div class="bdl_mic_nrjj">.*?<a target="_blank" href="(.+?)">(.+?)</a>', data):
			_url, _name = _url.strip(), _name.strip()
			if _url != "" and _name != "":
				_file.write("{}\t{}\n".format(_name, _url))
	_file.close()

def crawl_anjuke_loupan_details(skip=0):
	collection = pymongo.MongoClient('localhost', 27017)['Shanghai']['anjuke_loupan_details']
	_index = -1
	for line in fileinput.input("crawl_anjuke_xiaoqu_loupan.txt"):
		_index += 1
		if _index < skip:
			continue
		print _index
		_name, _url = line.strip().split('\t')
		data = urllib2.urlopen(urllib2.Request(_url), timeout=60).read().replace('\r\n','').replace('\n','')		
		try:
			_location = re.findall(r'http://api\.map\.baidu\.com/staticimage\?center=(.*?)&', data) or ["0,0"]
			_location = _location[0]
			_content = re.sub('&nbsp;','',re.sub(r' +',' ',re.sub(r'<.*?>','',re.findall(r'<table class="config">(.+?)</table>', data)[0]))).strip()
			_area = re.findall(r'总建筑面积：(.*?)平米', _content) or ["0"]
			_area = float(_area[0])
			doc = {"name":_name,
					"url":_url,
					'position':{
						'type': "Point",
						'coordinates':[float(cord) for cord in _location.split(',')]
					},
					"content":_content,
					"area":_area
				}
			collection.update({'url':_url},doc,upsert=True)
		except:
			continue
	fileinput.close()


if __name__ == "__main__":
	crawl_anjuke_xiaoqu()
	crawl_anjuke_xiaoqu_households(19350)
	crawl_anjuke_loupan()
	crawl_anjuke_loupan_details(3037)

