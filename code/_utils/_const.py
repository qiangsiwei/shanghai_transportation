# -*- coding: UTF-8 -*-

_metro_station_num = 288
_lng_bound = [121.308,121.598]
_lat_bound = [31.146,31.306]
_map_width, _map_height = 850, 550
_grid_width, _grid_height = 28, 24
_cube_width, _cube_height = 24, 20

def inbound(lng, lat):
	if _lng_bound[0]<lng<_lng_bound[1] and _lat_bound[0]<lat<_lat_bound[1]:
		return True
	else:
		return False

# 经纬度和地图坐标转换
def XY2cord(x, y):
	lng = 1.*x/_map_width*(_lng_bound[1]-_lng_bound[0])+_lng_bound[0]
	lat = _lat_bound[1]-1.*y/_map_height*(_lat_bound[1]-_lng_bound[0])
	return lng, lat

def cord2XY(lng, lat):
	x = (lng-_lng_bound[0])/(_lng_bound[1]-_lng_bound[0])*_map_width
	y = (_lat_bound[1]-lat)/(_lat_bound[1]-_lat_bound[0])*_map_height
	return int(x), int(y)

# 经纬度到栅格坐标转换
def cord2gridXY(lng, lat, get_str=False):
	x = (lng-_lng_bound[0])/(_lng_bound[1]-_lng_bound[0])*_grid_width
	y = (lat-_lat_bound[0])/(_lat_bound[1]-_lat_bound[0])*_grid_height
	if not get_str:
		return int(x), int(y)
	else:
		return "{}_{}".format(int(x),int(y))

def gridXY2cord(x, y):
	lng = 1.*(x+0.5)/_grid_width*(_lng_bound[1]-_lng_bound[0])+_lng_bound[0]
	lat = 1.*(y+0.5)/_grid_height*(_lat_bound[1]-_lat_bound[0])+_lat_bound[0]
	return lng, lat

# 栅格坐标到地图坐标转换
def gridXY2mapXY(x, y):
	return 30*x+10, 23*(_grid_height-1-y)

