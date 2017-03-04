# -*- coding: UTF-8 -*-

import sys
sys.path.append('../_utils')
from _const import _grid_width, _grid_height, _cube_width, _cube_height, inbound, XY2cord, cord2XY, cord2gridXY, gridXY2mapXY
from _colors import _color_red_list, _color_orange_list, get_color

import fileinput
import numpy as np
from pylab import *

def draw_metro_OD_gridmap():
	lnglat_dict = {}
	for line in fileinput.input("../../data/data_exp/metroSpatial.csv"):
		station, lng, lat = line.strip().split(',')
		lnglat_dict[station] = (float(lng), float(lat))
	fileinput.close()
	produce = [{} for h in xrange(24)]
	consume = [{} for h in xrange(24)]
	for line in fileinput.input("../../data/data_toy/2015_04_01_metro.txt"):
		card_id, time_start, loc_start, time_finish, loc_finish = line.strip().split(',')
		loc_start, loc_finish = loc_start.split("号线")[1].strip(), loc_finish.split("号线")[1].strip()
		hour_start, hour_finish = int(time_start.split(' ')[1].split(':')[0]), int(time_finish.split(' ')[1].split(':')[0])
		produce[hour_start][loc_start] = produce[hour_start].get(loc_start,0)+1
		consume[hour_finish][loc_finish] = consume[hour_finish].get(loc_finish,0)+1
	fileinput.close()
	for (_name, _matrix) in [('O', produce),('D', consume)]:
		for _h in xrange(5,24):
			fig, ax = plt.subplots(1,subplot_kw={'xticks': [], 'yticks': []})
			image = plt.imread('../../result/_map_bounds/shanghai_nokia.png')
			ax.imshow(image)
			_min, _max = min(_matrix[_h].values()), max(_matrix[_h].values())
			for _location, _count in _matrix[_h].iteritems():
				_lng, _lat = lnglat_dict[_location]
				if inbound(_lng, _lat):
					circle = plt.Circle(cord2XY(_lng,_lat), 1.*(_count-_min)/(_max-_min)*15+2, fc=_color_orange_list[10]['color'], alpha=0.5, linewidth=0)
					ax.add_patch(circle)
			plt.savefig("../../result/distribution_OD/metro_{}/shanghai_metro_{}_{}h_nokia.png".format(_name,_name,_h))

def draw_taxi_OD_gridmap():
	produce = [[[0 for y in range(_grid_height)] for x in range(_grid_width)] for h in xrange(24)]
	consume = [[[0 for y in range(_grid_height)] for x in range(_grid_width)] for h in xrange(24)]
	for line in fileinput.input("../../data/data_toy/2015_04_01_taxi.txt"):
		_, time_start, loc_start, time_finish, loc_finish, _, _ = line.strip().split(',')
		hour_start, hour_finish = int(time_start.split(' ')[1].split(':')[0]), int(time_finish.split(' ')[1].split(':')[0])
		lng_start, lat_start, lng_finish, lat_finish = float(loc_start.split(' ')[0]), float(loc_start.split(' ')[1]), float(loc_finish.split(' ')[0]), float(loc_finish.split(' ')[1])
		if inbound(lng_start, lat_start):
			gx_start, gy_start = cord2gridXY(lng_start, lat_start)
			produce[hour_start][gx_start][gy_start] += 1
		if inbound(lng_finish, lat_finish):
			gx_finish, gy_finish = cord2gridXY(lng_finish, lat_finish)
			consume[hour_finish][gx_finish][gy_finish] += 1
	fileinput.close()
	for (_name, _matrix) in [('O', produce),('D', consume)]:
		for _h in xrange(24):
			fig, ax = plt.subplots(1,subplot_kw={'xticks': [], 'yticks': []})
			image = plt.imread('../../result/_map_bounds/shanghai_nokia.png')
			ax.imshow(image)
			_slice = np.array(_matrix[_h])
			_min, _max = _slice.min(), _slice.max()
			for x in range(_grid_width):
				for y in range(_grid_height):
					cube_color = get_color(_color_orange_list, _min, _max, _matrix[_h][x][y])
					if cube_color != '#ffffff':
						cube = plt.Rectangle(gridXY2mapXY(x,y), _cube_width, _cube_height, fc=cube_color, alpha=0.6, linewidth=0)
						ax.add_patch(cube)
			plt.savefig("../../result/distribution_OD/taxi_{}/shanghai_taxi_{}_{}h_nokia.png".format(_name,_name,_h))

def draw_metro_gravity_model_coef_gridmap():
	lnglat_dict = {}
	for line in fileinput.input("../../data/data_exp/metroSpatial.csv"):
		station, lng, lat = line.strip().split(',')
		lnglat_dict[station] = (float(lng), float(lat))
	fileinput.close()
	_matrix = {}
	for line in fileinput.input("../../data/statistics/intermediate/gravity_model_coef_metro.txt"):
		_station, _lng, _lat, _coef1, _coef2 = line.strip().split(',')
		_matrix[_station] = float(_coef1)
		# _matrix[_station] = float(_coef2)
	fileinput.close()
	fig, ax = plt.subplots(1,subplot_kw={'xticks': [], 'yticks': []})
	image = plt.imread('../../result/_map_bounds/shanghai_nokia.png')
	ax.imshow(image)
	_min, _max = min(_matrix.values()), max(_matrix.values())
	for _location, _value in _matrix.iteritems():
		_lng, _lat = lnglat_dict[_location]
		if inbound(_lng, _lat):
			circle = plt.Circle(cord2XY(_lng,_lat), 1.*(_value-_min)/(_max-_min)*15+2, fc=_color_orange_list[10]['color'], alpha=0.5, linewidth=0)
			ax.add_patch(circle)
	plt.show()

def draw_taxi_gravity_model_coef_gridmap():
	lnglat_dict = {}
	for line in fileinput.input("../../data/data_exp/taxiSpatial.csv"):
		station, lng, lat = line.strip().split(',')
		lnglat_dict[station] = (float(lng), float(lat))
	fileinput.close()
	_matrix = [[0 for y in range(_grid_height)] for x in range(_grid_width)]
	for line in fileinput.input("../../data/statistics/intermediate/gravity_model_coef_taxi.txt"):
		_station, _lng, _lat, _coef1, _coef2 = line.strip().split(',')
		_gx, _gy = [int(_cord) for _cord in _station.split('_')]
		_matrix[_gx][_gy] = float(_coef1)
		# _matrix[_gx][_gy] = float(_coef2)
	fileinput.close()
	fig, ax = plt.subplots(1,subplot_kw={'xticks': [], 'yticks': []})
	image = plt.imread('../../result/_map_bounds/shanghai_nokia.png')
	ax.imshow(image)
	_slice = np.array(_matrix)
	_min, _max = _slice.min(), _slice.max()
	for x in range(_grid_width):
		for y in range(_grid_height):
			cube_color = get_color(_color_orange_list, _min, _max, _matrix[x][y])
			if cube_color != '#ffffff':
				cube = plt.Rectangle(gridXY2mapXY(x,y), _cube_width, _cube_height, fc=cube_color, alpha=0.6, linewidth=0)
				ax.add_patch(cube)
	plt.show()


if __name__ == "__main__":
	# draw_metro_OD_gridmap()
	# draw_taxi_OD_gridmap()
	draw_metro_gravity_model_coef_gridmap()
	draw_taxi_gravity_model_coef_gridmap()

