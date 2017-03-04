# -*- coding: UTF-8 -*-

import sys
sys.path.append('../_utils')
from _const import _metro_station_num, _lng_bound, _lat_bound, _grid_width, _grid_height, _cube_width, _cube_height, inbound, XY2cord, cord2XY, cord2gridXY, gridXY2cord, gridXY2mapXY
from _colors import _color_red_list, _color_yellow_list, _color_green_list, get_color

import json
import math
import time
import gzip
import fileinput

def get_lnglat_dict(src, get_str=False):
	lnglat_dict = {}
	for line in fileinput.input("../../data/data_exp/{0}Spatial.csv".format(src)):
		station, lng, lat = line.strip().split(',')
		if not get_str:
			lnglat_dict[station] = (float(lng), float(lat))
		else:
			lnglat_dict[station] = "{0},{1}".format(lng,lat)
	fileinput.close()
	return lnglat_dict

def generate_tensor(src, daytype):
	lnglat_dict = get_lnglat_dict(src)
	OD_statistics, location_slot_dict, _index = {}, {}, 0
	for line in gzip.open("../../data/statistics/intermediate/MAE_{0}_{1}.txt.gz".format(src,daytype)):
		_hour, _location_begin, _location_end, _len, _mean, _mae, _, _ = line.strip().split('\t')
		_location_begin, _location_end = _location_begin.strip(), _location_end.strip()
		_lng_start, _lat_start = lnglat_dict[_location_begin]
		_lng_end, _lat_end = lnglat_dict[_location_end]
		if inbound(_lng_start,_lat_start) and inbound(_lng_end,_lat_end):
			OD_statistics[(_location_begin,_location_end,_hour)] = float(_mean)
			for _location in [_location_begin, _location_end]:
				if not _location in location_slot_dict:
					location_slot_dict[_location] = _index
					_index += 1
	_location_slot_num = len(location_slot_dict)

	with open("../../data/tensor_data2/tensor_index_{0}.txt".format(src),'w') as _file:
		for _location, _index in location_slot_dict.iteritems():
			_file.write("{0}\t{1}\n".format(_index,_location))

	tensor_data = [[[0 for h in range(24)] for y in range(_location_slot_num)] for x in range(_location_slot_num)]
	for (_location_begin,_location_end,_hour), _total in OD_statistics.iteritems():
		tensor_data[location_slot_dict[_location_begin]][location_slot_dict[_location_end]][int(_hour)] = _total
		# tensor_data[location_slot_dict[_location_begin]][location_slot_dict[_location_end]][int(_hour)] = math.log(_total,2)

	with open("../../data/tensor_data1/tensor_{0}_{1}.txt".format(src,daytype),'w') as _file:
		_file.write(json.dumps(tensor_data))

	tensor_data = [[[100.0*tensor_data[x][y][h]/sum(tensor_data[x][y]) if sum(tensor_data[x][y])!=0 else 0 for h in range(24)] for y in range(_location_slot_num)] for x in range(_location_slot_num)]
	with open("../../data/tensor_data2/tensor_{0}_{1}.txt".format(src,daytype),'w') as _file:
		_file.write(json.dumps(tensor_data))

def plot_volumn_statistics():
	import numpy as np
	from pylab import *
	from scipy import interpolate
	from matplotlib.ticker import MultipleLocator, FormatStrFormatter

	fig = plt.figure(figsize=(8,5))
	ax = fig.add_subplot(111)
	for subplot, (src, daytype, linestyle) in enumerate([('metro','workdays','r-'),('metro','holidays','g-'),('taxi','workdays','r:'),('taxi','holidays','g:')]):  
		tensor_abs = np.array(json.loads(open("../../data/tensor_data1/tensor_{0}_{1}.txt".format(src,daytype)).read()))
		hour_count = [tensor_abs[:,:,i].sum() for i in xrange(24)]
		tck = interpolate.splrep(range(24),hour_count,s=0)
		x = np.arange(0,23,0.1)
		y = interpolate.splev(x,tck,der=0)
		plt.plot(x,y,linestyle,label='{0} {1}'.format(src.capitalize(),daytype))
		plt.xlim(0,24-1)
		plt.ylim(0,5.5*10**5)
		plt.xlabel('Time /hour')
		plt.ylabel('Count')
	handles, labels = ax.get_legend_handles_labels()
	ax.legend(handles,labels,loc=1)
	xmajorLocator = MultipleLocator(1)
	xmajorFormatter = FormatStrFormatter('%d')
	ax.xaxis.set_major_locator(xmajorLocator)
	ax.xaxis.set_major_formatter(xmajorFormatter)
	# show()
	for postfix in ('eps','png'):
		plt.savefig('../../figure/{0}/03.{0}'.format(postfix))

def tensor_statistics():
	from pylab import *

	fig = plt.figure(figsize=(8,5))
	ax1 = fig.add_subplot(111)
	for _src, _daytype, _linetype in [('metro','workdays','r-'), ('metro','holidays','b-'), ('taxi','workdays','r:'), ('taxi','holidays','b:')]: # workdays
		tensor_data = json.loads(open("../../data/tensor_data1/tensor_{0}_{1}.txt".format(_src,_daytype)).read())
		data = [tensor_data[i][j][k] for i in xrange(len(tensor_data)) for j in xrange(len(tensor_data[0])) for k in xrange(len(tensor_data[0][0]))]
		values, base = np.histogram(data, bins=1000)
		plt.loglog(base[:-1],values,_linetype,label="{0} {1}".format(_src.capitalize(),_daytype),linewidth=2)
		handles, labels = ax1.get_legend_handles_labels()
		ax1.legend(handles,labels)
	plt.xlim(2,8*10**2)
	plt.ylim(0,10**6)
	plt.xlabel('Value in tensor')
	plt.ylabel('Count')
	subplots_adjust(bottom=0.15, left=0.15, right=0.9, top=0.9)
	# plt.show()
	for postfix in ('eps','png'):
		plt.savefig('../../figure/{0}/04.{0}'.format(postfix))

def tensor_decomposition(src, daytype, method="", component_num=0, sample_rate=1.0):
	import numpy as np
	import sys
	sys.path.append('./scikit-tensor')
	from sktensor import dtensor, tucker_hooi
	from tensor_decomposition_gradient import Gradient_Tensor_Decomposition
	# https://gist.github.com/panisson/7719245
	from _ncp import *
	from _beta_ntf import *

	_component_num = component_num
	tensor_data = json.loads(open("../../data/tensor_data1/tensor_{0}_{1}.txt".format(src,daytype)).read())

	if src == "metro":
		tensor_data = [[[100.0*tensor_data[x][y][h]/sum(tensor_data[x][y]) if sum(tensor_data[x][y])!=0 else 0 \
						for h in range(24)] for y in range(len(tensor_data[0]))] for x in range(len(tensor_data))]
	if src == "taxi":
		if daytype == "workdays":
			tensor_data = [[[tensor_data[x][y][h] if sum(tensor_data[x][y])>=90 else 0 \
							for h in range(24)] for y in range(len(tensor_data[0]))] for x in range(len(tensor_data))]
		if daytype == "holidays":
			tensor_data = [[[tensor_data[x][y][h] if sum(tensor_data[x][y])>=60 else 0 \
							for h in range(24)] for y in range(len(tensor_data[0]))] for x in range(len(tensor_data))]

	# --- sample ---
	A = np.array(tensor_data)
	l1, l2, l3 = A.shape
	A = A.reshape((l1*l2*l3,))
	m = [i for i, a in enumerate(A) if a != 0]
	C = np.random.choice(len(m), len(m)*(1.-sample_rate))
	for i in C:
		A[m[i]] = 0
	A = A.reshape((l1,l2,l3))

	# --- decomposition ---
	T = dtensor(tensor_data)
	# core, (matrix_time, matrix_location_start, matrix_location_finish) = tucker_hooi(T, _component_num, init='random')
	# core, (matrix_time, matrix_location_start, matrix_location_finish) = Gradient_Tensor_Decomposition(T, _component_num, 30, 0.0001, 0.1)
	if method == "ANLS_BPP":
		X_approx = nonnegative_tensor_factorization(dtensor(A), _component_num, method='anls_bpp', min_iter=50, max_iter=500)
		X_approx = X_approx.totensor()
	elif method == "ANLS_AS":
		X_approx = nonnegative_tensor_factorization(dtensor(A), _component_num, method='anls_asgroup', min_iter=50, max_iter=500)
		X_approx = X_approx.totensor()
	elif method == "Beta_NTF":
		beta_ntf = BetaNTF(A.shape, n_components=_component_num, beta=2, n_iter=100, verbose=True)
		beta_ntf.fit(A)
		matrix_location_start, matrix_location_finish, matrix_time = beta_ntf.factors_
		E = np.zeros((_component_num, _component_num, _component_num))
		for k in xrange(_component_num):
			E[k][k][k] = 1
		C = dtensor(E)
		X_approx = C.ttm(matrix_location_start, 0).ttm(matrix_location_finish, 1).ttm(matrix_time, 2)

	# --- compute error ---
	# X_err = abs(T - X_approx).sum()/T.sum()
	X_err = (((T - X_approx)**2).sum()/(T**2).sum())**0.5
	print "Error:", X_err

def plot_error_component_num():
	from pylab import *
	from scipy import interpolate
	from matplotlib.ticker import MultipleLocator, FormatStrFormatter

	ANLS_BPP_metro_workdays = [0.5596,0.4420,0.4180,0.4068,0.4004,0.3950]
	ANLS_AS_metro_workdays  = [0.5596,0.4420,0.4180,0.4068,0.4004,0.3950]
	Beta_NTF_metro_workdays = [0.5596,0.4421,0.4184,0.4089,0.4021,0.3995]

	ANLS_BPP_metro_holidays = [0.5751,0.5471,0.5402,0.5364,0.5348,0.5326]
	ANLS_AS_metro_holidays  = [0.5751,0.5471,0.5402,0.5364,0.5348,0.5323]
	Beta_NTF_metro_holidays = [0.5751,0.5472,0.5409,0.5377,0.5362,0.5351]

	ANLS_BPP_taxi_workdays = [0.7972,0.6136,0.5572,0.5225,0.4960,0.4774]
	ANLS_AS_taxi_workdays  = [0.7972,0.6136,0.5481,0.5105,0.4822,0.4609]
	Beta_NTF_taxi_workdays = [0.7972,0.6251,0.5681,0.5296,0.5109,0.4909]

	ANLS_BPP_taxi_holidays = [0.9515,0.7553,0.7019,0.6844,0.6705,0.6164]
	ANLS_AS_taxi_holidays  = [0.9515,0.7553,0.6943,0.6786,0.6645,0.6079]
	Beta_NTF_taxi_holidays = [0.9515,0.7553,0.7073,0.6997,0.6800,0.6213]

	fig = plt.figure(figsize=(12,5))
	fig.subplots_adjust(left=0.08,right=0.96)
	for subplot, prefix, ymin, ymax, ANLS_BPP_workdays, ANLS_AS_workdays, Beta_NTF_workdays, ANLS_BPP_holidays, ANLS_AS_holidays, Beta_NTF_holidays in [\
		(1, "Metro", 0.3, 0.6, ANLS_BPP_metro_workdays, ANLS_AS_metro_workdays, Beta_NTF_metro_workdays, ANLS_BPP_metro_holidays, ANLS_AS_metro_holidays, Beta_NTF_metro_holidays),\
		(2, "Taxi", 0.3, 1.0, ANLS_BPP_taxi_workdays, ANLS_AS_taxi_workdays, Beta_NTF_taxi_workdays, ANLS_BPP_taxi_holidays, ANLS_AS_taxi_holidays, Beta_NTF_taxi_holidays)]:
		ax1 = fig.add_subplot("12{0}".format(subplot))
		for l, (error, linestyle, label) in enumerate([(ANLS_BPP_workdays,'rx-',"ANLS BPP"),\
													   (ANLS_AS_workdays,'bx-',"ANLS AS")]):
			plt.plot([i+0.05*l for i in range(1,7)],error,linestyle,label="Weekday {0}".format(label),linewidth=1)
		for l, (error, linestyle, label) in enumerate([(ANLS_BPP_holidays,'rx--',"ANLS BPP"),\
													   (ANLS_AS_holidays,'bx--',"ANLS AS")]):
			plt.plot([i+0.05*l for i in range(1,7)],error,linestyle,label="Weekend {0}".format(label),linewidth=1)
		plt.xlim(0.5,6.5)
		plt.ylim(ymin,ymax)
		plt.xlabel('Number of component')
		plt.ylabel('Relative Error')
		plt.title(prefix)
		handles, labels = ax1.get_legend_handles_labels()
		ax1.legend(handles, labels, 3)
	# show()
	for postfix in ('eps','png'):
		plt.savefig('../../figure/{0}/05.{0}'.format(postfix))

def plot_error_sample_rate():
	from pylab import *
	from scipy import interpolate
	from matplotlib.ticker import MultipleLocator, FormatStrFormatter

	ANLS_BPP_metro_workdays = [0.6535,0.6206,0.5871,0.5509,0.5164,0.4803,0.4505,0.4276]
	ANLS_AS_metro_workdays  = [0.6534,0.6209,0.5875,0.5513,0.5153,0.4821,0.4503,0.4273]
	Beta_NTF_metro_workdays = [0.6521,0.6222,0.5856,0.5533,0.5171,0.4820,0.4503,0.4280]

	ANLS_BPP_metro_holidays = [0.7123,0.6872,0.6608,0.6338,0.6075,0.5831,0.5618,0.5464]
	ANLS_AS_metro_holidays  = [0.7124,0.6870,0.6610,0.6335,0.6079,0.5833,0.5615,0.5463]
	Beta_NTF_metro_holidays = [0.7135,0.6887,0.6634,0.6344,0.6086,0.5846,0.5631,0.5489]

	ANLS_BPP_taxi_workdays = [0.7901,0.7520,0.7120,0.6667,0.6303,0.6072,0.5864,0.5560]
	ANLS_AS_taxi_workdays  = [0.7831,0.7436,0.7090,0.6646,0.6217,0.5977,0.5742,0.5481]
	Beta_NTF_taxi_workdays = [0.7996,0.7688,0.7279,0.6806,0.6366,0.6155,0.5903,0.5679]

	ANLS_BPP_taxi_holidays = [0.8927,0.8406,0.8136,0.7920,0.7721,0.7496,0.7258,0.7019]
	ANLS_AS_taxi_holidays  = [0.8961,0.8288,0.8099,0.7788,0.7690,0.7357,0.7138,0.6963]
	Beta_NTF_taxi_holidays = [0.9014,0.8434,0.8188,0.8015,0.7756,0.7516,0.7341,0.7143]

	fig = plt.figure(figsize=(12,5))
	fig.subplots_adjust(left=0.08,right=0.96)
	for subplot, prefix, ymin, ymax, ANLS_BPP_workdays, ANLS_AS_workdays, Beta_NTF_workdays, ANLS_BPP_holidays, ANLS_AS_holidays, Beta_NTF_holidays in [\
		(1, "Metro", 0.3, 0.8, ANLS_BPP_metro_workdays, ANLS_AS_metro_workdays, Beta_NTF_metro_workdays, ANLS_BPP_metro_holidays, ANLS_AS_metro_holidays, Beta_NTF_metro_holidays),\
		(2, "Taxi", 0.3, 1.0, ANLS_BPP_taxi_workdays, ANLS_AS_taxi_workdays, Beta_NTF_taxi_workdays, ANLS_BPP_taxi_holidays, ANLS_AS_taxi_holidays, Beta_NTF_taxi_holidays)]:
		ax1 = fig.add_subplot("12{0}".format(subplot))
		for l, (error, linestyle, label) in enumerate([(ANLS_BPP_workdays,'rx-',"ANLS BPP"),\
													   (ANLS_AS_workdays,'bx-',"ANLS AS")]):
			plt.plot([0.1*i+0.005*l for i in range(2,10)],error,linestyle,label="Weekday {0}".format(label),linewidth=1)
		for l, (error, linestyle, label) in enumerate([(ANLS_BPP_holidays,'rx--',"ANLS BPP"),\
													   (ANLS_AS_holidays,'bx--',"ANLS AS")]):
			plt.plot([0.1*i+0.005*l for i in range(2,10)],error,linestyle,label="Weekend {0}".format(label),linewidth=1)
		plt.xlim(0.15,0.95)
		plt.ylim(ymin,ymax)
		plt.xlabel('Sample rate')
		plt.ylabel('Relative Error')
		plt.title(prefix)
		handles, labels = ax1.get_legend_handles_labels()
		ax1.legend(handles, labels, 3)
	# show()
	for postfix in ('eps','png'):
		plt.savefig('../../figure/{0}/06.{0}'.format(postfix))

def discover_pattern(src, daytype):
	import sys
	sys.path.append('./scikit-tensor')
	from sktensor import dtensor
	import numpy as np
	from _ncp import *
	from pylab import *
	from scipy import interpolate
	from matplotlib.ticker import MultipleLocator, FormatStrFormatter

	_component_num = 3
	tensor_data = json.loads(open("../../data/tensor_data2/tensor_{0}_{1}.txt".format(src,daytype)).read())

	T = dtensor(tensor_data)
	matrix_location_start, matrix_location_finish, matrix_time = nonnegative_tensor_factorization(T, _component_num, method='anls_asgroup', min_iter=20, max_iter=40).U
	matrix_time = np.transpose(matrix_time)

	strength_time = sum(matrix_time,axis=1)
	strength_location_start = sum(matrix_location_start,axis=0)
	strength_location_finish = sum(matrix_location_finish,axis=0)
	for i in range(3):
		print i, strength_time[i]*strength_location_start[i]*strength_location_finish[i]

	# 时间模式
	fig = plt.figure(figsize=(6,5))
	fig.subplots_adjust(left=0.10,right=0.98,top=0.95,bottom=0.10)
	ax = fig.add_subplot(111)
	for index, linestyle, label in [(0,'k-','Pattern 1'),(1,'k--','Pattern 2'),(2,'k:','Pattern 3')]:
		tck = interpolate.splrep(range(24),[100.0*y/max(matrix_time[index]) for y in matrix_time[index]],s=0)
		x = np.arange(0,23,0.1)
		y = interpolate.splev(x,tck,der=0)
		plt.plot(x,y,linestyle,label=label,linewidth=2)
	plt.xlim(0,23)
	plt.ylim(0,125)
	plt.title('{0} ({1})'.format(src.capitalize(),daytype.capitalize()))
	plt.xlabel('Time /hour')
	plt.ylabel('Value')
	handles, labels = ax.get_legend_handles_labels()
	ax.legend(handles,labels,loc=2)
	xmajorLocator = MultipleLocator(1)
	xmajorFormatter = FormatStrFormatter('%d')
	ax.xaxis.set_major_locator(xmajorLocator)
	ax.xaxis.set_major_formatter(xmajorFormatter)
	show()
	# for postfix in ('eps','png'):
	# 	plt.savefig('../../figure/{0}/07_{1}_{2}.{0}'.format(postfix,src,daytype))

	# 空间模式
	lnglat_dict = {}
	for line in fileinput.input("../../data/data_exp/{0}Spatial.csv".format(src)):
		station, lng, lat = line.strip().split(',')
		lnglat_dict[station] = (float(lng), float(lat))
	fileinput.close()

	for line in fileinput.input("../../data/tensor_data2/tensor_index_{0}.txt".format(src)):
		index, station = line.strip().split('\t')
		lnglat_dict[int(index)] = [station, lnglat_dict[station]]
		del lnglat_dict[station]
	fileinput.close()

	fig = plt.figure(figsize=(12,4))
	fig.subplots_adjust(left=0.02,right=0.98,top=0.98,bottom=0.02,wspace=0.08)
	color_dict = {0:_color_red_list,1:_color_yellow_list,2:_color_green_list}
	tensor_abs = np.array(json.loads(open("../../data/tensor_data1/tensor_{0}_{1}.txt".format(src,daytype)).read()))
	tensor_abs_src = [tensor_abs[i,:,:].sum() for i in xrange(len(matrix_location_start))]
	tensor_abs_dst = [tensor_abs[:,i,:].sum() for i in xrange(len(matrix_location_finish))]
	for subplot, (matrix_location, title) in enumerate([(matrix_location_start, "{0} ({1}) Origin".format(src.capitalize(),daytype.capitalize())), (matrix_location_finish,"{0} ({1}) Destination".format(src.capitalize(),daytype.capitalize()))]):
		_sum = matrix_location.sum(axis=0)
		_avg = [1.*s/_sum.sum() for s in _sum]
		matrix_location = [[1.*v/matrix_location[p].sum()-_avg[i] \
							for i,v in enumerate(matrix_location[p])] \
								for p in xrange(len(matrix_location))]
		matrix_location = [matrix_location[p].index(max(matrix_location[p])) \
								for p in xrange(len(matrix_location))]

		ax = fig.add_subplot("12{0}".format(subplot+1))
		image = plt.imread('../../result/_map_bounds/shanghai_nokia.png')
		ax.imshow(image)
		for _index, _class in enumerate(matrix_location):
			(_station), (_lng, _lat) = lnglat_dict[_index]
			if src == "metro":
				cube_color = color_dict[_class][10]['color']
				circle = plt.Circle(cord2XY(_lng,_lat), 10, fc=cube_color, alpha=0.5, linewidth=0)
				ax.add_patch(circle)
			if src == "taxi":
				_x, _y = [int(c) for c in _station.split('_')]
				if subplot == 0:
					cube_color = get_color(color_dict[_class], min(tensor_abs_src), max(tensor_abs_src), tensor_abs_src[_index])
				if subplot == 1:
					cube_color = get_color(color_dict[_class], min(tensor_abs_dst), max(tensor_abs_dst), tensor_abs_dst[_index])
				if cube_color != '#ffffff':
					cube = plt.Rectangle(gridXY2mapXY(_x,_y), _cube_width, _cube_height, fc=cube_color, alpha=0.6, linewidth=0)
				ax.add_patch(cube)
		ax.set_xticks([])
		ax.set_yticks([])
		plt.title(title)
	show()
	# for postfix in ('eps','png'):
	# 	plt.savefig('../../figure/{0}/08_{1}_{2}.{0}'.format(postfix,src,daytype))


if __name__ == "__main__":
	# generate_tensor('metro','workdays')
	# generate_tensor('metro','holidays')
	# generate_tensor('taxi','workdays')
	# generate_tensor('taxi','holidays')
	# plot_volumn_statistics()
	# tensor_statistics()
	# method = "ANLS_BPP"
	# method = "ANLS_AS"
	# method = "Beta_NTF"
	# metro, taxi, workdays, holidays
	# tensor_decomposition('taxi','holidays',method=method,component_num=3,sample_rate=0.9)
	# plot_error_component_num()
	# plot_error_sample_rate()
	# metro, taxi, workdays, holidays
	# discover_pattern('metro','workdays')
	# discover_pattern('metro','holidays')
	# discover_pattern('taxi','workdays')
	# discover_pattern('taxi','holidays')
	pass

