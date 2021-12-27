from typing import Counter
import cv2
import numpy as np
import math
import matplotlib
from PIL import Image
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt
from numpy.lib.type_check import imag
from sympy.utilities.iterables import multiset_permutations, subsets, variations

class UnionFind(object):
	def __init__(self, n):
		self.uf = [i for i in range(n + 1)]
		self.sets_count = n

	def find(self, p):
		r = self.uf[p]
		if p != r:
			r = self.find(r)
		self.uf[p] = r
		return r

	def union(self, p, q):
		"""连通p,q 让q指向p"""
		proot = self.find(p)
		qroot = self.find(q)
		if proot == qroot:
			return
		elif self.uf[proot] > self.uf[qroot]:  # 负数比较, 左边规模更小
			# self.uf[qroot] += self.uf[proot]
			self.uf[proot] = qroot
		else:
			# self.uf[proot] += self.uf[qroot]    # 规模相加
			self.uf[qroot] = proot
		self.sets_count -= 1  # 连通后集合总数减一

	def is_connected(self, p, q):
		"""判断pq是否已经连通"""
		return self.find(p) == self.find(q)  # 即判断两个结点是否是属于同一个祖先


class court_model(object):
	def __init__(self, img):
		super().__init__()
		self.standard = dict()
		# 垂直线
		self.standard['vlines'] = np.array([[14, 29, 14, 450],
						   [29, 29, 29, 450],
						   [110, 29, 110, 177],
						   [110, 303, 110, 450],
						   [191, 29, 191, 450],
						   [205, 29, 205, 450]])
		# 水平线
		self.standard['hlines'] = np.array([[14, 29, 205, 29],
						   [14, 53, 205, 53],
						   [14, 177, 205, 177],
						   [14, 303, 205, 303],
						   [14, 426, 205, 426],
						   [14, 450, 205, 450]])
		self.standard['indexes'] = np.arange(12)
		self.calculate_intersections_as_matrix()

		self.image = dict()
		i = 0
		while i < 4 and (self.image.get('lines') is None or self.image.get('lines').shape[0] < 5):
			candidate, contours = self.white_pixel_extract(img, threshold=5+5*i)
			self.line_detection(candidate, contours, img)
			i += 1
		if self.image['lines'].shape[0] < 5:
			print('Court Model Init Failed! Court lines not enough!')
		else:
			self.M, self.score = self.model_fitting()
			pers = cv2.warpPerspective(img, self.M, (img.shape[1], img.shape[0]))
			cv2.imwrite("mediating/perspective.png", pers, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
	
	def calculate_intersections_as_matrix(self, lines=None):
		'''
		calculate intersection of lines in the standard model. If img_shape is not None, we remove intersections those are out of image box.
		:param: slines: n * 4 matrix, a list of line (x1, y1, x2, y2)
		:param: img_shape: such as (1080, 1980, 3).
		:return: points: n * n * 2, points[i, j] = the intersection of line i and line j, and it's location is (x, y)
		'''
		# calculate standard intersections matrix
		if lines is None:
			hlines = self.standard['hlines']
			vlines = self.standard['vlines']
		
			points = np.ones(hlines.shape[0] * vlines.shape[0] * 2, dtype='int').reshape((hlines.shape[0], vlines.shape[0], 2))
			points = -1 * points
			for i, l1 in enumerate(hlines):
				for j, l2 in enumerate(vlines):
					points[i, j, 0], points[i, j, 1] = self.calculate_intersection(l1, l2)
			self.standard['intersections'] = points
			return
		
		# calculate image intersections matrix
		n = lines.shape[0]
		points = np.ones(n * n * 2, dtype='int').reshape((n, n, 2)) * (-1)
		for i, l1 in enumerate(lines):
			for j, l2 in enumerate(lines):
				if i >= j:
					continue
				points[i, j, 0], points[i, j, 1] = self.calculate_intersection(l1, l2)
		
		self.image['lines'] = lines
		self.image['intersections'] = points

		return


	def line_constraint(self, g, tao, sigmal, sigmad):
		"""
		remove region that is not a line, such as people's white t-shirt.
		we can find the formula from paper "Robust Camera Calibration for Sport Videos using Court Models".
		:params: g: the image. It is a (m * n) matrix, every pixel has one value, 255 or 0. 255 means this pixel is on candidate line.
		:params: tao: a line's width threshold
		:params: sigmal: the threshold
		:params: sigmad: the threshold
		:return: l: image after removing. It has two values. If a pixel in a line, it is "255", else, it is "0".
		"""
		height, width = g.shape
		t1 = g - np.concatenate((np.zeros((tao, width)), g[:height - tao, :]), axis=0)
		t2 = g - np.concatenate((g[tao:, :], np.zeros((tao, width))), axis=0)
		# l = np.zeros(g.shape)
		l = np.where((g >= sigmal) & (t1 > sigmad) & (t2 > sigmad), 255, 0)

		t1 = g - np.concatenate((np.zeros((height, tao)), g[:, :width - tao ]), axis=1)
		t2 = g - np.concatenate((g[:, tao:], np.zeros((height, tao))), axis=1)
		l = np.where((l == 255) | ((g >= sigmal) & (t1 > sigmad) & (t2 > sigmad)), 255, 0)
		l = l.astype(np.uint8)
		return l


	def white_pixel_extract(self, img, threshold=5):
		h, w = img.shape[:2]
		# 1. 找到绿色区域
		# convert to HSV image
		hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

		# HARD CODED COURT COLOR :(
		court_color = np.uint8([[[123, 152, 76]]])

		hsv_court_color = cv2.cvtColor(court_color, cv2.COLOR_BGR2HSV)
		hue = hsv_court_color[0][0][0]

		# define range of green color in HSV - Again HARD CODED! :(
		lower_color = np.array([hue - threshold,40,10])
		upper_color = np.array([hue + threshold,255,255])

		# Threshold the HSV image to get only green colors
		mask = cv2.inRange(hsv_img, lower_color, upper_color)
		cv2.imwrite("mediating/mask.png", mask, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

		# 2. 框出球场的范围（凸包）contours
		blured = cv2.blur(mask, (5, 5))
		mask = np.zeros((h+2, w+2), np.uint8)  #掩码长和宽都比输入图像多两个像素点，满水填充不会超出掩码的非零边缘 
		#进行泛洪填充
		cv2.floodFill(blured, mask, (w-1,h-1), (0, 0, 0), (2,2,2),(3,3,3),8)
		cv2.imwrite("mediating/floodfill.png", blured, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

		# #得到灰度图
		# gray = cv2.cvtColor(blured,cv2.COLOR_BGR2GRAY) 
		# cv2.imwrite("mediating/gray.png", gray) 
		
		#定义结构元素 
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
		#开闭运算，先开运算去除背景噪声，再继续闭运算填充目标内的孔洞
		opened = cv2.morphologyEx(blured, cv2.MORPH_OPEN, kernel)
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(20, 20))
		closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel) 
		cv2.imwrite("mediating/closed.png", closed)
		
		#求二值图
		ret, binary = cv2.threshold(closed,50,255,cv2.THRESH_BINARY) 
		cv2.imwrite("mediating/binary.png", binary) 
		
		# 找到轮廓
		contours, hierarchy = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		valid_contours = np.array([cv2.contourArea(cnt) for cnt in contours])
		valid_contours = np.argwhere(valid_contours > 300)[:, 0]
		contours = np.array(contours)[valid_contours].tolist()
		
		hulls = [cv2.convexHull(cnt) for cnt in contours]
		poly = img.copy()
		cv2.polylines(poly, hulls, True, (0, 0, 255), 2)  # red
		cv2.imwrite("mediating/poly.png", poly)
		# # 绘制轮廓
		cnt_fill = np.zeros((h, w),dtype=np.uint8)
		cnt_fill = cv2.drawContours(cnt_fill,contours,-1,255,cv2.FILLED)
		# cnt_fill = cv2.drawContours(cnt_fill, contours, -1, 255, 3)
		cv2.imwrite("mediating/cnt_fill.png", cnt_fill)
		result = cv2.bitwise_and(blured, cnt_fill)
		result = cv2.bitwise_not(result)
		# 绘制结果
		cv2.imwrite("mediating/result.png", result)

		# 3. 求灰度图，并二值化，找出白色像素的区域，并去掉大块的非线性的白色像素
		# # Bitwise-AND mask and original image
		# res = cv2.bitwise_and(img,img, mask=binary)
		# cv2.imwrite("mediating/basketball_res.png", res, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
		# cv2.imwrite("mediating/origin.png", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

		# im_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
		retval, im_at_fixed = cv2.threshold(result, 50, 255, cv2.THRESH_BINARY) # 二值化
		cv2.imwrite("mediating/im_at_fixed.png", im_at_fixed, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
		# remove dressed region
		c1 = self.line_constraint(im_at_fixed, 10, 128, 20)
		cv2.imwrite("mediating/candidate.png", c1, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

		# todo: remove textured region
		# c2 = cv2.cornerEigenValsAndVecs(c1, 5, 3)
		return c1, hulls


	def draw_line(self, imgName, img, lines, coloridx=None):
		colors = np.array([(25, 25, 112), (123, 104, 238), (0, 191, 255), (255, 218, 185), (47, 79, 79),
					(255, 127, 0), (0, 0, 0), (139, 0, 0), (0, 205, 0), (67, 205, 128),
					(0, 197, 205), (105, 139, 34), (139, 134, 78), (255, 193, 37), (205, 92, 92),
					(178, 34, 34), (255, 20, 147), (139, 69, 19), (148, 0, 211), (139, 137, 137)])
		n = lines.shape[0]
		if n == 0:
			return
		if coloridx is None:
			coloridx = np.arange(n)
		for i, line in enumerate(lines):
			cidx = coloridx[i]
			x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
			r, b, g = int(colors[cidx % 20, 0]), int(colors[cidx % 20, 1]), int(colors[cidx % 20, 2])
			cv2.line(img, (x1, y1), (x2, y2), (r, b, g), 1)
		cv2.imwrite(imgName, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


	def is_same_line(self, line1, line2, normal1, normal2, athres, dthres):
		'''
		given two lines "normal1" and "normal2" , if they are the same line, return true; else, return false.
		base: ||normal1|| = ||normal2|| = 1
		:params: normal1: the normal of line1, (n1, n2, d)
		:params: normal2: the normal of line2, (n1, n2, d)
		:params: athres: the angle threshold, if two lines' angle less than athres, we think they may be the same line.
		:params: dthres: the distance threshold, if two lines' distance shorter than dthres, we think they may be the same line.
		
		:return: true if n1T * n2 < cos(0.75C) && |d1 - d2| < 8
				false else
		'''
		d11 = self.distance_point_line(line1[:2], line2)
		d12 = self.distance_point_line(line1[2:], line2)
		d21 = self.distance_point_line(line2[:2], line1)
		d22 = self.distance_point_line(line2[2:], line1)
		cosAngle = normal1[0] * normal2[0] + normal1[1] * normal2[1]
		return cosAngle > athres and ((d11 < dthres and d12 < dthres) or (d21 < dthres and d22 < dthres))

	def calculate_normal(self, lines):
		"""
		normal: n * 5, a normal = (n1, n2, d, k, b)
		"""
		# get normal: n1*x + n2 * y - d = 0
		line0, line1, line2, line3 = lines[:, 0:1], lines[:, 1:2], lines[:, 2:3], lines[:, 3:]
		normal = np.concatenate([
			line2 - line0, line3 - line1, np.abs(line1 * line2 - line0 * line3)
		], axis=1, dtype='float')

		tmp = np.sum(normal[:, :2] ** 2, axis=1, keepdims=True) ** .5
		normal /= tmp

		# get k and b. line: y = k * x + b
		kb = np.concatenate([
			line3 - line1, line2 * line1 - line0 * line3
		], axis=1, dtype='float')
		tmp = line2 - line0
		kb = kb / tmp
		normal = np.concatenate((normal, kb), axis=1, dtype='float')

		return normal

	def calculate_y_line(self, line, x):
		x1, y1, x2, y2 = line[:4]
		A = y1 - y2
		B = x2 - x1
		C = x1 * y2 - x2 * y1
		if B == 0:
			return (y1 + y2) / 2
		return -(A * x + C) / B

	def merge_same_lines(self, lines, normals, ufsidx):
		pureidx = np.unique(ufsidx)
		purelinesarray = []
		# 修改多条线逆合成一条线的算法，斜率取均值，然后找多条线段的中心点，用中心点拟合
		for p in pureidx:
			idx = np.argwhere(ufsidx == p).sum(axis=1)
			tmpn = normals[idx]
			tmpl = lines[idx]

			# new line: y = avgk * x + avgb
			avgk = tmpn[:, 3].mean()
			avgx = tmpl[:, (0, 2)].mean()
			y = np.array([self.calculate_y_line(ll, avgx) for ll in tmpl])
			avgy = y.mean()
			avgb = avgy - avgk * avgx

			# get a new line and add to return vecs
			x2 = tmpl[:, (0, 2)].max()
			x1 = tmpl[:, (0, 2)].min()
			pline = np.array([x1, x1 * avgk + avgb, x2, x2 * avgk + avgb], dtype='int')

			purelinesarray.append(pline)
		
		purelines = np.array(purelinesarray)
		purenormals = self.calculate_normal(purelines)
		return purelines, purenormals


	def remove_invalid_line(self, lines, contours):
		'''
		remove invalid and duplicated lines.
		:param: lines: n * 4, a line = (x1, y1, x2, y2)
		normal: n * 5, a normal = (n1, n2, d, k, b)
		:return: purelines: a (m * 4) matrix, m <= n.
			purenormals: a (m * 5) matrix
		'''
		# remove lines those are not in the court contours.
		valid = np.zeros((lines.shape[0]), dtype=int)
		for cnt in contours:
			p1 = np.array([cv2.pointPolygonTest(cnt, (int(l[0]), int(l[1])), True) for l in lines])
			p2 = np.array([cv2.pointPolygonTest(cnt, (int(l[2]), int(l[3])), True) for l in lines])
			v = np.where((p1 > 0) & (p2 > 0), 1, 0)
			valid = cv2.bitwise_or(valid, v).reshape(-1)
		lines = lines[np.argwhere(valid == 1)[:, 0]]

		# 计算法线
		normal = self.calculate_normal(lines)

		# get same-line set, if line a and line b are the same line, ufs.uf[a] = ufs.uf[b]
		ufs = UnionFind(lines.shape[0])
		angleThres = np.cos(np.pi / 240)
		dThres = 8
		for i in range(lines.shape[0]):
			for j in range(lines.shape[0]):
				if i < j and self.is_same_line(lines[i], lines[j], normal[i], normal[j], angleThres, dThres):
					ufs.union(i + 1, j + 1)

		ufsidx = np.array(ufs.uf)[1:] - 1
		
		# remove duplicate lines, every color only need 1 line.
		purelines, purenormals = self.merge_same_lines(lines, normal, ufsidx)
		idx = np.lexsort(purenormals.T[:4, :])
		return purelines[idx], purenormals[idx]

	
	def line_detection(self, img, contours, source_image):
		"""
		:params: img: input image. we detecte lines from this image.
		:return: purelines: detected lines.
		"""
		# preprocess image
		mask = img
		height, width = mask.shape
		skel = np.zeros([height, width], dtype=np.uint8)  # [height,width,3]
		kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
		while (np.count_nonzero(mask) != 0):
			eroded = cv2.erode(mask, kernel)
			temp = cv2.dilate(eroded, kernel)
			temp = cv2.subtract(mask, temp)
			skel = cv2.bitwise_or(skel, temp)
			mask = eroded.copy()
		cv2.imwrite("mediating/skel.png", skel, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

		# # option: dilate + erode
		# closed = cv2.morphologyEx(skel, cv2.MORPH_CLOSE, kernel)
		# cv2.imshow("closed", closed)

		# option: dilate
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
		dilt = cv2.dilate(skel, kernel, iterations=1)
		cv2.imwrite("mediating/dilate.png", dilt, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

		# hough line detection
		edges = cv2.Canny(dilt, 32, 200, apertureSize=3)
		cv2.imwrite("mediating/edges.png", edges, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
		lines = cv2.HoughLinesP(edges, 1, np.pi / 360, 15, minLineLength=200, maxLineGap=30)

		lines = lines.reshape((lines.shape[0], -1))
		result2 = source_image.copy()
		self.draw_line("mediating/multiline.png", result2, lines)

		purelines, purenormals = self.remove_invalid_line(lines, contours)
		self.image['normal'] = purenormals
		result3 = source_image.copy()
		self.draw_line("mediating/pureline.png", result3, purelines)
		self.calculate_intersections_as_matrix(purelines)

		return


	def calculate_intersection(self, line1, line2):
		"""
		compute intersection of line1 and line2. If line1 and line2 don't have intersection, we return (-1, -1).
		:params: line1: (x1, y1, x2, y2)
		:params: line2: (x1, y1, x2, y2)
		:return: ptx, pty: the intersection (ptx, pty)
		"""
		x1, y1, x2, y2 = line1[0], line1[1], line1[2], line1[3]
		x3, y3, x4, y4 = line2[0], line2[1], line2[2], line2[3]
		d = ((x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4))
		if d == 0:
			return -1, -1
		tmp1 = x1 * y2 - y1 * x2
		tmp2 = x3 * y4 - y3 * x4
		ptx = (tmp1 * (x3 - x4) - (x1 - x2) * tmp2) / d;
		pty = (tmp1 * (y3 - y4) - (y1 - y2) * tmp2) / d;
		return ptx, pty


	def compute_intersections_as_list(self, lines):
		'''
		calculate intersection of lines in the image.
		:param: slines: n * 4 matrix, a list of line[x1, y1, x2, y2]
		:return: points: m * 4, m point, a point(i, j, x, y) belongs to both line i and line j, and it's location is (x, y)
		'''
		tmp = []
		for i, l1 in enumerate(lines):
			for j, l2 in enumerate(lines):
				if i >= j:
					continue
				ptx, pty = self.calculate_intersection(l1, l2)
				if ptx != -1 and pty != -1:
					tmp.append([i, j, ptx, pty])
		points = np.array(tmp, dtype=np.int32)
		# if points.shape[0] > 2:
		#     points = points[np.lexsort([points[:, 1], points[:, 0]]), :]
		return points


	def get_standard_intersection(self, idxes):
		'''
		get points[idxes] from standard points.
		:params: standard_points: n * n * (x, y). We only use upper triangle of this matrix, which means we only get standard_points[i, j] when i < j.
		:params: idxes: list of standard lines. For example, idxes=[0,2] means line[14, 29, 205, 29] and line[14, 177. 205, 177]
		:returns: points: lines' intersetion. a point is [i, j, x, y], which is the intersection of line i and line j, and the point is (x, y)
		'''
		tmp = []
		for i in range(idxes.shape[0]):
			for j in range(idxes.shape[0]):
				if i >= j:
					continue
				l, r = min(idxes[i], idxes[j]), max(idxes[i], idxes[j])
				ptx, pty = self.standard['intersections'][l, r, 0], self.standard['intersections'][l, r, 1]
				if ptx != -1 and pty != -1:
					tmp.append([i, j, ptx, pty])
		points = np.array(tmp, dtype='int')
		if points.shape[0] > 1:
			points = points[np.lexsort([points[:, 1], points[:, 0]]), :]
		return points


	def calculate_mapping_lines(self, M):
		'''
		image(u, v, 1) = M * standard(x, y, 1)^T
		When we get a homography M, we need calculate the lines mapped from the standard lines[idxes] to the lines on the image.
		:params: M: homography
		:params: idxes: standard line indexes we need to map.
		:return: lines: n * (x1, y1, x2, y2), mapping lines to the image.
		'''
		sl = np.concatenate((self.standard['hlines'], self.standard['vlines']), axis=0)
		n = sl.shape[0]
		point1 = np.hstack((sl[:, :2], np.ones((n, 1), dtype='int')))
		mapping1 = np.dot(M, point1.T)
		mapping1 = mapping1.T
		point2 = np.hstack((sl[:, 2:], np.ones((n, 1), dtype='int')))
		mapping2 = np.dot(M, point2.T)
		mapping2 = mapping2.T
		return np.concatenate((mapping1[:, :2], mapping2[:, :2]), axis=1)


	def distance_point_line(self, point, line):
		'''
		calculate the distance from a point 'point' to a line 'line'.
		:params: point: (x0, y0)
		:params: line: (x1, y1, x2, y2)
		:return: distance
		'''
		x0, y0 = point[0], point[1]
		x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
		A = y1 - y2
		B = x2 - x1
		C = x1 * y2 - y1 * x2
		mole = np.abs(A * x0 + B * y0 + C)
		deno = (A ** 2 + B ** 2) ** .5
		return mole / deno


	def calculate_score(self, maplines):
		'''
		for px in each (imgline, mapline), we calculate distance(px, mapline) and add all distance to get score.
		:params: imglines: we recognize lines from the image, n * (x1, y1, x2, y2)
		:params: maplines: using homography M, we map the standard_line to image, n * (x1, y1, x2, y2)
		:return: score: this mapping's score. We like lower score.
		'''
		imglines = self.image['lines']
		inormal = self.image['normal']
		mnormal = self.calculate_normal(maplines)
		score = 0
		angleThres = np.cos(np.pi / 240)
		dThres = 8
		for i in range(maplines.shape[0]):
			for j in range(imglines.shape[0]):
				flag = self.is_same_line(maplines[i], imglines[j], mnormal[i], inormal[j], angleThres, dThres)
				if flag is False:
					continue

				il, ml = imglines[j], maplines[i]
				iA = il[1] - il[3]
				iB = il[0] - il[2]
				iC = il[1] * il[2] - il[0] * il[3]
				mA = ml[1] - ml[3]
				mB = ml[0] - ml[2]
				mC = ml[1] * ml[2] - ml[0] * ml[3]
				deno = (mA ** 2 + mB ** 2) ** .5
				for x0 in range(il[0], il[2] + 1):
					y0 = -(iA * x0 + iC) / iB
					mole = np.abs(mA * x0 + mB * y0 + mC)
					score += (mole / deno)
		return score


	def line_grouping(self, intersections):
		s = np.sum(intersections, axis=2)
		d = np.argwhere(s > 0)
		hgroup = np.unique(d[:, 0])
		vgroup = np.unique(d[:, 1])
		return hgroup, vgroup

	def get_points(self, lidx, type='image'):
		h1, h2, v1, v2 = lidx[0], lidx[1], lidx[2], lidx[3]
		if type == 'image':
			points = self.image['intersection'][[h1, h2]][:, [v1, v2]]
		elif type == 'standard':
			points = self.standard['intersection'][[h1, h2]][:, [v1, v2]]
		points = points.reshape((4, 2))
		return points
	
	def model_fitting_once(self, ig_points):
		s = 2147483647
		resM = np.zeros((3, 3), dtype=np.float16)
		for sh in range(3):
			for sv in range(4):
				st_points = self.get_points(np.array(sh, sh+1, sv, sv+1), type='standard')

				# 3. calculate homography matrix H
				M, mask = cv2.findHomography(ig_points, st_points, cv2.RANSAC, 3.0)
				if M is None:
					continue
				# 4. get lines from standard model using M.
				# for each point(x, y) in standard model, we calculate (u, v, 1) = M*(x, y, 1)^T
				mapping_lines = self.calculate_mapping_lines(M)
				# 5. calculate accuracy score of p, and get the most suitable one.
				score = self.calculate_score(mapping_lines)
				if s < score:
					s = score
					resM = M
		return resM, s
	
	def model_fitting(self):
		'''
		fit standard model
		:param: lines: detected lines from image
		:param: points: points[i, j] = intersection point of line i and line j
		:return:
		'''

		s = 2147483647
		resM = np.zeros((3, 3), dtype=np.float16)
		# get all possible line lists
		# for each line list:
		isx = self.image['intersections'][:, :, 0]
		valid_points_idx = np.argwhere(isx > 0) # 找到所有非(-1, -1)的点的坐标
		for pi in subsets(np.arange(valid_points_idx.shape[0]), 4):
			p = np.array(pi)
			xyidx = valid_points_idx[p]

			# 检查四个点是否有三点共线，如果有，则跳过
			a = list(Counter(xyidx.flatten()))
			if a.count(2) != len(a):
				continue

			# 检查四个点是否构成凸包，如果不是，则跳过
			points = self.image['intersections'][xyidx[:, 0], xyidx[:, 1]]
			hull = cv2.convexHull(points, clockwise=False, returnPoints=True)
			if hull.shape[0] < 4:
				continue
			points = hull.reshape((4, -1))
			# 找到四个点，能够组成矩形，匹配每个小框框
			print(xyidx)
			for i in range(4):
				M, score = self.model_fitting_once(points)
				if score < s:
					s = score
					resM = M
					print(p, "score: ", s)
				points = points[[1, 2, 3, 0]]

		return resM, s

	def init_court_model(self, img):
		"""
		init court model from image
		:return: M: homography. image(u, v, w) = M \dot standard(x, y, 1)^T
		"""

		result4 = img.copy()
		M, score = self.model_fitting(self.image['lines'])
		perspective = cv2.warpPerspective(result4, M, (result4.shape[1], result4.shape[0]))
		cv2.imwrite("mediating/suitable.png", perspective, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

		return M

	def is_in_court(self, ankle_points, M):
		"""
		we have a standard model defined by standard_lines and a homography M. 
		We calculate a point whether in this court or not.
		:params: ankle_points: 2 * 2 matrix. [left ankle, right ankle], every ankle = (x, y)
		:params: M: homography
		:return: true if 
				false else
		"""

		point1 = np.hstack((ankle_points, np.ones((2, 1), dtype='int')))
		mapping1 = np.dot(M, point1.T).T
		mapping1 = mapping1[:, :2]
		def in_standard(point):
			x0, y0 = point[0], point[1]
			return x0 <= 205 and x0 >= 14 and y0 <= 450 and y0 >= 14
		
		return in_standard(mapping1[0, :]) and in_standard(mapping1[1, :])
# ----------------- END ----------------------

# cv2.waitKey(500000)
