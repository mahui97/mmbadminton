import cv2
import numpy as np
import math
import matplotlib
from PIL import Image
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt
from sympy.utilities.iterables import multiset_permutations, variations

# converted = convert_hls(img)
# yellow color mask

# lower = np.uint8([10, 0,   100])
# upper = np.uint8([40, 255, 255])
# yellow_mask = cv2.inRange(image, lower, upper)
# combine the mask
# mask = cv2.bitwise_or(white_mask, yellow_mask)

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
		self.standard['lines'] = np.array([[14, 29, 205, 29],
						   [14, 53, 205, 53],
						   [14, 177, 205, 177],
						   [14, 303, 205, 303],
						   [14, 426, 205, 426],
						   [14, 450, 205, 450],
						   [14, 29, 14, 450],
						   [29, 29, 29, 450],
						   [110, 29, 110, 177],
						   [110, 303, 110, 450],
						   [191, 29, 191, 450],
						   [205, 29, 205, 450]])
		self.standard['indexes'] = np.arange(12)
		_, self.standard['intersections'] = self.calculate_intersections_as_matrix(self.standard['lines'])

		self.image = dict()
		self.image['lines'] = np.zeros((1, 4))
		self.image['intersections'] = np.zeros((1,1,2))
		i = 0
		while i < 4 and self.image['lines'].shape[0] < 4:
			candidate = self.white_pixel_extract(img, threshold=190-12*i)
			self.image['lines'], self.image['intersections'] = self.line_detection(candidate, img)
			i += 1
		if self.image['lines'].shape[0] < 4:
			print('Court Model Init Failed! Court lines not enough!')
			return -1
		self.M = self.model_fitting()
	
	def calculate_intersections_as_matrix(self, slines, img_shape=None):
		'''
		calculate intersection of lines in the standard model. If img_shape is not None, we remove intersections those are out of image box.
		:param: slines: n * 4 matrix, a list of line (x1, y1, x2, y2)
		:param: img_shape: such as (1080, 1980, 3).
		:return: points: n * n * 2, points[i, j] = the intersection of line i and line j, and it's location is (x, y)
		'''
		n = slines.shape[0]
		points = np.zeros(n * n * 2, dtype='int').reshape((n, n, 2))
		for i, l1 in enumerate(slines):
			for j, l2 in enumerate(slines):
				if i >= j:
					continue
				points[i, j, 0], points[i, j, 1] = self.calculate_intersection(l1, l2)
		if img_shape == None:
			return points
		pshape = points.shape
		p0 = points[:, :, 0].reshape((pshape[0], pshape[1], 1))
		p0 = np.where((p0 >= 0) & (p0 <= img_shape[0]), p0, -1)
		p1 = points[:, :, 1].reshape((pshape[0], pshape[1], 1))
		p1 = np.where((p0 >= 0) & (p0 <= img_shape[1]), p1, -1)
		points = np.concatenate((p0, p1), axis=2)

		xx = np.sum(points, axis=(1,2))
		invalid_idx = np.argwhere(xx < 0)
		points = np.delete(points, invalid_idx, axis=0)
		points = np.delete(points, invalid_idx, axis=1)

		lines = np.delete(slines, invalid_idx, axis=0)

		return lines, points


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


	def white_pixel_extract(self, img, threshold=185):
		# white pixel
		# image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
		# image = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
		im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   #转换了灰度化
		cv2.imwrite("mediating/gray.png", im_gray, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
		retval, im_at_fixed = cv2.threshold(im_gray, threshold, 255, cv2.THRESH_BINARY) # 二值化
		cv2.imwrite("mediating/im_at_fixed.png", im_at_fixed, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

		# lower = np.uint8([0, 170, 20])
		# upper = np.uint8([255, 255, 255])
		# white_mask = cv2.inRange(image, lower, upper)
		# cv2.imwrite("mediating/mask.png", white_mask, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

		# remove dressed region
		c1 = self.line_constraint(im_at_fixed, 10, 128, 20)
		cv2.imwrite("mediating/candidate.png", c1, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

		# todo: remove textured region
		# c2 = cv2.cornerEigenValsAndVecs(c1, 5, 3)
		return c1


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


	def remove_duplicate_line(self, lines, img):
		'''
		remove duplicate lines.
		:param: lines: n * 4, a line = (x1, y1, x2, y2)
		normal: n * 5, a normal = (n1, n2, d, k, b)
		:return: purelines: a (m * 4) matrix, m <= n.
			purenormals: a (m * 5) matrix
		'''
		# get normal: n1*x + n2 * y - d = 0
		line0, line1, line2, line3 = lines[:, 0:1], lines[:, 1:2], lines[:, 2:3], lines[:, 3:]
		normal = np.concatenate([
			line2 - line0, line3 - line1, np.abs(line1 * line2 - line0 * line3)
		], axis=1, dtype='float')

		tmp = np.sum(normal[:, :2] ** 2, axis=1, keepdims=True) ** .5
		normal /= tmp

		# get same-line set, if line a and line b are the same line, ufs.uf[a] = ufs.uf[b]
		ufs = UnionFind(lines.shape[0])
		def is_same_line(normal1, normal2, athres, dthres):
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
			deltaD = abs(normal1[2] - normal2[2])
			cosAngle = normal1[0] * normal2[0] + normal1[1] * normal2[1]
			return cosAngle > athres and deltaD < dthres
		angleThres = np.cos(np.pi / 240)
		dThres = 8
		for i, n1 in enumerate(normal):
			for j, n2 in enumerate(normal):
				if i < j and is_same_line(n1, n2, angleThres, dThres):
					ufs.union(i + 1, j + 1)

		# # draw lines, the same lines have the same color
		ufsidx = np.array(ufs.uf)[1:] - 1
		# print(coloridx)
		result2 = img.copy()
		self.draw_line("mediating/multiline.png", result2, lines, coloridx=ufsidx)

		# remove duplicate lines, every color only need 1 line.
		kb = np.concatenate([
			line3 - line1, line2 * line1 - line0 * line3
		], axis=1, dtype='float')
		tmp = line2 - line0
		kb = kb / tmp
		normal = np.concatenate((normal, kb), axis=1, dtype='float')
		pureidx = np.unique(ufsidx)
		purelinesarray = []
		purenormalsarray = []
		for p in pureidx:
			tmpn = normal[np.argwhere(ufsidx == p).sum(axis=1)]
			avgk = tmpn[:, 3].mean()
			avgb = tmpn[:, 4].mean()

			# get a new line and add to return vecs
			tmpl = lines[np.argwhere(ufsidx == p).sum(axis=1)]
			x2 = tmpl[:, (0, 2)].max()
			x1 = tmpl[:, (0, 2)].min()
			pline = np.array([x1, x1 * avgk + avgb, x2, x2 * avgk + avgb], dtype='int')

			tmp = (1 + avgk ** 2) ** .5
			pnormal = np.array(
				[pline[2] - pline[0], pline[3] - pline[1], np.abs(pline[1] * pline[2] - pline[0] * pline[3])],
				dtype='float')
			pnormal /= tmp
			pnormal = np.insert(pnormal, 2, [avgk, avgb])
			purelinesarray.append(pline)
			purenormalsarray.append(pnormal)

		# get new lines 'purelines' and new normals 'purenormals'
		purelines = np.array(purelinesarray)
		purenormals = np.array(purenormalsarray)
		result3 = img.copy()
		self.draw_line("mediating/pureline.png", result3, purelines, coloridx=pureidx)
		return purelines, purenormals


	def line_detection(self, img, source_image):
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
		edges = cv2.Canny(dilt, 50, 200, apertureSize=3)
		cv2.imwrite("mediating/edges.png", edges, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
		lines = cv2.HoughLinesP(edges, 1, np.pi / 360, 15, minLineLength=270, maxLineGap=20)
		# print(lines.shape)

		lines = lines.reshape((lines.shape[0], -1))
		purelines, purenormals = self.remove_duplicate_line(lines, source_image)
		purelines, image_intersections = self.calculate_intersections_as_matrix(purelines, img_shape=img.shape)

		return purelines, image_intersections


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


	def calculate_mapping_lines(self, M, idxes):
		'''
		image(u, v, 1) = M * standard(x, y, 1)^T
		When we get a homography M, we need calculate the lines mapped from the standard lines[idxes] to the lines on the image.
		:params: standard_lines: constant matrix
		:params: M: homography
		:params: idxes: standard line indexes we need to map.
		:return: lines: n * (x1, y1, x2, y2), mapping lines to the image.
		'''
		n = idxes.shape[0]
		sl = self.standard['lines'][idxes]

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
		mole = np.abs((y1 - y2) * x0 + (x1 - x2) * y0 - (x1 - x2) * (x1 + y1))
		deno = ((y1 - y2) ** 2 + (x1 - x2) ** 2) ** .5
		return mole / deno


	def calculate_score(self, imglines, maplines):
		'''
		for px in each (imgline, mapline), we calculate distance(px, mapline) and add all distance to get score.
		:params: imglines: we recognize lines from the image, n * (x1, y1, x2, y2)
		:params: maplines: using homography M, we map the standard_line to image, n * (x1, y1, x2, y2)
		:return: score: this mapping's score. We like lower score.
		'''
		score = 0
		for i in range(imglines.shape[0]):
			il, ml = imglines[i], maplines[i]
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
		for p in variations(self.standard_indexes, self.image['lines'].shape[0]):
			pi = np.array(p)
			# 1. get interpret points
			spoints = self.get_standard_intersection(pi)

			# 2. match standard points and image points
			imagePoints = []
			standPoints = []
			for li in spoints:
				for si in self.standard['intersections']:
					if li[0] == si[0] and li[1] == si[1]:
						imagePoints.append(li[2:])
						standPoints.append(si[2:])
			imagePoints = np.array(imagePoints)
			standPoints = np.array(standPoints)
			if imagePoints.shape[0] < 4:
				print("intersection not enough.")
				continue
			# 3. calculate homography matrix H
			M, mask = cv2.findHomography(imagePoints, standPoints, cv2.RANSAC, 3.0)
			if M is None:
				continue
			'''
			perspective = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
			cv2.imshow("perspective", perspective)
			print("get 1 result, end!")
			return
			'''
			# 4. get lines from standard model using M.
			# for each point(x, y) in standard model, we calculate (u, v, 1) = M*(x, y, 1)^T
			mapping_lines = self.calculate_mapping_lines(self.standard_lines, M, pi)
			# 5. calculate accuracy score of p, and get the most suitable one.
			score = self.calculate_score(self.image['lines'], mapping_lines)
			if score < s:
				s = score
				resM = M
				print(p, "score: ", s)

		# if pi[0] == 1 and pi[1] == 2 and pi[2] == 6 and pi[3] == 7 and pi[4] == 8:
		# 	print("---[1,2,6,7,8]: ---\n", spoints)
		# 	return resM, s
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
