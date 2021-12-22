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


def get_l(g, tao, sigmal, sigmad):
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


def white_pixel_extract(img):
	# white pixel
	image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
	lower = np.uint8([0, 170, 20])
	upper = np.uint8([255, 255, 255])
	white_mask = cv2.inRange(image, lower, upper)
	# cv2.imwrite("mask.png", white_mask, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

	# remove dressed region
	c1 = get_l(white_mask, 10, 128, 20)
	# cv2.imwrite("candidate.png", c1, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

	# todo: remove textured region
	# c2 = cv2.cornerEigenValsAndVecs(c1, 5, 3)
	return c1

#
def draw_line(imgName, img, lines, coloridx=None):
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


def remove_duplicate_line(lines):
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
	angleThres = np.cos(np.pi / 240)
	dThres = 8
	for i, n1 in enumerate(normal):
		for j, n2 in enumerate(normal):
			if i < j and is_same_line(n1, n2, angleThres, dThres):
				ufs.union(i + 1, j + 1)

	# # draw lines, the same lines have the same color
	ufsidx = np.array(ufs.uf)[1:] - 1
	# print(coloridx)
	# result2 = img.copy()
	# draw_line("multiline.png", result2, lines, coloridx=coloridx)

	# remove duplicate lines, every color has only 1 line.
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
	return purelines, purenormals


def line_detection(img):
	"""
	:params: img: input image. we detecte lines from this image.
	:return: purelines: detected lines.
	"""
	# preprocess image
	mask = img
	height, width = mask.shape
	skel = np.zeros([height, width], dtype=np.uint8)  # [height,width,3]
	kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
	temp_nonzero = np.count_nonzero(mask)
	while (np.count_nonzero(mask) != 0):
		eroded = cv2.erode(mask, kernel)
		temp = cv2.dilate(eroded, kernel)
		temp = cv2.subtract(mask, temp)
		skel = cv2.bitwise_or(skel, temp)
		mask = eroded.copy()
	# cv2.imwrite("skel.png", skel, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

	# # option: dilate + erode
	# closed = cv2.morphologyEx(skel, cv2.MORPH_CLOSE, kernel)
	# cv2.imshow("closed", closed)

	# option: dilate
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
	dilt = cv2.dilate(skel, kernel, iterations=1)
	# cv2.imwrite("dilate.png", dilt, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

	# hough line detection
	edges = cv2.Canny(dilt, 50, 200, apertureSize=3)
	# cv2.imwrite("edges.png", edges, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
	lines = cv2.HoughLinesP(edges, 1, np.pi / 360, 15, minLineLength=270, maxLineGap=20)
	# print(lines.shape)

	lines = lines.reshape((lines.shape[0], -1))
	purelines, purenormals = remove_duplicate_line(lines)

	return purelines


def compute_intersection(line1, line2):
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


def compute_intersections_as_list(lines):
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
			ptx, pty = compute_intersection(l1, l2)
			if ptx != -1 and pty != -1:
				tmp.append([i, j, ptx, pty])
	points = np.array(tmp, dtype=np.int32)
	# if points.shape[0] > 2:
	#     points = points[np.lexsort([points[:, 1], points[:, 0]]), :]
	return points


def compute_intersections_as_matrix(slines):
	'''
	calculate intersection of lines in the standard model.
	:param: slines: n * 4 matrix, a list of line (x1, y1, x2, y2)
	:return: points: n * n * 2, points[i, j] = the intersection of line i and line j, and it's location is (x, y)
	'''
	n = slines.shape[0]
	points = np.zeros(n * n * 2, dtype='int').reshape((n, n, 2))
	for i, l1 in enumerate(slines):
		for j, l2 in enumerate(slines):
			if i >= j:
				continue
			points[i, j, 0], points[i, j, 1] = compute_intersection(l1, l2)
	return points


def get_standard_intersection(standard_points, idxes):
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
			ptx, pty = standard_points[l, r, 0], standard_points[l, r, 1]
			if ptx != -1 and pty != -1:
				tmp.append([i, j, ptx, pty])
	points = np.array(tmp, dtype='int')
	if points.shape[0] > 1:
		points = points[np.lexsort([points[:, 1], points[:, 0]]), :]
	return points


def calculate_mapping_lines(standard_lines, M, idxes):
	'''
	image(u, v, 1) = M * standard(x, y, 1)^T
	When we get a homography M, we need calculate the lines mapped from the standard lines[idxes] to the lines on the image.
	:params: standard_lines: constant matrix
	:params: M: homography
	:params: idxes: standard line indexes we need to map.
	:return: lines: n * (x1, y1, x2, y2), mapping lines to the image.
	'''
	n = idxes.shape[0]
	sl = standard_lines[idxes]

	point1 = np.hstack((sl[:, :2], np.ones((n, 1), dtype='int')))
	mapping1 = np.dot(M, point1.T)
	mapping1 = mapping1.T
	point2 = np.hstack((sl[:, 2:], np.ones((n, 1), dtype='int')))
	mapping2 = np.dot(M, point2.T)
	mapping2 = mapping2.T
	return np.concatenate((mapping1[:, :2], mapping2[:, :2]), axis=1)


def distance_point_line(point, line):
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


def calculate_score(imglines, maplines):
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


def model_fitting_nb(varias, points, image_lines, standard_points, standard_lines):
	"""
	We enumerate all series from standard model, and we need to calculate each series' score. Finally, we select score-max one as result.
	:type varias: np.array((m, n))
	"""
	s = 2147483647
	resM = np.zeros((3, 3), dtype=np.float16)
	for p in varias:
		# 1. get interpret points
		spoints = get_standard_intersection(standard_points, p)

		if spoints.shape[0] < 4:
			# print("this image don't have enough point, model fitting break!")
			continue
		# 2. match standard points and image points
		imagePoints = []
		standPoints = []
		for li in points:
			for si in spoints:
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
		mapping_lines = calculate_mapping_lines(standard_lines, M, p)
		# 5. calculate accuracy score of p, and get the most suitable one.
		score = calculate_score(image_lines, mapping_lines)
		if score < s:
			s = score
			resM = M
		print(p, "score: ", s)
	return resM, s


def model_fitting(image_lines, standard_indexes, standard_points, standard_lines):
	'''
	fit standard model
	:param: lines: detected lines from image
	:param: points: points[i, j] = intersection point of line i and line j
	:return:
	'''

	points = compute_intersections_as_list(image_lines)
	if points.shape[0] < 4:
		print("the image points not enough")
		return
	s = 2147483647
	resM = np.zeros((3, 3), dtype=np.float16)
	# get all possible line lists
	# for each line list:
	for p in variations(standard_indexes, image_lines.shape[0]):
		pi = np.array(p)
		# 1. get interpret points
		spoints = get_standard_intersection(standard_points, pi)

		# 2. match standard points and image points
		imagePoints = []
		standPoints = []
		for li in points:
			for si in spoints:
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
		mapping_lines = calculate_mapping_lines(standard_lines, M, pi)
		# 5. calculate accuracy score of p, and get the most suitable one.
		score = calculate_score(image_lines, mapping_lines)
		if score < s:
			s = score
			resM = M
			print(p, "score: ", s)

	# if pi[0] == 1 and pi[1] == 2 and pi[2] == 6 and pi[3] == 7 and pi[4] == 8:
	# 	print("---[1,2,6,7,8]: ---\n", spoints)
	# 	return resM, s
	return resM, s

def init_court_model(img):
	"""
	init court model from image
	:return: M: homography. image(u, v, w) = M \dot standard(x, y, 1)^T
	"""
	candidate = white_pixel_extract(img)
	lines = line_detection(candidate)
	standard_lines = np.array([[14, 29, 205, 29],
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
	standard_indexes = np.arange(12)
	standard_points = compute_intersections_as_matrix(standard_lines)

	result4 = img.copy()
	M, score = model_fitting(lines, standard_indexes=standard_indexes, standard_points=standard_points, standard_lines=standard_lines)
	perspective = cv2.warpPerspective(result4, M, (result4.shape[1], result4.shape[0]))
	#  cv2.imwrite("suitable.png", perspective, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

	return M

def is_in_court(ankle_points, M):
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
