import numpy as np

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

def distance_point_line(points, line):
    '''
    calculate the distance from points 'points' to a line 'line'.
    :params: points: [(x0, y0)]
    :params: line: (x1, y1, x2, y2)
    :return: distance
    '''
    x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - y1 * x2
    deno = (A ** 2 + B ** 2) ** .5
    # multi points
    if points.size > 2:
        moles = np.abs(A * points[:, 0] + B * points[:, 1] + C)
        return moles / deno
    
    # single point
    x0, y0 = points[0], points[1]
    mole = np.abs(A * x0 + B * y0 + C)
    return mole / deno

def get_y(line, x):
    '''
    :params: line: (x1, y1, x2, y2)
    :params: x: np.float32
    :return: y: (x, y) in line === (y1 - y2) / (x1 - x2) = (y - y1) / (x - x1)
    '''
    x_type = type(x)
    x1, y1, x2, y2 = line[:4]
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    avgy = (y1 + y2) / 2
    if B == 0:
        if x_type == np.ndarray:
            return np.ones((x.shape)) * avgy
        return avgy
    return -(A * x + C) / B

def get_x(line, y):
    '''
    :params: line: (x1, y1, x2, y2)
    :params: x: np.float32
    :return: y: (x, y) in line === (y1 - y2) / (x1 - x2) = (y - y1) / (x - x1)
    '''
    y_type = type(y)
    x1, y1, x2, y2 = line[:4]
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    avgx = (x1 + x2) / 2
    if A == 0:
        if y_type == np.ndarray:
            return np.ones((y.shape)) * avgx
        return avgx
    return (B * y - C) / A