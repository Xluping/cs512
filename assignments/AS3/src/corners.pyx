from __future__ import division
import numpy as np

def harrisCorners(image,Ixx,Iyy,Ixy,int window_size,int k):
	cdef int x, y, pad, height, width
	cdef float cornerness
	height = image.shape[0]
	width = image.shape[1]

	cornerList = []
	colorimg = image
	#colorimg = image.copy()
	pad = window_size

	print "Finding Corners....."
	for y in range(pad, height-pad):
		for x in range(pad, width-pad):
			windowIxx = Ixx[y-pad:y+pad+1, x-pad:x+pad+1]
			windowIxy = Ixy[y-pad:y+pad+1, x-pad:x+pad+1]
			windowIyy = Iyy[y-pad:y+pad+1, x-pad:x+pad+1]
			Sxx = windowIxx.sum()
			Sxy = windowIxy.sum()
			Syy = windowIyy.sum()

			M = np.matrix([[Sxx,Sxy],[Sxy,Syy]])
			#evalues = np.linalg.eig(M)
			#evals = evalues[0]

			det = np.linalg.det(M)
			trace = np.trace(M)
			cornerness = det - k * (trace ** 2)

			cornerList.append([cornerness, x, y])
	print "Corners Detected....."

	return cornerList