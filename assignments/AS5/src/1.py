import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


testimg = cv2.imread("test.png")
testimg = cv2.resize(testimg,(500,500))
gray = cv2.cvtColor(testimg,cv2.COLOR_BGR2GRAY)
patternsize = (7,7)

ret, corners = cv2.findChessboardCorners(gray, patternsize, None)
#cv2.imshow('img', testimg)
#cv2.waitKey()
#cv2.destroyAllWindows()

objp = np.zeros((7*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7, 0:7].T.reshape(-1 , 2)

objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
if(ret):
    
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria);
    
    objpoints.append(objp)
    imgpoints.append(corners2)

    cv2.drawChessboardCorners(testimg, patternsize, corners2, ret);
    cv2.imshow('img', testimg)
    cv2.waitKey()

cv2.destroyAllWindows()

with open("points.txt", "w") as f:
    for i in range(0, len(imgpoints[0])):
        f.write(str(objpoints[0][i][0])+" "+str(objpoints[0][i][0])+" "+str(objpoints[0][i][0])+" "+str(imgpoints[0][i][0][0])+" "+str(imgpoints[0][i][0][1])+"\n");
f.close()

#with open("img_pts.txt", "w") as f:
#    for pt in imgpoints[0]:
#        f.write(str(pt[0][0])+"\t"+str(pt[0][1])+"\n");
#f.close()
#
#with open("world_pts.txt", "w") as f:
#    for pt in objpoints[0]:
#        f.write(str(pt[0])+"\t"+str(pt[1])+"\t"+str(pt[2])+"\n");
#f.close()

#ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

