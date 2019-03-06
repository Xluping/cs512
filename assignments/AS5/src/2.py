import cv2
import numpy as np


points = np.loadtxt("points.txt")

#obj_points = np.loadtxt("cali data/ncc-worldPt.txt")
#img_points = np.loadtxt("cali data/ncc-imagePt.txt")

obj_points = [] # 3d points in real world space
img_points = [] # 2d points in image plane.

for pt in points:
    obj_points.append([pt[0],pt[1],pt[2]]);
    img_points.append([pt[3],pt[4]]);

objpoints = np.array(obj_points)
imgpoints = np.array(img_points)


N = len(imgpoints)

M = np.zeros((2*N, 12), dtype=np.float64)
#M.shape

for i in range(0, N):
    X, Y, Z = objpoints[i] #model points
    u, v = imgpoints[i] #image points

    row_1 = np.array([ -X, -Y, -Z, -1, 0, 0, 0, 0, X*u, Y*u, Z*u, u])
    row_2 = np.array([ 0, 0, 0, 0, -X, -Y, -Z, -1, X*v, Y*v, Z*v, v])
    M[2*i] = row_1
    M[(2*i) + 1] = row_2

#    print ("p_model {0} \t p_obs {1}".format((X, Y), (u, v)))


u, sigma, vt = np.linalg.svd(M)
#vt.shape


M_norm = vt[np.argmin(sigma)]
#M_norm.shape

M_norm = M_norm.reshape(3, 4)

#normalized_h = M_norm / np.sqrt((np.sum(M_norm**2)))

a1 = M_norm[0 , 0:3]
a2 = M_norm[1 , 0:3]
a3 = M_norm[2 , 0:3]
b = M_norm[0:3 , 3]



#determine unknown parameters
#scale factor
#rho = np.linalg.norm(a3)
rho = 1/np.sqrt(a3[0]**2+a3[1]**2+a3[2]**2)

u0 = (np.absolute(rho)**2)*np.dot(a1,a3)
v0 = (np.absolute(rho)**2)*np.dot(a2,a3)
alpha_v = np.sqrt(((np.absolute(rho)**2)*np.dot(a2,a2))-(v0**2))

s = ((np.absolute(rho)**4)*np.dot(np.cross(a1,a3), np.cross(a2,a3)))/alpha_v

alpha_u = np.sqrt(((np.absolute(rho)**2)*np.dot(a1,a1))-(s**2)-(u0**2))

#K*
K_star = [[alpha_u, s, u0],
          [0, alpha_v, v0],
          [0, 0, 1]]
K_star = np.array(K_star)

#T*
T_star = np.absolute(rho)*np.matmul(np.linalg.inv(K_star),b)

r3 = np.absolute(rho) * a3
r1 = ((np.absolute(rho)**2)*np.cross(a2,a3))/alpha_v
r2 = np.cross(r3,r1)

#R*
R_star = [r1, r2, r3]
R_star = np.array(R_star)



m1 = M_norm[0 , 0:4]
m2 = M_norm[1 , 0:4]
m3 = M_norm[2 , 0:4]

objpoints2 = np.zeros(objpoints.shape)

objpoints2 = cv2.convertPointsToHomogeneous(objpoints)
objpoints2 = objpoints2.reshape(objpoints2.shape[0], objpoints2.shape[2])


total = 0
for i in range(0, len(imgpoints)):
    total = total + ((np.absolute(imgpoints[i][0]-(np.dot(m1,objpoints2[i])/np.dot(m3,objpoints2[i])))**2) + (np.absolute(imgpoints[i][1]-(np.dot(m2,objpoints2[i])/np.dot(m3,objpoints2[i])))**2))
    
error_rate = total/len(imgpoints)    




