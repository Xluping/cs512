import numpy as np 
import cv2
import sys
import os

refPnt1 = []
refPnt2 = []
refPnt_left = []
refPnt_right = []
marking = True
imageFile1 = "point_file_marked1.txt"
imageFile2 = "point_file_marked2.txt"

def markPoints1(event,x,y,flags,params):
    global refPnt1, marking

    if event == cv2.EVENT_LBUTTONDOWN:
        marked_pt1 = cv2.circle(image1_resize,(x,y),4,(0,0,255),-1)
        refPnt1.append((x,y))
        cv2.imshow('Marking Image 1', marked_pt1)

def markPoints2(event,x,y,flags,params):
    global refPnt2, marking

    if event == cv2.EVENT_LBUTTONDOWN:
        marked_pt2 = cv2.circle(image2_resize,(x,y),4,(0,0,255),-1)
        refPnt2.append((x,y))
        cv2.imshow('Marking Image 2', marked_pt2)

def markPointsL(event,x,y,flags,params):
    global refPnt_left, marking

    if event == cv2.EVENT_LBUTTONDOWN:
        marked_left = cv2.circle(image_left,(x,y),4,(0,0,255),-1)
        refPnt_left.append((x,y))
        cv2.imshow('Left Image', marked_left)

def markPointsR(event,x,y,flags,params):
    global refPnt_right, marking

    if event == cv2.EVENT_LBUTTONDOWN:
        marked_right = cv2.circle(image_right,(x,y),4,(0,0,255),-1)
        refPnt_right.append((x,y))
        cv2.imshow('Right Image', marked_right)

def printHelp():
    print ("1 - Mark points in Image 1")
    print ("2 - Mark points in Image 2")
    print ("s - Save marked points for Image 1")
    print ("w - Save marked points for Image 2")
    print ("e - Get the estimated fundamental matrix from the marked images and reset images")
    print ("l - Enter the point from the left image to find the epipolar line for the right image")
    print ("t - Display the right image with epipolar line")
    print ("r - Enter the point from the right image to find the epipolar line for the left image")
    print ("j - Display the left image with epipolar line")
    print ("p - Print the epipoles in homogenous coordinates")
    print ("esc - Quit")

def readPoints(imageFile1, imageFile2):
    with open(imageFile1) as f:
        image1_pts = []
        for line in f:
            line = line.split()
            if line:
                image1_pts.append(line)

    with open(imageFile2) as f2:
        image2_pts = []
        for line in f2:
            line = line.split()
            if line:
                image2_pts.append(line)

    imagePnts1 = []
    imagePnts2 = []

    for li in image1_pts:
        li = map(float, li)
        imagePnts1.append(li)

    for li in image2_pts:
        li = map(float, li)
        imagePnts2.append(li)

    x = len(imagePnts1)
    y = len(imagePnts2)

    if x != y:
        print ("Mark the same number of points in both images")

    imagePnts1 = np.asarray(imagePnts1)
    imagePnts2 = np.asarray(imagePnts2)

    return imagePnts1, imagePnts2

def estimateFundamental(imagePnts1, imagePnts2):

    img_mean1 = np.mean(imagePnts1, axis = 0)
    img_mean2 = np.mean(imagePnts2, axis = 0)

    std1 = np.std(imagePnts1, axis = 0)
    std2 = np.std(imagePnts2, axis = 0)

    matrix_std1 = np.array([[std1[0], 0, 0], [0, std1[1], 0], [0,0,1]])
    matrix_std2 = np.array([[std2[0], 0, 0], [0, std2[1], 0], [0,0,1]])

    matrix_mean1 = np.array([[1, 0, -img_mean1[0]], [0, 1, -img_mean1[1]], [0,0,1]])
    matrix_mean2 = np.array([[1, 0, -img_mean2[0]], [0, 1, -img_mean2[1]], [0,0,1]])

    M1 = np.dot(matrix_std1,matrix_mean1)
    M2 = np.dot(matrix_std2,matrix_mean2)

    normalized_imagePnts1 =  []
    normalized_imagePnts2 = []

    for p1,p2 in zip(imagePnts1,imagePnts2):
        p1 = np.append(p1,[1], axis = 0)
        p2 = np.append(p2,[1], axis = 0)

        n1 = np.dot(M1, p1)
        n2 = np.dot(M2, p2)

        normalized_imagePnts1.append(n1)
        normalized_imagePnts2.append(n2)

    normalized_imagePnts1 = np.asarray(normalized_imagePnts1)
    normalized_imagePnts2 = np.asarray(normalized_imagePnts2)

    row = []

    for n1,n2 in zip(normalized_imagePnts1, normalized_imagePnts2):
        x1,y1 = n1[0], n1[1]
        x2,y2 = n2[0], n2[1]
        row.append([(x1*x2)]+[(x1*y2)]+[x1]+[(y1*x2)]+[(y1*y2)]+[y1]+[x2]+[y2]+[1])

    A = []

    for r in row:
        A.append(r)

    A = np.asarray(A)

    Ua, s, Va = np.linalg.svd(A, full_matrices=True)
    Va = np.transpose(Va)
	
    estimate = Va[:,8]
    F = estimate.reshape(3,3)

    U, D, V = np.linalg.svd(F, full_matrices=True)

    newD = []
    newD.append([D[0]]+[0]+[0])
    newD.append([0]+[D[1]]+[0])
    newD.append([0]+[0]+[0])

    newD = np.asarray(newD).reshape(3,3)

    newF = np.dot(U, np.dot(newD,V))

    finalF = np.dot(M1, np.dot(newF,M2))

    print ("Estimated Fundamental matrix is", finalF)

    return finalF

def epipole(F):
    U,D,V = np.linalg.svd(F, full_matrices=True)

    V = np.transpose(V)

    left_epipole = V[:,2]

    right_epipole = U[:,2]
	
    print ('Left Epipole', left_epipole)
    print ('Right Epipole', right_epipole)


file1 = raw_input("Provide image 1: ")
file2 = raw_input("Provide image 2: ")
image1_orig = cv2.imread(file1)
image2_orig = cv2.imread(file2)
#image1_bg = cv2.cvtColor(image1_orig, cv2.COLOR_BGR2GRAY)
#image2_bg = cv2.cvtColor(image2_orig, cv2.COLOR_BGR2GRAY)
image1_resize = cv2.resize(image1_orig, (640, 480))
image2_resize = cv2.resize(image2_orig, (640, 480))
cv2.imshow('Image 1', image1_resize)
cv2.imshow('Image 2', image2_resize)


while (True):

    key = cv2.waitKey(5) & 255

    if key == ord('1'):
        cv2.namedWindow('Marking Image 1')
        b = cv2.setMouseCallback('Marking Image 1',markPoints1)
        cv2.imshow('Marking Image 1', image1_resize)
        print ("Marked Image 1")

    if key == ord('2'):
        cv2.namedWindow('Marking Image 2')
        b = cv2.setMouseCallback('Marking Image 2',markPoints2)
        cv2.imshow('Marking Image 2', image2_resize)
        print ("Marked Image 2")

    if key == ord('s'):
        if(len(refPnt1) == 0):
            print ("Error: No point selected. Try again")
        else:
            with open("point_file_marked1.txt", "w") as pts1:
                for i in range(len(refPnt1)):
                    mark = refPnt1[i]
                    x = mark[0]
                    y = mark[1]
                    point = str(x) + " " + str(y)
                    pts1.write(point + "\n")

            pts1.close()

            print ("Saved Marked Image 1 Point File")

    if key == ord('w'):
        if(len(refPnt2) == 0):
            print ("Error: No point selected. Try again")
        else:
            with open("point_file_marked2.txt", "w") as pts2:
                for i in range(len(refPnt2)):
                    mark = refPnt2[i]
                    x = mark[0]
                    y = mark[1]
                    point = str(x) + " " + str(y)
                    pts2.write(point + "\n")
            
            pts2.close()

            print ("Saved Marked Image 2 Point File")
		
    if key == ord('e'):
        image_points1 , image_points2 = readPoints(imageFile1, imageFile2)

        matrixF = estimateFundamental(image_points1, image_points2)
		
        image_left = cv2.resize(image1_orig, (640, 480))
        image_right = cv2.resize(image2_orig, (640, 480))
		
        refPnt_left = []
        refPnt_right = []

    if key == ord('l'):
        cv2.destroyAllWindows()
        image_left = cv2.resize(image1_orig, (640, 480))
        refPnt_left = []
        cv2.namedWindow('Left Image')
        b = cv2.setMouseCallback('Left Image',markPointsL)
        cv2.imshow('Left Image', image_left)
 		
    if key == ord('t'):
        if(len(refPnt_left) == 0):
            print ("Error: No point selected. Try again")
        else:
            image_right = cv2.resize(image2_orig, (640, 480))
            rightImage = image_right.copy() 
            pntL = refPnt_left
            r,c = rightImage.shape[:2]

            for i in range(len(pntL)):
                mark = pntL[i]
                x = mark[0]
                y = mark[1]
                temp = []
                temp.append([x]+[y]+[1])
            temp = np.asarray(temp)
            print ('Point entered', temp)
            lineR = np.dot(temp,matrixF)
            lineR.reshape(-1,3)

            print ('Line computed', lineR)
            for j in range(len(lineR)):
                r = lineR[j]

                x0,y0 = map(int, [0, -r[2]/r[1] ])
                x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
                
                print (x0, y0)
                print (x1, y1)
            
            image_right_lined = cv2.line(rightImage, (x0,y0), (x1,y1), (0,0,255), 2)
            cv2.imshow('Epipole Right Image', image_right_lined)

    if key == ord('r'):
        cv2.destroyAllWindows()
        image_right = cv2.resize(image2_orig, (640, 480))
        refPnt_right = []
        cv2.namedWindow('Right Image')
        b = cv2.setMouseCallback('Right Image',markPointsR)
        cv2.imshow('Right Image', image_right)

    if key == ord('j'):
        if(len(refPnt_right) == 0):
            print ("Error: No point selected. Try again")
        else:
            image_left = cv2.resize(image1_orig, (640, 480))
            leftImage = image_left.copy()
            pntR = refPnt_right
            r,c = leftImage.shape[:2]

            for i in range(len(pntR)):
                mark = pntR[i]
                x = mark[0]
                y = mark[1]
                temp = []
                temp.append([x]+[y]+[1])
            
            temp = np.asarray(temp)
            print ('Point entered', temp)
            lineL = np.dot(temp,matrixF)
            lineL.reshape(-1,3)
            print ('Line computed', lineL)
            for j in range(len(lineL)):
                r = lineL[j]
                x0,y0 = map(int, [0, -r[2]/r[1] ])
                x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
			
            image_left_lined = cv2.line(leftImage, (x0,y0), (x1,y1), (0,0,255), 2)
            cv2.imshow('Epipole Left Image', image_left_lined)
 		
    if key == ord('p'):
        epipole(matrixF)

    if key == ord('h'):
        printHelp()
	
    if key == 27:
        break
        cv2.waitKey(0)
        cv2.destroyAllWindows()

