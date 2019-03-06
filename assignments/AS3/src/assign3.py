import numpy as np
import cv2
import sys
import corners
from matplotlib import pyplot as plt

ramp_frames = 30
def get_image():

 	retval, image = cap.read()
 	return image

def defineCorners(image, window_size, k, threshold, scale):
	x = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=scale)
	y = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=scale)
	Ixx = x**2
	Ixy = y*x
	Iyy = y**2

	temp = []
	getCorners = []
	colorimg = image.copy()
	#colorimg = cv2.cvtColor(colorimg, cv2.COLOR_GRAY2RGB)
	size = window_size
	thres = threshold
	weight = k 
	getCorners = corners.harrisCorners(colorimg, Ixx,Iyy,Ixx,size, weight)

	getCorners.sort(reverse = True)
	temp = getCorners[:thres]
	#print "Finding corners......."
	for x in  temp:
		[z,xcor,ycor] =  x
		#print xcor,ycor
		colorimg = cv2.rectangle(colorimg, (xcor,ycor), (xcor+10,ycor+10),(255,0,0), -5)


	# print temp[:1]
	# print temp[:2]
	
	#for x in (x[1] for x in temp):
	#	print x
	#	for y in (y[2] for y in temp):
	#		print y
	#		colorimg = cv2.rectangle(colorimg, (x,y), (x+size,y+size),(255,0,0), 5)
	
	# for s in range(len(temp)):
	# 	x = temp[1:s]
	# 	y = temp[2:s]
	# 	colorimg = cv2.rectangle(colorimg, (x,y), (x+size,y+size),(255,0,0), 5)

	return colorimg
			
def cornerSlider(X):
	global image1
	gaussian = cv2.getTrackbarPos('Gaussian Scale',winName)
	size = cv2.getTrackbarPos('Neighbourhood Size',winName)    
	weight = cv2.getTrackbarPos('Weight',winName)
	threshold = cv2.getTrackbarPos('Threshold',winName)
	gaussian = gaussian * 2 + 1
	size = size * 2 + 1
	weight = weight / 100
	#print threshold
	threshold = (((threshold + 1) * image1.size) / 10000)
	#print threshold
	#print image1.size
	cornerimg = defineCorners(image1, size, weight, threshold, gaussian)
	
	cv2.imshow(winName,cornerimg)

def cornerSlider2(X):
	global image2
	gaussian = cv2.getTrackbarPos('Gaussian Scale',winName2)
	size = cv2.getTrackbarPos('Neighbourhood Size',winName2)    
	weight = cv2.getTrackbarPos('Weight',winName2)
	threshold = cv2.getTrackbarPos('Threshold',winName2)
	gaussian = gaussian * 2 + 1
	size = size * 2 + 1
	weight = weight / 100
	threshold = (((threshold + 1) * image1.size) / 10000)
	#print image
	cornerimg = defineCorners(image2, size, weight, threshold, gaussian)
	
	cv2.imshow(winName2,cornerimg)



# if (len(sys.argv) < 2):
cap = cv2.VideoCapture(0)
count = 0
while (True):

	ret, frame = cap.read()
	key = cv2.waitKey(5) & 255
    # Our operations on the frame come here
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Display the resulting frame
	cv2.imshow('frame',gray)

	if key == ord('c'):
		print 'Taking Image....'
		cv2.imwrite('%d.png' % count, gray)
		print 'Image Saved...'
		count += 1
	if key == 27:
		cap.release()
		cv2.destroyAllWindows()
		break

image1 = cv2.imread("0.png")
image1 = cv2.resize(image1, (640, 480))
#print "Resized..."
#image1 = cv2.line(image1, (0,0),(511,511),(255,0,0),5)
#print "drawn"
image2 = cv2.imread("1.png")
image2 = cv2.resize(image2, (640, 480))
# else:
# 	file1 = sys.argv[1]
# 	#file2 = sys.argv[2]
# 	image1 = cv2.imread(file1,0)
# 	#image2 = cv2.imread(file2,0)
# 	image1 = cv2.resize(image1, (640, 480))
# 	print "Resized..."
# 	image1 = cv2.line(image1, (0,0),(511,511),(255,0,0),5)
# 	print "drawn"
# 	#image2 = cv2.resize(image2, (image2.shape[1]/4, image2.shape[0]/4))


winName = 'Corner Image 1'
winName2 = 'Corner Image 2'
cv2.imshow(winName, image1)
cv2.createTrackbar('Gaussian Scale',winName,0,10,cornerSlider)
cv2.createTrackbar('Neighbourhood Size',winName,0,10,cornerSlider)
cv2.createTrackbar('Weight',winName,0,15,cornerSlider)
cv2.createTrackbar('Threshold',winName,0,10,cornerSlider)
cv2.imshow(winName2, image2)
cv2.createTrackbar('Gaussian Scale',winName2,0,10,cornerSlider2)
cv2.createTrackbar('Neighbourhood Size',winName2,0,10,cornerSlider2)
cv2.createTrackbar('Weight',winName2,0,15,cornerSlider2)
cv2.createTrackbar('Threshold',winName2,0,10,cornerSlider2)

# orb = cv2.ORB()
# kp1, des1 = orb.detectAndCompute(image1,None)
# kp2, des2 = orb.detectAndCompute(image2,None)

# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# # Match descriptors.
# matches = bf.match(des1,des2)

# # Sort them in the order of their distance.
# matches = sorted(matches, key = lambda x:x.distance)

# # Draw first 10 matches.
# img3 = cv2.drawMatches(image1,kp1,image2,kp2,matches[:10], flags=2)

# plt.imshow(img3),plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
