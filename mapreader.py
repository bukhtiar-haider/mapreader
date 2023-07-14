#-------------------------------------------------------------------------------
"""
This program accepts 1 command line argument (image name)
It takes an overhead image of a map on a blue table
Identifies a red pointer on the map
Returns the x,y location between (0 - 1)
Also returns the bearing of the pointer CW from North
It can be tested using the harness.py file for accuracy before deployment
'harness.py' has the ground truth values for comparison with the solution
"""
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

import cv2
import numpy as np
import math
import sys

#-------------------------------------------------------------------------------
# Functional routine definitions
#-------------------------------------------------------------------------------


def display(image): #Debugging purposes | To visualize algorithm effect
	cv2.imshow('img', image)
	cv2.waitKey(0)

def get_img_inf(filename):
	image = cv2.imread(filename) #Read image
	dims = image.shape #Matrix shape tells us image dimensions
	h = dims[0]
	w = dims[1]
	return image, h, w

def get_contours(image):
	#Pipeline
	#1. Grayscale
	_process = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	#2. Denoise using Gaussian Blur
	_process = cv2.GaussianBlur (_process, (15, 15), 0) 
	#3. Remove small features in the binary image using a morphological close.
	kernel = np.ones ((3,3), np.uint8)
	_process = cv2.erode (_process, kernel, iterations=3)
	#4. Threshold
	_process = cv2.adaptiveThreshold (_process, 255, \
	cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
	#5. Edge Detection
	_process = cv2.Canny(_process,100,200)
	#6. Denoise again
	_process = cv2.GaussianBlur (_process, (5, 5), 0)
	#7. Get Contours
	contours, junk = cv2.findContours (_process, cv2.RETR_EXTERNAL, \
	cv2.CHAIN_APPROX_SIMPLE)
	#cv2.drawContours (image, contours, -1, (0, 0, 255), 5)
	return contours
	
def get_largest_contour(contours):
	area = cv2.contourArea(contours[0])
	index = 0
	#Select index for largest contour
	#Artefacts may have caused irrelevant smaller contour
	for x in range (len(contours)):
		tmp = cv2.contourArea(contours[x])
		if(tmp>area):
			area = tmp
			index = x
	return contours[index]

#Countour approximation using Douglas-Peucker Algorithm
def contour_to_polygon(contour, image):
	# Parameter for epsilon value: 0.04 works well with map/pointer borders
	approx_percent = 0.04
	# Calculates Perimeter. Second arg specifies a closed curve
	epsilon = approx_percent * cv2.arcLength(contour, True) 
	corners = cv2.approxPolyDP(contour, epsilon, True)
	#Debugging Purposes || Black mask with the same dims as original image
	mask = np.zeros_like(image) 
	#Debugging Purposes || Draw polygon from corners
	cv2.polylines(mask, [corners], True, (0,0,255), 1, cv2.LINE_AA) 
	#display(mask)
	#display(image)
	return corners

def process_corners(corners):
	_corners = [[] for _ in range (len(corners))]
	for it in range (len(corners)):
		_corners[it].append(corners[it][0][0])
		_corners[it].append(corners[it][0][1])
	return sort_corners(np.asarray(_corners))
	
def sort_corners(processed_corners):
	#This function is taken from
	#https://pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
	rect = np.zeros((4, 2), dtype = "float32") #To store 4 2D coords
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = processed_corners.sum(axis = 1)
	rect[0] = processed_corners[np.argmin(s)]
	rect[2] = processed_corners[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(processed_corners, axis = 1)
	rect[1] = processed_corners[np.argmin(diff)]
	rect[3] = processed_corners[np.argmax(diff)]
	return rect
	
# For Debugging || Draws extracted corners on the image	
def draw_corners(sortedcorners, image):
	for obj in sortedcorners:
		cv2.circle(image, (int(obj[0]), int(obj[1])), 3, (255,255,255), -1)


#Calc/apply transformation matrix from map corners to image bounds
def crop_warp(sortedcorners, h, w, image):
	#Setting new coords to image bounds
	ls = [0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]
	dst = np.array([ls], dtype=np.float32)
	#Setting map corner coords from the sortedcorners object
	src = np.array(sortedcorners, dtype=np.float32)
	#Calculate transformation matrix
	t_matrix = cv2.getPerspectiveTransform(src, dst)
	#Applying transformation matrix
	#Interpolation flags INTER_LINEAR & INTER_CUBIC preferred for enlarging image
	newimage = cv2.warpPerspective(image, t_matrix, (int(w-1), int(h-1)), \
	flags=cv2.INTER_LINEAR)
	return newimage, cv2.cvtColor(newimage, cv2.COLOR_BGR2HSV)

#Identify arrow using hsv ranges
#Extract corresponding area out of a black mask to find contours	
def extract_arrow_contours(image, image_hsv):
	#Lower and Upper hsv bounds for the 'green' in the North arrow
	arrow_low = np.array([60,51,89])
	arrow_high = np.array([90,255,255])
	#Black background with green arrow extracted to identify contours
	mask = cv2.inRange(image_hsv, arrow_low, arrow_high) 
	mask = cv2.GaussianBlur (mask, (9, 9), 0) #Smoothing edges
	mask = cv2.GaussianBlur (mask, (3, 3), 0) #Smoothing edges
	contours, junk = cv2.findContours (mask, cv2.RETR_EXTERNAL, \
	cv2.CHAIN_APPROX_SIMPLE)
	#tmp = image.copy()
	#cv2.drawContours (tmp, contours, -1, (0, 0, 255), 5)
	#display(tmp)
	return contours
	
#Identify pointer using hsv ranges (two in this case)
#Extract corresponding areas out of black masks and combine them
def extract_pointer_contours(rotated_image, rotated_image_hsv):
	#Upper and lower HSV bounds for red pointer
	pointer_low = np.array([0, 51, 153 ]) 
	pointer_high = np.array([5, 255, 255]) 
	pointer_2 = np.array([175, 63, 140]) 
	pointer_2_ = np.array([180, 255, 255]) 
	#Calculate masks with each range
	mask = cv2.inRange(rotated_image_hsv, pointer_low, pointer_high)
	mask2 = cv2.inRange(rotated_image_hsv, pointer_2, pointer_2_)
	#display(mask)
	#display(mask2)
	mask = mask + mask2
	mask = cv2.GaussianBlur (mask, (9, 9), 0) #Smoothing edges
	mask = cv2.GaussianBlur (mask, (3, 3), 0) #Smoothing edges
	#display(mask)
	#Finding contour from combined masks
	contours, junk = cv2.findContours (mask, cv2.RETR_EXTERNAL, \
	cv2.CHAIN_APPROX_SIMPLE)
	tmp = rotated_image.copy()
	cv2.drawContours (tmp, contours, -1, (0, 0, 255), 5)
	#display(tmp)
	return contours

#Uses a point on the north pointing arrow and calculates distance to image edges
#Smallest distance is the corner it's closest to
#Returns false if its closest to top-right corner and true otherwise	
def closest_border(contour, h, w):
	topleft = math.dist(contour[0][0], [0, 0])
	topright = math.dist(contour[0][0], [0, w])
	botleft = math.dist(contour[0][0], [h, 0])
	botright = math.dist(contour[0][0], [h, w])
	if(not(topright > (topleft and botleft and botright))):
		return True

#Flips image based on a boolean switch
def check_and_flip(flip, image):
	if(flip):
		rotated = cv2.rotate(image, cv2.ROTATE_180)
		return rotated, cv2.cvtColor(rotated, cv2.COLOR_BGR2HSV)
	else:
		return image, cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

#Takes corners of an (almost)isocles triangle & identifies pointer tip and base
#The largest sum of distances from a point to the others is the pointer tip
def identify_pointer_tip(c):
	#'c' contains pointer corners
	first = math.dist(c[0][0], c[1][0]) + math.dist(c[0][0], c[2][0])
	second = math.dist(c[1][0], c[0][0]) + math.dist(c[1][0], c[2][0])
	third = math.dist(c[2][0], c[1][0]) + math.dist(c[2][0], c[0][0])
	ls = [first, second, third]
	head_index = ls.index(max(ls)) #Stores index of largest sum of distance values
	base_points = []
	for x in range (3):
		if (x != head_index):
			base_points.append(x)
	return head_index, base_points
	#create line from edge to midpoint of remaining points

#Object preprocessing
def points_to_np_arr(points):
	pt1 = np.array(points[0][0])
	pt2 = np.array(points[1][0])
	pt3 = np.array(points[2][0])
	return np.array([pt1, pt2, pt3])
	
#Calculates the midpoint for the base of the pointer triangle
#Construct a vector between the pointer head and the base midpoint
def midpoint(corners, baseindex):
	pt = (corners[baseindex[0]] + corners[baseindex[1]])/2
	return [int(pt[0]), int(pt[1])]

#Uses vector between pointer head and midpoint of base to identify bearing	
def calc_bearing(pointer, midpoint):
	#Construct vector between head and midpoint
	vec = np.array([pointer[0] - midpoint[0], pointer[1] - midpoint[1]])
	unit_vec = vec / np.linalg.norm(vec) #Convert to a unit vector
	#img coords are 0,0 in top left corner
	#north pointing vector is 0, -1 (instead of 0,1)
	dot_p = np.dot(unit_vec, [0, -1])
	angle = np.arccos(dot_p) #returns angle in radians
	deg = np.rad2deg(angle) #converting to degrees
	#np.arccos(dot) returns smaller angle (CW/AntiCW) between north and our vector
	#ifvector is in -ve 'x' quadrant, the angle was calculated anti-clockwise
	if(vec[0] < 1): #if vector lies in -ve 'x' quadrant
		deg = 360 - deg #subtract angle from '360' to obtain clockwise angle
	return deg

#uses pointer tip coordinates to calculte relative location on a 0-1 scale
def calc_coords(pointer, h, w):
	loc_x = pointer[0]/w
	loc_y = pointer[1]/h
	#flip y for coherence of matrix and output scales
	#matrix has y = 0 at the top of the image
	#output scale has y = 1 at the top of the image
	#( y = 1 - y ) to restore coherence
	return [loc_x, 1-loc_y]
	
#-------------------------------------------------------------------------------	
#-------------------------------------------------------------------------------
# This part packages the functional routines into bundles
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

def extract_map_corners(image):
	# Step 1/4. Gets a list of contours
	contours = get_contours(image)
	# Step 2/4. Returns the contour for the map
	map_contour = get_largest_contour(contours)
	# Step 3/4. Approximates a polygon over the contours and returns its corners
	unsorted_corners = contour_to_polygon(map_contour, image)
	# Step 4/4. Sorts the corners for the transformation function and returns them
	sorted_corners = process_corners(unsorted_corners)
	#draw_corners(sortedcorners, image) #Just for debugging purposes
	return sorted_corners
	
def fix_orientation(map_, map_hsv, h, w):
	# Step 1/4 Get contours using hsv ranges
	contours = extract_arrow_contours(map_, map_hsv)
	# Step 2/4 Extract contour corresponding to the 'North' arrow
	arrow_contour = get_largest_contour(contours)
	# Step 3/4 Check if arrow isn't on the top-right and image needs to be flipped
	need_flip = closest_border(arrow_contour, h, w)
	# Step 4/4 Return properly oriented map image
	rotated, rotated_hsv = check_and_flip(need_flip, map_)
	return rotated, rotated_hsv
	
def extract_pointer_corners(map_, map_hsv):
	# Step 1/4. Get contours from hsv ranges
	contours = extract_pointer_contours(map_, map_hsv)
	# Step 2/4. Get contour corresponding to the pointer
	pointer_contour = get_largest_contour(contours)
	# Step 3/4. Approximate polygon and get its corners
	raw_corners = contour_to_polygon(pointer_contour, map_)
	# Step 4/4. Identify and return the head/base points independantly
	_head, _base = identify_pointer_tip(raw_corners) 
	#converting corners list into np array
	corners = points_to_np_arr(raw_corners)
	return _head, _base, corners
	
def get_pointer_attr(corners, head, base, h, w):
	mid = midpoint(corners, base)
	bearing = calc_bearing(corners[head], mid)
	position = calc_coords(corners[head], h, w)
	return position[0], position[1], bearing
	
	
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Main program.
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

# Ensure we were invoked with a single argument.
if len (sys.argv) != 2:
    print ("Usage: %s <image-file>" % sys.argv[0], file=sys.stderr)
    exit (1)

#print ("The filename to work on is %s." % sys.argv[1])
raw_image, h, w = get_img_inf(sys.argv[1])

# Step 1. Identify map corner coordinates
sortedcorners = extract_map_corners(raw_image)

# Step 2. Crop and create new image out of the map
map_, maphsv = crop_warp(sortedcorners, h, w, raw_image)

# Step 3. Flip the image if required
map_oriented, maphsv_oriented = fix_orientation(map_, maphsv, h, w)

# Step 4. Extract pointer corners and identify the tip
_head, _base, _pointercorners = extract_pointer_corners(map_oriented, \
maphsv_oriented)

# Step 5. Get the pointer position and bearing
xpos, ypos, hdg = get_pointer_attr(_pointercorners, _head, _base, h, w)

# Step 6. Output the position and bearing
print ("POSITION %.3f %.3f" % (xpos, ypos))
print ("BEARING %.1f" % hdg)
