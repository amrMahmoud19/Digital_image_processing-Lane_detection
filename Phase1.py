#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


def PerspectiveTransform(src,dst):
    mat = cv2.getPerspectiveTransform(src,dst)
    mat_inv = cv2.getPerspectiveTransform(dst,src)
    return mat,mat_inv

def warpPerspective(img, mat, size):
    return cv2.warpPerspective(img, mat,size)

#function for converting the image to hls mode then return s as it is the one we are intrested in studying
def trans2hls(image):
    img_hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    (h,l,s) = cv2.split(img_hls)
    #MODIFIED
    return h,l,s


# In[3]:


#images must have the same dimension
#takes a list of images and produces an image of them concantenated in a 2D array-like shape
#need to take care of colored images 
def debug_mode(images , resize_factor):
    x_dim = int(images[0].shape[1] / resize_factor)
    y_dim = int(images[0].shape[0] / resize_factor)
    n_h_images = len(images)
    
    #if there are odd number of images add a dummy black image to the end of the list
    if (n_h_images % 2 )!= 0:
        images.append(np.zeros_like(images[n_h_images - 1]))    
        
    i = 0
    horiz_conc_image = list()
    while i < n_h_images:
        #if gray scale convert to RGB
        if len(images[i].shape) == 2:
            images[i] = cv2.resize(cv2.cvtColor(images[i],cv2.COLOR_GRAY2BGR), (x_dim , y_dim), interpolation= cv2.INTER_LINEAR)
        else:
            images[i] = cv2.resize(images[i], (x_dim , y_dim), interpolation= cv2.INTER_LINEAR)
        if len(images[i+1].shape) == 2:
            images[i + 1] = cv2.resize(cv2.cvtColor(images[i+1],cv2.COLOR_GRAY2BGR), (x_dim , y_dim), interpolation= cv2.INTER_LINEAR)
        else:
            images[i + 1] = cv2.resize(images[i+1], (x_dim , y_dim), interpolation= cv2.INTER_LINEAR)
        #concatenate each two successive images horizontally
        horiz_conc_image.append(cv2.hconcat([images[i] , images[i+1]]))
        i += 2
    #concatenate the resulting horizontally conc. images vertically
    debug_image = cv2.vconcat(horiz_conc_image)

    return debug_image


# In[4]:


def detect_edges(image , s_thresh , l_thresh , shad_thresh= (50,155)):
    (h,l,s) = trans2hls(image)
    
    canny_l = cv2.Canny(l,l_thresh[0],l_thresh[1])
    ext_shadow = np.zeros_like(h)
    ext_shadow[(l < shad_thresh[0]) & (s > shad_thresh[1])] = 1
    
    
#     canny_l[canny_l == 255] = 1
    s_binary = np.zeros_like(s)
    s_binary[(s >= s_thresh[0]) & (s <= s_thresh[1])] = 255
    
    combined_binary = np.zeros_like(canny_l)
    combined_binary[(s_binary == 255) | (canny_l == 255)] = 255
    combined_binary[ext_shadow == 1] = 0
    return combined_binary,s_binary,canny_l


# In[5]:


def sliding_window(img,dst_colored, window_size , right_poly_old, init , frame_discard):
    #window_size = (hight , width)
    #shape= (720 * 1280)
    #convert to colored img to draw colored line and windows on top of it
    
    out_img = cv2.cvtColor(img , cv2.COLOR_GRAY2RGB)
    
    nwindows = int(img.shape[0] / window_size[0])
   
    # find peaks of left and right lanes
    histogram = np.sum(img, axis=0)
    midpoint = int(histogram.shape[0]//2)
    start_left_x= np.argmax(histogram[:midpoint - 115])
    start_right_x = np.argmax(histogram[midpoint + 100:]) + midpoint + 100
    
    #get positions of white pixels in original img
    white_pixels = img.nonzero()
    white_x = np.array(white_pixels[1])
    white_y = np.array(white_pixels[0])

    
    # the left and right lane indices that we are going to find
    left_lane_indices = []
    right_lane_indices = []
    
    for window in range(nwindows):
        
        # find the boundary of each window
        win_bot = img.shape[0] - (window+1)*window_size[0]
        win_top = img.shape[0] - window*window_size[0]
        left_lane_lbound = start_left_x - window_size[1]
        left_lane_rbound = start_left_x + window_size[1]
        right_lane_lbound = start_right_x - window_size[1]
        right_lane_rbound = start_right_x + window_size[1]
        
        #draw the windows in red
        cv2.rectangle(dst_colored,(left_lane_lbound,win_bot),(left_lane_rbound,win_top),(255,0,0), 3) 
        cv2.rectangle(dst_colored,(right_lane_lbound,win_bot),(right_lane_rbound,win_top),(255,0,0), 3) 
        
        #locate the white pixels that lie within current window 
        good_left_inds = ((white_y >= win_bot) & (white_y < win_top) & 
        (white_x >= left_lane_lbound) &  (white_x < left_lane_rbound)).nonzero()[0]
        good_right_inds = ((white_y >= win_bot) & (white_y < win_top) & 
        (white_x >= right_lane_lbound) &  (white_x < right_lane_rbound)).nonzero()[0]
        
        left_lane_indices.append(good_left_inds)
        right_lane_indices.append(good_right_inds)
        
        #if the window contain black pixels don't shift it
        if len(good_left_inds) > 65:
            start_left_x = int(np.mean(white_x[good_left_inds]))
        if len(good_right_inds) > 65:        
            start_right_x = int(np.mean(white_x[good_right_inds]))

            
    left_lane_indices = np.concatenate(left_lane_indices)
    right_lane_indices = np.concatenate(right_lane_indices)

    
    leftx = white_x[left_lane_indices]
    lefty = white_y[left_lane_indices] 
    rightx = white_x[right_lane_indices]
    righty = white_y[right_lane_indices] 

    #fit a 2nd degree curve to the white pixels positions we found
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    
    #predict the current position
    ploty = np.linspace(0, 719, 720 )
    left_fitx = np.polyval(left_fit , ploty)
    right_fitx = np.polyval(right_fit , ploty)
    
    left_poly = np.asarray(tuple(zip(left_fitx,ploty)) ,np.int32)
    right_poly = np.asarray(tuple(zip(right_fitx,ploty)),np.int32)
    if init == True:
        if abs(right_poly_old[30][0] - right_poly[30][0]) > frame_discard:
            right_poly = right_poly_old
    
    #draw the lanes 
    cv2.polylines(dst_colored , [left_poly] , isClosed = False , color=(0,255,0) , thickness=50)
    cv2.polylines(dst_colored , [right_poly], isClosed = False , color=(0,0,255), thickness= 50)
    
    return dst_colored, right_poly,left_fitx , right_fitx,ploty


# In[6]:


def calc_off_dist(frame, right_lane_pts, left_lane_pts):                     #Calculating distance off center
        
        centre_car = (frame.shape[1]//2, 720)                                #determining center of car from original view
        
        #transforming centre coordinates into bird-eye view
        px = (M[0][0]*centre_car[0] + M[0][1]*centre_car[1] + M[0][2]) / ((M[2][0]*centre_car[0] + M[2][1]*centre_car[1] + M[2][2]))
        py = (M[1][0]*centre_car[0] + M[1][1]*centre_car[1] + M[1][2]) / ((M[2][0]*centre_car[0] + M[2][1]*centre_car[1] + M[2][2]))
        
        centre_car = (int(px), int(py))
        
        #getting left and right lane points indicating width of lane which is 3 meters
        right_lane_x = right_lane_pts[0]
        
        left_lane_x  = left_lane_pts[0]
        
        dst_px = abs(right_lane_x - left_lane_x)    #width of lane in pixels
        dst_real = 300                              #distance of lane in cm
        scale = dst_real/dst_px                     #scaling factor for transfroming distance from pixels to cm
        
        centre_lane = (dst_px//2 + left_lane_x)
        dst_off_px = abs(centre_lane - centre_car[0])  #distance off centre in pixels
        
        dst_off = dst_off_px*scale               #distance off centre in meters
        dst_off = round(dst_off/100,2)
        
        return dst_off


# In[ ]:





# In[7]:


input_name = "project_video"
path = r"C:\Users\amrmo\Documents\Digital_image_processing-Lane_detection\Project_data\\"+ input_name +".mp4"
cap = cv2.VideoCapture(path)
#Project_video Thresholds
# s_thresh = (160, 255) 
# l_thresh = (150 , 255)

challenge = False

#thresholding of s channel
#opt = 75
s_thresh = (75, 255) 
#canny
l_thresh = (140 , 255)
#shadow threshold
#(lightness , sat)
#inc l dec s
#200,100
shad_thresh = (150,100)

debug = 1
debug_resize = 3
ny_pipeline = 3
output_name = input_name + '_output'
size = (1280 , 720)
if debug == 1:
    pipeline = []
    output_name += '_debug'
    size = ((1280 // debug_resize) * 2 , (720 // debug_resize) * ny_pipeline)


if challenge == False:
    #for project video
    input_top_left = [550,468]
    input_top_right = [742,468]
    input_bottom_right = [1280,720]
    input_bottom_left = [128,720]
else:
    #for challenge video
    input_top_left = [580,500]
    input_top_right = [760,500]
    input_bottom_right = [1180,720]
    input_bottom_left = [170,720]

src_pt = np.float32([input_bottom_left,input_top_left,input_top_right,input_bottom_right])
dst_pt = np.float32([[0,720],[0,0],[1280,0],[1280,720]])
init = False
right_poly_old = np.zeros((720 , 2 , 2) , np.int32)

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter(output_name +'.mp4', fourcc, 25, size)

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        
        (combined_binary,l,s) = detect_edges(frame , s_thresh , l_thresh,shad_thresh)
#         kernel_op = np.asarray([[0 , 1, 0],
#                                [0, 1 , 0],
#                                [0 , 1, 0]] , np.uint8)
        kernel_co = np.ones((3,3) , np.uint8)
#         combined_binary = cv2.morphologyEx(combined_binary, cv2.MORPH_OPEN ,kernel_op)
        combined_binary = cv2.dilate(combined_binary,kernel_co)
        M,Minv = PerspectiveTransform(src_pt , dst_pt)
        dst = warpPerspective(combined_binary ,M , (1280 , 720))
                
#         dst_colored = perspective_warp(frame ,src=input_points , dst=p2)
        dst_colored = warpPerspective(frame ,M , (1280 , 720))
        out_img,right_poly_new ,left_fit , right_fit,ploty= sliding_window(dst , dst_colored, (72 , 128),right_poly_old,init,150)
        right_poly_old = right_poly_new
        
        left = np.array([np.transpose(np.vstack([left_fit, ploty]))])
        right = np.array([np.flipud(np.transpose(np.vstack([right_fit, ploty])))])
        points = np.hstack((left, right))
        
        
        init = True
        cv2.fillPoly(out_img,np.int_(points),color= (255,0,0))
        
        re_bird = warpPerspective(out_img , Minv , (1280,720) )
        
        #printing the off-centre distance on video frame
        dst_off = calc_off_dist(frame ,right_poly_new[350], left_poly[350])
        re_bird = cv2.putText(img=re_bird, text='The car is '+str(dst_off)+' off centre' , org=(0,100), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=2, color=(255, 255, 255),thickness=3)
        
        image = np.zeros_like(frame)
        cv2.addWeighted(frame, 0.5, re_bird, 0.5,0, image)
        
        if debug == 1:
            pipeline.append(l)
            pipeline.append(s)
            pipeline.append(combined_binary)
            pipeline.append(dst)
            pipeline.append(dst_colored)
            pipeline.append(image)
            image = debug_mode(pipeline , debug_resize)
            pipeline.clear()
        
        out.write(image)
        cv2.imshow('frame',image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()

out.release()

cv2.destroyAllWindows()

