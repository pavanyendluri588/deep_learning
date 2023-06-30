#importing the modules 
import cv2
import numpy as np 

#loading the images 
path='E:\\deep_learning\\images\\house.jpg'
img1 = cv2.imread(path,1)
img0 = cv2.imread(path,0)
img_1 = cv2.imread(path,-1)

#creating the filter 
horizontal_filter =  np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
vertical_filter = np.array([[-1,0,1],[-2,0,2],[-3,0,3]])

filterd_image=cv2.filter2D(img1,-1,horizontal_filter)
cv2.imshow("original image",img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow("horizontal filterd image",filterd_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
filterd_image=cv2.filter2D(img0,-1,horizontal_filter)
cv2.imshow("original image",img0)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow("horizontal filterd image",filterd_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
filterd_image=cv2.filter2D(img_1,-1,horizontal_filter)
cv2.imshow("original image",img_1)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow("horizontal filterd image",filterd_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


filterd_image=cv2.filter2D(img1,-1,vertical_filter)
cv2.imshow("original image",img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow("horizontal filterd image",filterd_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
filterd_image=cv2.filter2D(img0,-1,vertical_filter)
cv2.imshow("original image",img0)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow("horizontal filterd image",filterd_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
filterd_image=cv2.filter2D(img_1,-1,vertical_filter)
cv2.imshow("original image",img_1)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow("horizontal filterd image",filterd_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
