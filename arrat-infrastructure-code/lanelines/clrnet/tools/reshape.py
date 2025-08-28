# This class reshapes images to desired image size. 
# An object of this class will take specific image size and convert to a pre-defined another image size
# The same class can be used to do coordinate transformations on any (x,y) data from re-sized image coordinates to original image coordinates.
# Limitation - resized image should have smaller resolution than the original image in both X AND Y 

# Prepared by: Punit Tulpule (transportaion research center)
# 09/16/22 
# Last edit 09/20/22: added self.im_ar==self.re_ar loop.


import cv2
import numpy as np

class reshape:
    def __init__(self,im_size,re_size):
        self.re_siz = re_size
        self.im_siz = im_size
        self.im_ar = im_size[1]/im_size[0]  #image aspect ratio
        self.re_ar = re_size[1]/re_size[0]  # Desired aspect ratio
        
        self.get_transform()
    
    def get_transform(self):
        
        x_crop = self.im_siz[1]
        y_crop = self.im_siz[0]
        
        if self.im_ar>self.re_ar:
            y_crop = self.im_siz[0]
            x_crop = self.im_siz[1]*self.re_ar/self.im_ar
        
        if self.im_ar<self.re_ar:
            x_crop = self.im_siz[1]
            y_crop = self.im_siz[0]*self.im_ar/self.re_ar
        
        
        
        
        LT = [int((self.im_siz[1]-x_crop)/2),int((self.im_siz[0]-y_crop)/2)]  #left top
        BR = [int(LT[0]+x_crop),int(LT[1]+y_crop)]                            #Bottom right
        
        
        r = y_crop/self.re_siz[0]   # ratio of pixel resolution (after image is cropped, the aspect ratio is the sam
        
        self.A = np.array([[r,0],[0,r]])
        self.B = -np.array(LT)
                
        
    def image(self,im):
        
        LT = -self.B
        BR = [self.im_siz[1]+self.B[0], self.im_siz[0]+self.B[1]]
        
        crp_img = im[LT[1]:BR[1],LT[0]:BR[0]]
        
        img_resz = cv2.resize(crp_img,[self.re_siz[1],self.re_siz[0]])
        
        
        return img_resz
    
    def data(self,dt):
        if dt.ndim==1:
            dt_new = np.dot(dt,self.A)-self.B
        else:
            dt_new = np.array([col -self.B for col in np.dot(dt,self.A)])
            
        data_t=np.rint(dt_new).astype(int)
        return data_t
        
