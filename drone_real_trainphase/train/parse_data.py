import cv2
import os
import h5py
import pandas as pd
import numpy as np
import math
import re
import glob
#from show_gate_pose import Gate
import _thread
from pathlib import Path
import random
class Parse_helper:
    
    def __init__(self,file_froup,img_file):
        pass
        self.file_group = file_froup
        self.file_path_img = img_file#['../image/2019-03-15-16-06-18','']
        #self.file_path_pose = pose_f#['../pose/2019-03-15-16-06-18','']
        self.image = None
        self.pose = None
        
        self.phi_max = 90
        self.phi_min = -90
        
        self.theta_max = 90
        self.theta_min = -90
        
        self.r_max = 7 #(horizon 6m, height 2m)
        self.r_min = 0.1 # for the minimum offset
        
        self.yaw_max = 180 #+-90
        self.yaw_min = -180
        
        self.now_gate = 0
        '''
        self.phi_max = 90
        self.phi_min = -90
        
        self.theta_max = 90
        self.theta_min = 0
        
        self.r_max = 6.5 #(horizon 6m, height 2m)
        self.r_min = 0.2 # for the minimum offset
        
        self.yaw_max = 90 #+-90
        self.yaw_min = -90
        '''
    
    def get_yaw_max(self):
        return self.yaw_max
    
    def get_horizon_max(self):
        return self.r_max
    
    def get_phi_max(self):
        return self.phi_max
    
    def get_theta_max(self):
        return self.theta_max

    def read_image_paths(self,idx_file = 0):
        pass
        img_paths = glob.glob(self.file_path_img[idx_file]+'/*.bmp')
        #print (img_paths)
        return img_paths
    def img_maksed(self,image):
        img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        low_red1 = (0, 43, 43)
        high_red1 = (18, 255, 255)
        
        low_red2 = (150, 43, 43)
        high_red2 = (180, 255, 255)
        mask1 = cv2.inRange(img_hsv, low_red1, high_red1)
        mask2 = cv2.inRange(img_hsv, low_red2, high_red2)
        mask = mask1+mask2
        return cv2.bitwise_and(image,image,mask)
    def read_pair(self):   
        img_path = str(self.file_path_img)
        img_path_tmp = img_path.rstrip('.bmp')
        #print (re.findall(r"\d+_\d+_\d+\.?\d*_\d+\.?\d*",img_path_tmp))
       
        info = re.findall(r"\d+_\d+_\d+\.?\d*_-?\d",img_path_tmp)[0].split('_')
        #print('info:',info)
        chunk_id,id_frame,circle_num,gate = info[0],info[1],info[2],info[3]
        self.now_gate = int(gate)
        pose_path = Path(self.file_group+'pose/pose_'+chunk_id+'_'+str(circle_num)+'_'+str(circle_num)+'.h5')
        print ('-----------------------------')
        print ('img_path:',str(img_path_tmp))
        print ('pose_path:',str(pose_path))
        pose_data = pd.read_hdf(str(pose_path), 'pose')
        self.pose = pose_data.loc[int(id_frame)]
        self.image = cv2.imread(img_path)
        #self.image = self.img_maksed(self.image)
        
        #self.image = cv2.resize(self.image,(200, 200), interpolation=cv2.INTER_CUBIC)
        
        #q = np.array([self.pose['Quaternion_w'],self.pose['Quaternion_x'],self.pose['Quaternion_y'],self.pose['Quaternion_z']])
        #print (q)
        #euler_angle = self.quater_to_euler(q)
        #print (euler_angle)
        
        pos = np.array([self.pose['r'],self.pose['theta'],self.pose['phi'],self.pose['yaw']])
        #print ('chunk_id:',chunk_id,'id_frame:',id_frame,'circle_num:',circle_num,'gate:',gate)
        return dict({'image':self.image,'pose':dict({'r':self.pose['r'],\
                                                'theta':self.pose['theta'],\
                                                'phi':self.pose['phi'],\
                                                'yaw':self.pose['yaw'],\
                                                })})
    def offset_pos(self,gate_position,mav_position):
         return gate_position - mav_position
        
    '''
    quater to euler angle /degree
    '''
    def quater_to_euler(self,q):
        w = q[0]
        x = q[1]
        y = q[2]
        z = q[3]
#-0.21735269  0.01972332  0.02020593 -0.97568475
        phi = math.atan2(2*(w*x+y*z),1-2*(x*x+y*y))
        theta = math.asin(2*(w*y-z*x))
        psi = math.atan2(2*(w*z+x*y),1-2*(z*z+y*y))

        Euler_Roll_x = phi*180/math.pi
        Euler_Pitch_y = theta*180/math.pi
        Euler_Yaw_z = psi*180/math.pi

        return (Euler_Roll_x,Euler_Pitch_y,Euler_Yaw_z)

    def generate_train_data(self,gate_pose,mav_pose):
        '''
        data_format: pos_x, pos_y, pos_z, r_x,_p_y, y_z
                      x
                      |
                      |
                      |
        y<------------  (phi is the angle with x,theta is the angle with height)
        '''
        horizon_dis = np.sqrt(pow(gate_pose[0]-mav_pose[0],2)+pow(gate_pose[1]-mav_pose[1],2))
        sin_phi = (gate_pose[1]-mav_pose[1])/horizon_dis
        phi = math.asin(sin_phi) * 180/math.pi 

        r = np.sqrt(pow(gate_pose[2]-mav_pose[2],2)+pow(horizon_dis,2)) 
        sin_theta = horizon_dis/r
        theta = math.asin(sin_theta) * 180/math.pi 

        yaw_delta = (gate_pose[5]-mav_pose[5])

        return np.array([[r,theta,phi,yaw_delta],[(r-self.r_min)/(self.r_max - self.r_min), (theta-self.theta_min)/(self.theta_max - self.theta_min), (phi - self.phi_min)/(self.phi_max - self.phi_min),(yaw_delta-self.yaw_min)/(self.yaw_max - self.yaw_min)],[sin_theta,sin_phi]])



if __name__ == '__main__':
    test_file = ["../data/2019-06-11-16-23-15/"]
    image_paths=(list(Path(test_file[0]+'image/').glob("*.bmp")))
    #print (image_paths)
    #Gate_Handle = Gate()
    set_pose = {'p_x':0,'p_y':0,'p_z':0,'r_x':0,'r_y':0,'r_z':0}
    #try:
        #thread.start_new_thread(Gate_Handle.set_gate_pose , (set_pose,) )
       #_thread.start_new_thread(Gate_Handle.start, () )
   # except:
        #print ("Error: unable to start thread")
    for image_path in image_paths:
        parse = Parse_helper(test_file[0],image_path)
        pair_data = parse.read_pair()

        gate_pose = np.array([1.43, 0.5, 1.4, 0, 0, np.rad2deg(0.726)])
        norm_vector = np.array([pair_data['pose']['x'],pair_data['pose']['y'],pair_data['pose']['z'],pair_data['pose']['yaw']],np.float)
        print ('gate_pose:',norm_vector)
        cv2.imshow("Image", parse.image)
        cv2.waitKey (0) 
    # while True:
        
    #     cv2.imshow("Image", parse.image)
    #     cv2.waitKey (0)  
    #     pass
        #print (pair_data['pose'])
