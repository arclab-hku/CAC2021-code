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
    
    def __init__(self,file_froup =None,img_file =None):
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

        self.gate_num  = 0 
    
    def get_yaw_max(self):
        return self.yaw_max
    def get_yaw_min(self):
        return self.yaw_min
    
    def get_r_max(self):
        return self.r_max
    def get_r_min(self):
        return self.r_min
    
    def get_phi_max(self):
        return self.phi_max
    def get_phi_min(self):
        return self.phi_min
    
    def get_theta_max(self):
        return self.theta_max
    def get_theta_min(self):
        return self.theta_min

    def read_image_paths(self,idx_file = 0):
        pass
        img_paths = glob.glob(self.file_path_img[idx_file]+'/*.bmp')
        #print (img_paths)
        return img_paths
        
    def get_gate_num(self):
        return self.gate_num

    def read_pair(self):   
        img_path = str(self.file_path_img)
        img_path_tmp = img_path.rstrip('.bmp')
        #print (re.findall(r"\d+_\d+_\d+\.?\d*_\d+\.?\d*",img_path_tmp))
        print (img_path_tmp)
        info = re.findall(r"\d+_\d+_\d+\.?\d*_-?\d",img_path_tmp)[0].split('_')
        #print('info:',info)
        chunk_id,id_frame,circle_num,gate = info[0],info[1],info[2],info[3]
        self.now_gate = int(gate)
        pose_path = Path(self.file_group+'pose/pose_'+chunk_id+'_'+str(circle_num)+'_'+str(circle_num)+'.h5')
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

        gt = np.array([self.pose['r'],self.pose['theta'],self.pose['phi'],self.pose['yaw']])
        gt_r= gt[0] * (self.get_r_max()-self.get_r_min()) + self.get_r_min()
        gt_theta = gt[1] * (self.get_theta_max() - self.get_theta_min()) + self.get_theta_min()
        gt_phi = gt[2] * (self.get_phi_max() - self.get_phi_min()) +  self.get_phi_min()
        yaw = gt[3] * (self.get_yaw_max() -  self.get_yaw_min()) +self.get_yaw_min()

        gt_horizen_dis =  gt_r * np.cos(np.deg2rad(gt_theta))
        p_x_orig = gt_horizen_dis * np.cos(np.deg2rad(gt_phi)) #+ mav_pose[0]
        p_y_orig = gt_horizen_dis * np.sin(np.deg2rad(gt_phi)) #+ mav_pose[1]
        p_z_orig = gt_r * np.sin(np.deg2rad(gt_theta)) #+ mav_pose[2]
        mav_pose = np.zeros(6)
        mav_pose[0],mav_pose[1],mav_pose[2],mav_pose[5] = self.pose['mav_x'],self.pose['mav_y'],self.pose['mav_z'],self.pose['mav_yaw']
        new_corr = self.transformaiton_mav_to_world(np.array([p_x_orig,p_y_orig,p_z_orig]),mav_pose)
        p_x,p_y,p_z = new_corr[0],new_corr[1],new_corr[2]
        pred_gate_pose_x = p_x + mav_pose[0]
        pred_gate_pose_y = p_y + mav_pose[1]
        pred_gate_pose_z = p_z + mav_pose[2]
        pred_gate_pose_yaw =  yaw + mav_pose[5]
        pred_gate_pose_yaw = -360 + pred_gate_pose_yaw if pred_gate_pose_yaw > 180 else pred_gate_pose_yaw
        pred_gate_pose_yaw =  360 + pred_gate_pose_yaw if pred_gate_pose_yaw <-180 else pred_gate_pose_yaw
        pred_gate_pose_yaw = np.deg2rad(pred_gate_pose_yaw)
        print ('chunk_id:',chunk_id,'id_frame:',id_frame,'circle_num:',circle_num,'gate:',gate)
        print ('relative_yaw',yaw,'now_mav_yaw',mav_pose[5])
        return dict({'image':self.image,'pose':dict({'x':pred_gate_pose_x,\
                                                'y':pred_gate_pose_y,\
                                                'z':pred_gate_pose_z,\
                                                'yaw':pred_gate_pose_yaw\
                                                })})
    def get_predict_pose(self,pred_relative):
        img_path = str(self.file_path_img)
        img_path_tmp = img_path.rstrip('.bmp')
        #print (re.findall(r"\d+_\d+_\d+\.?\d*_\d+\.?\d*",img_path_tmp))
        print (img_path_tmp)
        info = re.findall(r"\d+_\d+_\d+\.?\d*_-?\d",img_path_tmp)[0].split('_')
        #print('info:',info)
        chunk_id,id_frame,circle_num,gate = info[0],info[1],info[2],info[3]
        self.now_gate = int(gate)
        pose_path = Path(self.file_group+'pose/pose_'+chunk_id+'_'+str(circle_num)+'_'+str(circle_num)+'.h5')
        print ('pose_path:',str(pose_path))
        pose_data = pd.read_hdf(str(pose_path), 'pose')
        self.pose = pose_data.loc[int(id_frame)]
        
        mav_pose = np.zeros(6)
        mav_pose[0],mav_pose[1],mav_pose[2],mav_pose[5] = self.pose['mav_x'],self.pose['mav_y'],self.pose['mav_z'],self.pose['mav_yaw']

        pred_r= pred_relative[0] * (self.get_r_max()-self.get_r_min()) + self.get_r_min()
        pred_theta = pred_relative[1] * (self.get_theta_max() - self.get_theta_min()) + self.get_theta_min()
        pred_phi = pred_relative[2] * (self.get_phi_max() - self.get_phi_min()) +  self.get_phi_min()
        pred_yaw = pred_relative[3] * (self.get_yaw_max() -  self.get_yaw_min()) +self.get_yaw_min()

        pred_horizen_dis =  pred_r * np.cos(np.deg2rad(pred_theta))
        p_x_orig = pred_horizen_dis * np.cos(np.deg2rad(pred_phi)) #+ mav_pose[0]
        p_y_orig = pred_horizen_dis * np.sin(np.deg2rad(pred_phi)) #+ mav_pose[1]
        p_z_orig = pred_r * np.sin(np.deg2rad(pred_theta)) #+ mav_pose[2]
        new_corr = self.transformaiton_mav_to_world(np.array([p_x_orig,p_y_orig,p_z_orig]),mav_pose)
        p_x,p_y,p_z,yaw = new_corr[0],new_corr[1],new_corr[2],pred_yaw

        pred_gate_pose_x = p_x + mav_pose[0]
        pred_gate_pose_y = p_y + mav_pose[1]
        pred_gate_pose_z = p_z + mav_pose[2]
        pred_gate_pose_yaw =  yaw + mav_pose[5]
        pred_gate_pose_yaw = -360 + pred_gate_pose_yaw if pred_gate_pose_yaw > 180 else pred_gate_pose_yaw
        pred_gate_pose_yaw =  360 + pred_gate_pose_yaw if pred_gate_pose_yaw <-180 else pred_gate_pose_yaw
        pred_gate_pose_yaw = np.deg2rad(pred_gate_pose_yaw)
        return dict({'pose':dict({'x':pred_gate_pose_x,\
                                                'y':pred_gate_pose_y,\
                                                'z':pred_gate_pose_z,\
                                                'yaw':pred_gate_pose_yaw\
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

    def transformaiton_world_to_mav(self,gate,mav):
        yaw = mav[5]
        rotate_angle = yaw if yaw>=0 else 360+yaw
        Rotation = np.array([[np.cos(np.deg2rad(rotate_angle)),np.sin(np.deg2rad(rotate_angle)), 0],\
                             [-np.sin(np.deg2rad(rotate_angle)),np.cos(np.deg2rad(rotate_angle)),0],\
                             [0,0,1 ]])
        Ttranslation = np.array([-mav[0],-mav[1],-mav[2]])
        orig_Corr = np.array([gate[0],gate[1],gate[2]])
        new_Corr = np.dot(Rotation,orig_Corr + Ttranslation) 
        return new_Corr

    def transformaiton_mav_to_world(self,gate,mav):
        '''
        return relative coordiante in the world frame
        '''
        yaw = mav[5]
        rotate_angle = yaw if yaw>=0 else 360+yaw
        Rotation = np.array([[np.cos(np.deg2rad(rotate_angle)),np.sin(np.deg2rad(rotate_angle)), 0],\
                             [-np.sin(np.deg2rad(rotate_angle)),np.cos(np.deg2rad(rotate_angle)),0],\
                             [0,0,1 ]])
        Ttranslation = np.array([-mav[0],-mav[1],-mav[2]])
        orig_Corr = np.array([gate[0],gate[1],gate[2]])
        new_Corr = np.dot(Rotation.transpose(),orig_Corr) 

        return new_Corr

    def generate_train_data(self,gate_pose,mav_pose):
        '''
        data_format: pos_x, pos_y, pos_z, r_x,r_y, r_z
                      x
                      |
                      |
                      |
        y<------------  (phi is the angle with x,theta is the angle with height)
        '''
        
        '''
        here deal with the saltation 180<-0->-180
        ''' 
        tmp1 = 180 - math.fabs(gate_pose[5])
        tmp2 = 180 - math.fabs(mav_pose[5])
        # from 180->-180 is diffrent from -180->180
        sum_tmp = tmp1+tmp2 if gate_pose[5]<0 and mav_pose[5]>0 else -(tmp1+tmp2)
        yaw_delta = gate_pose[5] - mav_pose[5] if (tmp1+tmp2 )> math.fabs(gate_pose[5] - mav_pose[5]) else sum_tmp

        gate_corr = self.transformaiton_world_to_mav(gate_pose,mav_pose)

        r = np.sqrt(pow(gate_corr[0],2)+pow(gate_corr[1],2) + pow(gate_corr[2],2))

        sin_theta = gate_corr[2]/r
        theta = math.asin(sin_theta) * 180/math.pi 

        horizon_dis = np.sqrt(pow(gate_corr[0],2)+pow(gate_corr[1],2))
        sin_phi = gate_corr[1]/horizon_dis
        phi = math.asin(sin_phi) * 180/math.pi

        return np.array([[r,theta,phi,yaw_delta],[(r-self.r_min)/(self.r_max - self.r_min), (theta-self.theta_min)/(self.theta_max - self.theta_min), (phi - self.phi_min)/(self.phi_max - self.phi_min),(yaw_delta-self.yaw_min)/(self.yaw_max - self.yaw_min)]])



if __name__ == '__main__':
    test_file = ["../data/2019-06-13-19-28-41/"]
    image_paths=(list(Path(test_file[0]+'image/').glob("*.bmp")))
   
    for image_path in image_paths:
        parse = Parse_helper(test_file[0],image_path)
        pair_data = parse.read_pair()
        
        # gate_pose = np.array([10,10.5,1.93,0,0,0])
        # mav_pose =  np.array([pair_data['pose']['Pos_x'],pair_data['pose']['Pos_y'],pair_data['pose']['Pos_z'],\
        #                             pair_data['pose']['Roll_x'],pair_data['pose']['Pitch_y'],pair_data['pose']['Yaw_z']],np.float)
        # vectors = parse.generate_train_data(gate_pose,mav_pose)

        # norm_vector = vectors[1]
        norm_vector = np.array([pair_data['pose']['x'],pair_data['pose']['y'],pair_data['pose']['z'],pair_data['pose']['yaw']],np.float)
        print ('gate_pose:',norm_vector)

       
        # p_x = horizen_dis * np.cos(np.deg2rad(vectors[0][2]))
        # p_y = horizen_dis * np.sin(np.deg2rad(vectors[0][2])) # phi
       
        # set_pose['p_x'],set_pose['p_y'],set_pose['p_z']  = (p_y)/2, (-p_x)/2, 0
        # set_pose['r_x'],set_pose['r_y'],set_pose['r_z'] = pair_data['pose']['Roll_x'],\
        #                                                   pair_data['pose']['Pitch_y'],\
        #                                                   -pair_data['pose']['Yaw_z']
        
        # #print ('gt:',mav_pose)
        # Gate_Handle.set_gate_pose(set_pose)
        cv2.imshow("Image", parse.image)
        cv2.waitKey (0) 
    # while True:
        
    #     cv2.imshow("Image", parse.image)
    #     cv2.waitKey (0)  
    #     pass
        #print (pair_data['pose'])
