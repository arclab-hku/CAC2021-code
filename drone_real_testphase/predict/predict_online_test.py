import sys
sys.path.append('../')
sys.path.append('../src')
import rospy
from commander import Image_Capture
import keras as K
from geometry_msgs.msg import PoseStamped,Vector3
from model import get_DronNet_model
from parse_data import Parse_helper
#from generator import TrainImageGenerator, ValGenerator
from pathlib import Path
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adam
import tensorflow as tf
import os
import numpy as np
import _thread
import math
#from show_gate_pose import Gate
import cv2
import time
class Predcit:
    def __init__(self,model_file = "../models/weights.072-0.022.hdf5"):
        self.model = get_DronNet_model(3)
        self.model = K.models.load_model(str(Path(model_file)))
        #self.Gate_Handle = Gate()
        self.set_pose = {'p_x':0,'p_y':0,'p_z':0,'r_x':0,'r_y':0,'r_z':0,\
                'p_x_gt':0,'p_y_gt':0,'p_z_gt':0,'r_x_gt':0,'r_y_gt':0,'r_z_gt':0}
        try:
        #thread.start_new_thread(Gate_Handle.set_gate_pose , (set_pose,) )
            _thread.start_new_thread(self.Gate_Handle.start, () )
        except:
            print ("Error: unable to start thread")
    def get_predict(self,image):
        pred = self.model.predict(np.expand_dims(image,0))
        r = pred[0] * (parse.get_r_max()-parse.get_r_min()) + parse.get_r_min()
        theta = pred[1] * (parse.get_theta_max() - parse.get_theta_min()) + parse.get_theta_min()
        phi = pred[2] * (parse.get_phi_max() - parse.get_phi_min()) +  parse.get_phi_min()
        yaw = pred[3] * (parse.get_yaw_max() -  parse.get_yaw_min()) +parse.get_yaw_min()
        horizen_dis =  r * np.sin(np.deg2rad(theta))
        p_x = horizen_dis * np.cos(np.deg2rad(phi))
        p_y = horizen_dis * np.sin(np.deg2rad(phi)) # phi
        p_z = r * np.cos(np.deg2rad(theta))
        print ("pred_pose:",np.array([r,theta,phi,yaw]))
        self.set_pose['p_x'], self.set_pose['p_y'], self.set_pose['r_z']  = (p_y), (-p_x),yaw
        #show the gate in openGL
        #self.Gate_Handle.set_gate_pose(self.set_pose)
        return np.array([p_x,p_y,p_z,yaw])

class predict_online_test:
    def __init__(self):
        self.local_pose = None
        rospy.init_node("pred_pose_node")
        rate = rospy.Rate(100)
        local_pose_sub = rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self.local_pose_callback)

    def local_pose_callback(self,msg):
        self.Obtain_offboard_node(pose = msg)

    def Obtain_offboard_node(self,**dictArg):
        self.local_pose = dictArg['pose']

    def quater_to_euler(self,q):
        w = q[0]
        x = q[1]
        y = q[2]
        z = q[3]
        phi = math.atan2(2*(w*x+y*z),1-2*(x*x+y*y))
        theta = math.asin(2*(w*y-z*x))
        psi = math.atan2(2*(w*z+x*y),1-2*(z*z+y*y))

        Euler_Roll_x = phi*180/math.pi
        Euler_Pitch_y = theta*180/math.pi
        Euler_Yaw_z = psi*180/math.pi

        return (Euler_Roll_x,Euler_Pitch_y,Euler_Yaw_z)

if __name__== '__main__':
    pass #
    #model = get_DronNet_model(3)
    model = K.models.load_model(str(Path("../models/weights.072-0.022.hdf5")))
 
    
    #Gate_Handle = Gate()
    #set_pose = {'p_x':0,'p_y':0,'p_z':0,'r_x':0,'r_y':0,'r_z':0,\
    #            'p_x_gt':0,'p_y_gt':0,'p_z_gt':0,'r_x_gt':0,'r_y_gt':0,'r_z_gt':0}
    #try:
        #thread.start_new_thread(Gate_Handle.set_gate_pose , (set_pose,) )
    #    _thread.start_new_thread(Gate_Handle.start, () )
    #except:
    #    print ("Error: unable to start thread")

    parse = Parse_helper()
    p_handle = predict_online_test()
    img = Image_Capture()
    
    gt =  np.array([1.43, 0.5, 1.4, 0, 0, np.rad2deg(0.726)],np.float)
    
    while not rospy.is_shutdown():
        image = img.get_image()
        if image is not None:
            start = time.clock()
            pred = model.predict(np.expand_dims(image,0))

            elapsed = (time.clock() - start)
            print("Time used:",elapsed)
            #print (pair_data)
            pred_relative = np.squeeze(pred)
            
            mav_pose = np.zeros(6)

            pos = p_handle.local_pose
        
            q = np.array([pos.pose.orientation.w,pos.pose.orientation.x,pos.pose.orientation.y,pos.pose.orientation.z])
            euler_angle = p_handle.quater_to_euler(q)
            
            mav_pose[0] = pos.pose.position.x 
            mav_pose[1] = pos.pose.position.y
            mav_pose[2] = pos.pose.position.z
            mav_pose[5] = euler_angle[2]

            pred_r= pred_relative[0] * (parse.get_r_max()-parse.get_r_min()) + parse.get_r_min()
            pred_theta = pred_relative[1] * (parse.get_theta_max() - parse.get_theta_min()) + parse.get_theta_min()
            pred_phi = pred_relative[2] * (parse.get_phi_max() - parse.get_phi_min()) +  parse.get_phi_min()
            pred_yaw = pred_relative[3] * (parse.get_yaw_max() -  parse.get_yaw_min()) +parse.get_yaw_min()

            pred_horizen_dis =  pred_r * np.cos(np.deg2rad(pred_theta))
            p_x_orig = pred_horizen_dis * np.cos(np.deg2rad(pred_phi)) #+ mav_pose[0]
            p_y_orig = pred_horizen_dis * np.sin(np.deg2rad(pred_phi)) #+ mav_pose[1]
            p_z_orig = pred_r * np.sin(np.deg2rad(pred_theta)) #+ mav_pose[2]
            new_corr = parse.transformaiton_mav_to_world(np.array([p_x_orig,p_y_orig,p_z_orig]),mav_pose)
            p_x,p_y,p_z,yaw = new_corr[0],new_corr[1],new_corr[2],pred_yaw

            pred_gate_pose_x = p_x + mav_pose[0]
            pred_gate_pose_y = p_y + mav_pose[1]
            pred_gate_pose_z = p_z + mav_pose[2]
            pred_gate_pose_yaw =  yaw + mav_pose[5]
            pred_gate_pose_yaw = -360 + pred_gate_pose_yaw if pred_gate_pose_yaw > 180 else pred_gate_pose_yaw
            pred_gate_pose_yaw =  360 + pred_gate_pose_yaw if pred_gate_pose_yaw <-180 else pred_gate_pose_yaw

            
            print('pred_relative:',pred_relative)
            print ("gt_pose:",gt)
            print ("pred_pose:",np.array([pred_gate_pose_x,pred_gate_pose_y,pred_gate_pose_z,pred_gate_pose_yaw]))
