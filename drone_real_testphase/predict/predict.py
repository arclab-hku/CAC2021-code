import sys
sys.path.append('../')
sys.path.append('../src/')
import numpy as np
from geometry_msgs.msg import PoseStamped,Vector3
from generate_path import Generate_Path
from std_msgs.msg import Int32
import keras as K
from commander import Commander
from commander import Image_Capture
from show_gate_pose import Gate
from parse_data import Parse_helper
import pandas as pd
from pathlib import Path
import time
import cv2
import h5py
import math
import rospy
import os
class Run_Circle:
    def __init__(self):
        self.optimal_path =  None
        self.local_pose = None
        self.update_path =  False
        self.b_one_loop_completed = False        
        self.h5_chunk_size = 32 ## 0 is excluded
        self.chunk_id = 0
        self.count = 1
        self.circle_num  = 1 
        self.line_pd_dump = pd.DataFrame(np.zeros((self.h5_chunk_size,4)), columns = ["r","theta","phi","yaw"])
        rospy.init_node("pred_pose_node")
        rate = rospy.Rate(100)
        self.parse_data = Parse_helper()
        self.set_pose = {'p_x':0,'p_y':0,'p_z':0,'r_x':0,'r_y':0,'r_z':0,\
                        'p_x_gt':0,'p_y_gt':0,'p_z_gt':0,'r_x_gt':0,'r_y_gt':0,'r_z_gt':0,'gate_num':0}
        self.set_pub_pose = {'p_x':0,'p_y':0,'p_z':0,'r_x':0,'r_y':0,'r_z':0,\
                        'p_x_gt':0,'p_y_gt':0,'p_z_gt':0,'r_x_gt':0,'r_y_gt':0,'r_z_gt':0,'gate_num':0}
        #self.pred_gate_pose_pub = rospy.Publisher('gi/gate_pose_pred/pose', PoseStamped, queue_size=10)
        #self.pred_gate_pose_show_pub = rospy.Publisher('our/gate_pose_pred/pose_show', PoseStamped, queue_size=10)
        self.pred_gate_for_path_pub = rospy.Publisher("our/gate_pose_pred/pose_for_path", PoseStamped, queue_size=10)
        #self.gt_gate_pose_pub = rospy.Publisher('our/gate_pose_gt/pose', PoseStamped, queue_size=10)
        self.gate_num_pub = rospy.Publisher('our/gate/gate_num', Int32, queue_size=10)
        self.local_pose_sub = rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self.local_pose_callback)
        self.model = K.models.load_model(str(Path("../models/weights.005-0.105.hdf5")))
        self.save_data_label = None
        self.b_switch_gate = False
        self.start = 0
        self.now_gate = 0
        
        self.center_offset = 0.5 #the offset between the center of gate and the gate coordinate
        #the postion of all gates
        self.gate_pose_group = np.array([\
         #0 #about 45 degrees
         [1.43, 0.5, 1.4, 0, 0, np.rad2deg(0.726)],\
         #1
         [3.45, 1.31, 1.55, 0, 0, np.rad2deg(0)],\
         #2
         [5.39, 1.97, 1.55, 0, 0, np.rad2deg(0.688)]])
    '''
    deal the center offet of gate
    '''
    def tansfer_gate_center(self,gate_pose):
        assert len(gate_pose) == 6
        _yaw = gate_pose[5]
        y0 = gate_pose[1]
        x0 = gate_pose[0]

        if (math.fabs(math.fabs(_yaw)-90)<0.001):
            k= 0
        elif (math.fabs(_yaw - 0) < 1) or (math.fabs(math.fabs(_yaw) - 180) < 1):
            k= 100
        else:
            k = -1/math.tan(np.deg2rad(_yaw))
        
        if _yaw >=0 and _yaw <180:
            x = x0 - math.sqrt(pow(self.center_offset,2)/(k*k+1))
            y = y0 + k*(x-x0) if k != 100 else y0 + self.center_offset
        else:
            x = x0 + math.sqrt(pow(self.center_offset,2)/(k*k+1))
            y = y0 + k*(x-x0) if k != 100 else y0 - self.center_offset
        return np.array([x,y,gate_pose[2],gate_pose[3],gate_pose[4],_yaw])


    def Obtain_offboard_node(self,**dictArg):
        self.local_pose = dictArg['pose']
    
    def local_pose_callback(self, msg):
        self.Obtain_offboard_node(pose = msg)

    def publish_gate_pose(self,gate_pose,pose_show):
        pred_pose_helper = PoseStamped()
        pred_pose_helper.header.stamp = rospy.Time.now()
        pred_pose_helper.header.frame_id = 'pred_gate_pose'
        pred_pose_helper.pose.position.x = gate_pose['p_x']
        pred_pose_helper.pose.position.y = gate_pose['p_y']
        pred_pose_helper.pose.position.z = gate_pose['p_z']
        pred_pose_helper.pose.orientation.x = gate_pose['r_x']
        pred_pose_helper.pose.orientation.y = gate_pose['r_y']
        pred_pose_helper.pose.orientation.z = gate_pose['r_z']
        pred_pose_helper.pose.orientation.w = gate_pose['gate_num']
        #self.pred_gate_pose_pub.publish(pred_pose_helper)
        self.pred_gate_for_path_pub.publish(pred_pose_helper)
        #time.sleep(0.01)

        pred_pose_show_helper = PoseStamped()
        pred_pose_show_helper.header.stamp = rospy.Time.now()
        pred_pose_show_helper.header.frame_id = 'pred_gate_pose'
        pred_pose_show_helper.pose.position.x = pose_show['p_x']
        pred_pose_show_helper.pose.position.y = pose_show['p_y']
        pred_pose_show_helper.pose.position.z = pose_show['p_z']
        pred_pose_show_helper.pose.orientation.x = pose_show['r_x']
        pred_pose_show_helper.pose.orientation.y = pose_show['r_y']
        pred_pose_show_helper.pose.orientation.z = pose_show['r_z']
        pred_pose_show_helper.pose.orientation.w = pose_show['gate_num']
        #self.pred_gate_pose_show_pub.publish(pred_pose_show_helper)

        gt_pose_helper = PoseStamped()
        gt_pose_helper.header.stamp = rospy.Time.now()
        gt_pose_helper.header.frame_id = 'gt_gate_pose'
        gt_pose_helper.pose.position.x = pose_show['p_x_gt']
        gt_pose_helper.pose.position.y = pose_show['p_y_gt']
        gt_pose_helper.pose.position.z = pose_show['p_z_gt']
        gt_pose_helper.pose.orientation.x = pose_show['r_x_gt']
        gt_pose_helper.pose.orientation.y = pose_show['r_y_gt']
        gt_pose_helper.pose.orientation.z = pose_show['r_z_gt']
        gt_pose_helper.pose.orientation.w = 0
        #self.gt_gate_pose_pub.publish(gt_pose_helper)
        #time.sleep(0.01)


    
    def get_relavtive_pos(self,image):
        if (self.local_pose == None):
           return None
        pos= self.local_pose
        #print(pos)
        dict_pos = {} 
        dict_pos['Pos_x'] = pos.pose.position.x
        dict_pos['Pos_y'] = pos.pose.position.y
        dict_pos['Pos_z'] = pos.pose.position.z
        dict_pos['Quaternion_x'] = pos.pose.orientation.x
        dict_pos['Quaternion_y'] = pos.pose.orientation.y
        dict_pos['Quaternion_z'] = pos.pose.orientation.z
        dict_pos['Quaternion_w'] = pos.pose.orientation.w
        q = np.array([dict_pos['Quaternion_w'],dict_pos['Quaternion_x'],dict_pos['Quaternion_y'],dict_pos['Quaternion_z']])
        euler_angle = self.quater_to_euler(q)

        gate_pose = self.gate_pose_group[self.now_gate]
        gate_pose = self.tansfer_gate_center(gate_pose)
        mav_pose =  np.array([dict_pos['Pos_x'],dict_pos['Pos_y'],dict_pos['Pos_z'],\
                            euler_angle[0],euler_angle[1],euler_angle[2]],np.float)

        # p_x = gate_pose[0] - mav_pose[0] 
        # p_y = gate_pose[1] - mav_pose[1] 
        # p_z = gate_pose[2] - mav_pose[2]
        #vec = self.parse_data.generate_train_data(gate_pose,mav_pose)
        #image = cv2.resize(image,(200,200),interpolation=cv2.INTER_CUBIC)
        vec = self.model.predict(np.expand_dims(image,0))

        pred_relative = np.squeeze(vec)
        pred_r= pred_relative[0] * (self.parse_data.get_r_max()-self.parse_data.get_r_min()) + self.parse_data.get_r_min()
        pred_theta = pred_relative[1] * (self.parse_data.get_theta_max() - self.parse_data.get_theta_min()) + self.parse_data.get_theta_min()
        pred_phi = pred_relative[2] * (self.parse_data.get_phi_max() - self.parse_data.get_phi_min()) +  self.parse_data.get_phi_min()
        pred_yaw = pred_relative[3] * (self.parse_data.get_yaw_max() -  self.parse_data.get_yaw_min()) +self.parse_data.get_yaw_min()

        pred_horizen_dis =  pred_r * np.cos(np.deg2rad(pred_theta))
        p_x_orig = pred_horizen_dis * np.cos(np.deg2rad(pred_phi)) #+ mav_pose[0]
        p_y_orig = pred_horizen_dis * np.sin(np.deg2rad(pred_phi)) #+ mav_pose[1]
        p_z_orig = pred_r * np.sin(np.deg2rad(pred_theta)) #+ mav_pose[2]
        new_corr = self.parse_data.transformaiton_mav_to_world(np.array([p_x_orig,p_y_orig,p_z_orig]),mav_pose)
        p_x,p_y,p_z,yaw = new_corr[0],new_corr[1],new_corr[2],pred_yaw

        pred_gate_pose_x = p_x + mav_pose[0]
        pred_gate_pose_y = p_y + mav_pose[1]
        pred_gate_pose_z = p_z + mav_pose[2]
        pred_gate_pose_yaw =  yaw + mav_pose[5]
        pred_gate_pose_yaw = -360 + pred_gate_pose_yaw if pred_gate_pose_yaw > 180 else pred_gate_pose_yaw
        pred_gate_pose_yaw =  360 + pred_gate_pose_yaw if pred_gate_pose_yaw <-180 else pred_gate_pose_yaw


        #pred_gate_pose_yaw = np.deg2rad(pred_gate_pose_yaw)

        # print ('~~~~~~~~~~~~~*************~~~~~~~~~~~~~~')
        #print ('dis',dis)
        #print ('pred',p_x, p_y, p_z, yaw)
        # print ('pred_orig',p_x_orig, p_y_orig, p_z_orig, yaw)
        # print ('raw_data',gt_r, gt_theta, gt_phi, yaw)
        # print ('mav_pose',mav_pose)
        # print ('pred_pose',pred_gate_pose_x, pred_gate_pose_y, pred_gate_pose_z)
        # print ("gt_pose:",gate_pose[0],gate_pose[1],gate_pose[2],gate_pose[5])
        # print ('self.b_switch_gate:',self.b_switch_gate)

        self.set_pose['p_x'], self.set_pose['p_y'],self.set_pose['p_z'],self.set_pose['r_z']  = pred_gate_pose_x, pred_gate_pose_y, pred_gate_pose_z, pred_gate_pose_yaw
        self.set_pub_pose['p_x'], self.set_pub_pose['p_y'],self.set_pub_pose['p_z'],self.set_pub_pose['r_z']  = p_x_orig, p_y_orig, p_z_orig, yaw

        self.set_pose['p_x_gt'],self.set_pose['p_y_gt'],self.set_pose['p_z_gt'],self.set_pose['r_z_gt']  = gate_pose[0],gate_pose[1],gate_pose[2],gate_pose[5]
        self.set_pub_pose['p_x_gt'],self.set_pub_pose['p_y_gt'],self.set_pub_pose['p_z_gt'],self.set_pub_pose['r_z_gt']  = gate_pose[0]/10,gate_pose[1]/10,gate_pose[2]/10,gate_pose[5]
        # self.collect_data(gt,img,self.count,self.circle_num)
        self.count = self.count + 1
        return np.array([pred_gate_pose_x,pred_gate_pose_y,pred_gate_pose_z,pred_gate_pose_yaw])
        
    '''
    quater to euler angle /degree
    '''
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
    def run(self,img):
        image = img.get_image()
        
        if image is not None:
            rev_pos = self.get_relavtive_pos(image)
            print ("rev_pos:",rev_pos)
            self.publish_gate_pose(self.set_pose,self.set_pub_pose) ##  time.sleep 0.01 delay 0.01s
            print ('~~~~~~~~~~~~~*************~~~~~~~~~~~~~~')

           

if __name__== '__main__':
    mav = Run_Circle()
    img = Image_Capture()
    #pose_fname = time.strftime("predict"+"%Y-%m-%d-%H-%M-%S", time.localtime()) 
    #image_fname = pose_fname
    # os.makedirs ('../../'+pose_fname +'/pose/')
    # os.makedirs ('../../'+image_fname + '/image/')

    while not rospy.is_shutdown():  
        mav.run(img)
        # if img.get_image() is not None:
        #     cv2.imshow("Camera", img.get_image())
        #     cv2.waitKey (1)
                
        
    
