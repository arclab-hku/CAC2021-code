import sys
sys.path.append('../')
sys.path.append('../predict/')
import numpy as np
from geometry_msgs.msg import PoseStamped,Vector3
from generate_path import Generate_Path
from std_msgs.msg import Int32
from commander import Commander
from commander import Image_Capture
from show_gate_pose import Gate
from parse_data import Parse_helper
import pandas as pd
from pathlib  import Path
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
        self.mav_cur_pose = np.zeros(4)
        self.line_pd_dump = pd.DataFrame(np.zeros((self.h5_chunk_size,8)), columns = ["r","theta","phi","yaw","mav_x","mav_y","mav_z","mav_yaw"])
        rospy.init_node("pred_pose_node")
        rate = rospy.Rate(100)
        self.parse_data = Parse_helper()
        self.set_pose = {'p_x':0,'p_y':0,'p_z':0,'r_x':0,'r_y':0,'r_z':0,\
                        'p_x_gt':0,'p_y_gt':0,'p_z_gt':0,'r_x_gt':0,'r_y_gt':0,'r_z_gt':0,'gate_num':0}
        self.set_pub_pose = {'p_x':0,'p_y':0,'p_z':0,'r_x':0,'r_y':0,'r_z':0,\
                        'p_x_gt':0,'p_y_gt':0,'p_z_gt':0,'r_x_gt':0,'r_y_gt':0,'r_z_gt':0,'gate_num':0}
        self.pred_gate_pose_pub = rospy.Publisher('our/gate_pose_pred/pose', PoseStamped, queue_size=10)
        self.pred_gate_pose_show_pub = rospy.Publisher('our/gate_pose_pred/pose_show', PoseStamped, queue_size=10)
        self.pred_gate_for_path_pub = rospy.Publisher("our/gate_pose_pred/pose_for_path", PoseStamped, queue_size=10)
        self.gt_gate_pose_pub = rospy.Publisher('our/gate_pose_gt/pose', PoseStamped, queue_size=10)
        self.gate_num_pub = rospy.Publisher('our/gate/gate_num', Int32, queue_size=10)
        self.local_pose_sub = rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self.local_pose_callback)
        self.save_data_label = None
        self.b_switch_gate = False
        self.start = 0
        self.now_gate = 0
        self.debug = False
        self.clock_wise = False
        self.center_offset = 0.5 #the offset between the center of gate and the gate coordinate

        ###### clockwise, yaw is negative
        ###### Frame of VIO:
        ######        |x
        ######        |
        ######        |
        ###### y--------
        #the postion of all gates
        self.gate_pose_group = np.array([\
        #0 #about 45 degrees
        [1.33, 0.3, 1.3, 0, 0, np.rad2deg(0)],\
        #1
        [3.20, 0.31, 1.3, 0, 0, np.rad2deg(0.42)],\
        #2
        [4.90, 1.00, 1.4, 0, 0, np.rad2deg(0.87)],\
        #3
        [6.17, 2.22, 1.36, 0, 0, np.rad2deg(1.24)]])
        # #4
        # [22.6, 25.2, 1.93, 0, 0, np.rad2deg(2.0)],\
        # #5
        # [18.0, 30.5, 1.93, 0, 0, np.rad2deg(2.7)],\
        # #6
        # [11.3, 31.3, 1.93, 0, 0, np.rad2deg(-2.6)],\
        # #7
        # [ 6.6, 27.9, 1.93, 0, 0, np.rad2deg(-3)],\
        # #8
        # [ 0.88, 24.9, 1.93, 0, 0, np.rad2deg(-2.0)],\
        # #9
        # [ -0.19, 18.2, 1.93, 0, 0, np.rad2deg(-0.99)],\
        # #10
        # [ 4.4, 14.3, 1.93, 0, 0, np.rad2deg(-0.88)]])

        if self.clock_wise == True:
            self.gate_pose_group = self.gate_pose_group[::-1]
        
    '''
    deal the center offset of gate
    '''
    def tansfer_gate_center(self,gate_pose, orig_Corr = np.array([0,0,0])):
        # in real world, the offset of the gate is not existed
        assert len(gate_pose) == 6
        _yaw = gate_pose[5]
        y0 = gate_pose[1]
        x0 = gate_pose[0]
        # rotate_angle = _yaw if _yaw>=0 else 360+_yaw
        # Rotation = np.array([[np.cos(np.deg2rad(rotate_angle)),np.sin(np.deg2rad(rotate_angle)), 0],\
        #                     [-np.sin(np.deg2rad(rotate_angle)),np.cos(np.deg2rad(rotate_angle)),0],\
        #                     [0,0,1 ]])
        # new_Corr = np.dot(Rotation.transpose(),orig_Corr) 
        # return np.array([x0 + new_Corr[0] ,y0 + new_Corr[1],gate_pose[2],gate_pose[3],gate_pose[4],_yaw])
        return np.array([x0,y0,gate_pose[2],gate_pose[3],gate_pose[4],_yaw]) 

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
        self.pred_gate_pose_pub.publish(pred_pose_helper)
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
        self.pred_gate_pose_show_pub.publish(pred_pose_show_helper)

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
        self.gt_gate_pose_pub.publish(gt_pose_helper)
        #time.sleep(0.01)

    def collect_data(self,pose,mav_pose,img,count,circle_num):
        if self.debug  == False:
            dict_dump = {} #  map_dict = {'p_x':0,'p_y':1,'p_z':2,'Quaternion_x':3,'Quaternion_y':4,'Quaternion_z':5,'Quaternion_w':6} 
            #same with the xyzw quaternion, needed to be tranformed to polar coodinates or Euler Angles
            dict_dump['r'] = pose[0]
            dict_dump['theta'] = pose[1]
            dict_dump['phi'] =pose[2]
            dict_dump['yaw'] = pose[3]
            
            dict_dump['mav_x'] = mav_pose[0]
            dict_dump['mav_y'] = mav_pose[1]
            dict_dump['mav_z'] = mav_pose[2]
            dict_dump['mav_yaw'] = mav_pose[3]

            self.line_pd_dump.loc[count] =dict_dump
            print ("chunk_id:", self.chunk_id,"count:",count)
            global image_fname
            cv2.imwrite('../'+image_fname + '/image/' + 'camera_image_'+str(self.chunk_id)+'_'+str(count)+'_'+str(circle_num)+'_'+str(self.now_gate)+'.bmp',img.image_raw)
            #print (self.line_pd_dump.loc[count])

            if self.b_one_loop_completed == True:
                self.b_one_loop_completed = False
                path = '../'+pose_fname + '/pose/' +'pose_'+str(self.chunk_id)+'_'+str(circle_num)+'_' + str(circle_num)+'.h5'
                self.chunk_id = self.chunk_id + 1
                self.line_pd_dump.to_hdf(path,key = 'pose',mode='w')
                print ('wait 60 seconds')
                time.sleep(60)
                print ('completed')
                os._exit()
        else :
            pass

    
    def get_relavtive_pos(self,img):
        if self.local_pose ==None:
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
        gate_pose_forsave = self.tansfer_gate_center(gate_pose,np.array([0.0,0.0,0]))
        gate_pose= self.tansfer_gate_center(gate_pose)
        mav_pose =  np.array([dict_pos['Pos_x'],dict_pos['Pos_y'],dict_pos['Pos_z'],\
                            euler_angle[0],euler_angle[1],euler_angle[2]],np.float)

        self.mav_cur_pose = np.array([mav_pose[0],mav_pose[1],mav_pose[2],mav_pose[5]])
        # p_x = gate_pose[0] - mav_pose[0] 
        # p_y = gate_pose[1] - mav_pose[1] 
        # p_z = gate_pose[2] - mav_pose[2]
        
        #for training data
        vec = self.parse_data.generate_train_data(gate_pose_forsave,mav_pose)
        gt = vec[1]


        #for collecting data via judging the relative position
        vec = self.parse_data.generate_train_data(gate_pose,mav_pose)
        pose_gate_correct = vec[1]
        gt_r= pose_gate_correct[0] * (self.parse_data.get_r_max()-self.parse_data.get_r_min()) + self.parse_data.get_r_min()
        gt_theta = pose_gate_correct[1] * (self.parse_data.get_theta_max() - self.parse_data.get_theta_min()) + self.parse_data.get_theta_min()
        gt_phi = pose_gate_correct[2] * (self.parse_data.get_phi_max() - self.parse_data.get_phi_min()) +  self.parse_data.get_phi_min()
        yaw = pose_gate_correct[3] * (self.parse_data.get_yaw_max() -  self.parse_data.get_yaw_min()) +self.parse_data.get_yaw_min()

        gt_horizen_dis =  gt_r * np.cos(np.deg2rad(gt_theta))
        p_x_orig = gt_horizen_dis * np.cos(np.deg2rad(gt_phi)) #+ mav_pose[0]
        p_y_orig = gt_horizen_dis * np.sin(np.deg2rad(gt_phi)) #+ mav_pose[1]
        p_z_orig = gt_r * np.sin(np.deg2rad(gt_theta)) #+ mav_pose[2]
        
        #get the real relative position of gate (just rotate for judging the relative position between the gate and mav more conveniently)
        new_corr = self.parse_data.transformaiton_mav_to_world(np.array([p_x_orig,p_y_orig,p_z_orig]),mav_pose)
        p_x,p_y,p_z = new_corr[0],new_corr[1],new_corr[2]

        ## mav_pose [5] ->  the yaw
        # '''
        # here deal with the saltation 180<-0->-180
        # '''
        # tmp1 = 180 - math.fabs(gate_pose[5])
        # tmp2 = 180 - math.fabs(mav_pose[5])
        # yaw = gate_pose[5] - mav_pose[5] if (tmp1+tmp2 )> math.fabs(gate_pose[5] - mav_pose[5]) else tmp1+tmp2
        

        '''
        check the mav whether fly though a gate and switch its goal to next one
        '''
        dis = math.sqrt(pow(p_x,2)+pow(p_y,2))
        if(dis <= 0.5 and self.b_switch_gate == False):
                self.b_switch_gate = True
                self.start = time.time()
        
        if(self.b_switch_gate ==  True):
            delta_time = time.time() - self.start  
            if(delta_time > 0.1):
                self.b_switch_gate = False
                self.now_gate = self.now_gate + 1
                if (self.now_gate>len(self.gate_pose_group)-1):
                    self.now_gate = 0
                    self.b_one_loop_completed = True
                    self.collect_data(gt,self.mav_cur_pose,img,self.count,self.circle_num)
                    self.circle_num = self.circle_num + 1
                    self.count = 1
                    if (self.circle_num > 3):
                        os._exit()
                    
                '''
                redefine the goal
                '''
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
                gate_pose_forsave = self.tansfer_gate_center(gate_pose,np.array([0,0,0]))
                gate_pose= self.tansfer_gate_center(gate_pose)
                mav_pose =  np.array([dict_pos['Pos_x'],dict_pos['Pos_y'],dict_pos['Pos_z'],\
                                    euler_angle[0],euler_angle[1],euler_angle[2]],np.float)
                self.mav_cur_pose = np.array([mav_pose[0],mav_pose[1],mav_pose[2],mav_pose[5]])
                # p_x = gate_pose[0] - mav_pose[0]
                # p_y = gate_pose[1] - mav_pose[1]
                # p_z = gate_pose[2] - mav_pose[2]
                # yaw = gate_pose[5] - mav_pose[5]
                vec = self.parse_data.generate_train_data(gate_pose_forsave,mav_pose)
                gt = vec[1]

                vec = self.parse_data.generate_train_data(gate_pose,mav_pose)
                pose_gate_correct = vec[1]
                gt_r= pose_gate_correct[0] * (self.parse_data.get_r_max()-self.parse_data.get_r_min()) + self.parse_data.get_r_min()
                gt_theta = pose_gate_correct[1] * (self.parse_data.get_theta_max() - self.parse_data.get_theta_min()) + self.parse_data.get_theta_min()
                gt_phi = pose_gate_correct[2] * (self.parse_data.get_phi_max() - self.parse_data.get_phi_min()) +  self.parse_data.get_phi_min()
                yaw = pose_gate_correct[3] * (self.parse_data.get_yaw_max() -  self.parse_data.get_yaw_min()) +self.parse_data.get_yaw_min()

                gt_horizen_dis =  gt_r * np.sin(np.deg2rad(gt_theta))
                p_x_orig = gt_horizen_dis * np.cos(np.deg2rad(gt_phi)) #+ mav_pose[0]
                p_y_orig = gt_horizen_dis * np.sin(np.deg2rad(gt_phi)) #+ mav_pose[1]
                p_z_orig = gt_r * np.cos(np.deg2rad(gt_theta)) #+ mav_pose[2]
                new_corr = self.parse_data.transformaiton_mav_to_world(np.array([p_x_orig,p_y_orig,p_z_orig]),mav_pose)
                p_x,p_y,p_z = new_corr[0],new_corr[1],new_corr[2]

        pred_gate_pose_x = p_x + mav_pose[0]
        pred_gate_pose_y = p_y + mav_pose[1]
        pred_gate_pose_z = p_z + mav_pose[2]
        pred_gate_pose_yaw =  yaw + mav_pose[5]
        pred_gate_pose_yaw = -360 + pred_gate_pose_yaw if pred_gate_pose_yaw > 180 else pred_gate_pose_yaw
        pred_gate_pose_yaw =  360 + pred_gate_pose_yaw if pred_gate_pose_yaw <-180 else pred_gate_pose_yaw
        print ('~~~~~~~~~~~~~*************~~~~~~~~~~~~~~')
        print ('dis',dis)
        print ('pred',p_x, p_y, p_z, yaw)
        print ('mav_pose',mav_pose[0], mav_pose[1], mav_pose[2], mav_pose[5])
        print ('pred_pose',pred_gate_pose_x, pred_gate_pose_y, pred_gate_pose_z,pred_gate_pose_yaw)
        print ("gt_pose:",gate_pose_forsave[0],gate_pose_forsave[1],gate_pose_forsave[2],gate_pose_forsave[5])
        print ('self.b_switch_gate:',self.b_switch_gate)

        self.set_pose['p_x'], self.set_pose['p_y'],self.set_pose['p_z'],self.set_pose['r_z']  = p_x, p_y, p_z, yaw
        self.set_pub_pose['p_x'], self.set_pub_pose['p_y'],self.set_pub_pose['p_z'],self.set_pub_pose['r_z']  = p_x_orig, p_y_orig, p_z_orig, yaw

        self.set_pose['p_x_gt'],self.set_pose['p_y_gt'],self.set_pose['p_z_gt'],self.set_pose['r_z_gt']  = gate_pose[0],gate_pose[1],gate_pose[2],gate_pose[5]
        self.set_pub_pose['p_x_gt'],self.set_pub_pose['p_y_gt'],self.set_pub_pose['p_z_gt'],self.set_pub_pose['r_z_gt']  = gate_pose[0]/10,gate_pose[1]/10,gate_pose[2]/10,gate_pose[5]
        self.collect_data(gt,self.mav_cur_pose,img,self.count,self.circle_num)
        self.count = self.count + 1
        return np.array([p_x,p_y,p_z,yaw])
        
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
            print ("-----------------start-----------------")
            rev_pos = self.get_relavtive_pos(img)
            print ("rev_pos:",rev_pos)
            self.publish_gate_pose(self.set_pose,self.set_pub_pose) ##  time.sleep 0.01 delay 0.01s
            print ('~~~~~~~~~~~~~*************~~~~~~~~~~~~~~')

           

if __name__== '__main__':
    mav = Run_Circle()
    img = Image_Capture()
    pose_fname = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) 
    image_fname = pose_fname
    if mav.debug == False:
        os.makedirs ('../'+pose_fname +'/pose/')
        os.makedirs ('../'+image_fname + '/image/')

    
    while not rospy.is_shutdown():     
        #print ("-----------------start-----------------")
        mav.run(img)
        # if img.get_image() is not None:
        #     cv2.imshow("Camera", img.get_image())
        #     cv2.waitKey (1)
                
        
    
