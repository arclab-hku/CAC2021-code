
import sys
sys.path.append('../')
sys.path.append('../predict')
from commander import Commander
from generate_path import Generate_Path
import quadrocoptertrajectory as quadtraj
from geometry_msgs.msg import PoseStamped,Vector3
import numpy as np
import copy
import queue
import time
import math
import rospy
import threading
import tty
import os
import termios
class myQueue:
    def __init__(self,size = 16):
        self.queue_list = []
        self.size = size
        self.front = 0
        self.rear = 0
    def isEmpty(self):
        return len(self.queue_list)==0
    def isFull(self):
        if (self.rear-self.front +1) == self.size:
            return True
        else:
            return False
    def first(self):
        if self.isEmpty():
            raise Exception("QueueIsEmpty")
        else:
            return self.queue_list[self.front]
    def last(self):
        if self.isEmpty():
            raise Exception("QueueIsEmpty")
        else:
            return self.queue_list[self.rear]
    def add(self,obj):
        if self.isFull():
            raise Exception("QueueOverFlow")
        else:
            self.queue_list.append(obj)
            self.rear += 1
    def clear(self):
        del self.queue_list[:]
        self.front = 0
        self.rear = 0

    def get(self):
        if self.isEmpty():
            raise Exception("QueueIsEmpty")
        else:
            self.rear -=1
            return self.queue_list.pop(0)
    def show(self):
        print(self.queue_list)
    def queue(self):
        return self.queue_list

class Execute_Class:
    def __init__(self):
        self.pose = {'p_x':0,'p_y':0,'p_z':0,'r_x':0,'r_y':0,'r_z':0}
        self.pred_r =0 # r parameter
        self.optimal_path = None
        
        self.Duration = 1.5
        self.time_interval = 0.02
        self.time_interval_execute = 0.02
        self.update_path = False
        self.con  = Commander()
        self.path_queue_size = 60
        self.goal_pose = np.zeros(4)
        self.mav_pose = np.zeros(6)
        self.path_queue = myQueue(self.path_queue_size)
        self.path_buff = myQueue(self.path_queue_size)
        self.refresh_buff_flag = False
       # rospy.init_node("execute_path_node")
        rate = rospy.Rate(100)
        self.path_generation_pub = rospy.Publisher('our/path/generation', Vector3, queue_size=10)
        self.movement_pub = rospy.Publisher('our/path/movement', Vector3, queue_size=10)
        self.pred_pose_sub = rospy.Subscriber("our/gate_pose_pred/pose_for_path", PoseStamped, self.pred_pose_callback)

        # Define the input limits:
        fmin = 1  #[m/s**2]
        fmax = 100 #[m/s**2]
        wmax = 100 #[rad/s]
        minTimeSec = 0.02 #[s]
        # Define how gravity lies:
        gravity = [0,0,-9.81]
        self.path_handle = Generate_Path(fmin,fmax,wmax, minTimeSec,gravity)

       
        
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
    def yaw_sigmoid_func(self,x):
        eps = 10e-5
        x = x + 5
        x = x
        s = np.log(x)
        return s
    def update_path_func(self):
        global lock
        cur_pos= self.con.get_current_mav_pose()
            
        dict_pos = {} 
        dict_pos['Pos_x'] = cur_pos.pose.position.x
        dict_pos['Pos_y'] = cur_pos.pose.position.y
        dict_pos['Pos_z'] = cur_pos.pose.position.z
        dict_pos['Quaternion_x'] = cur_pos.pose.orientation.x
        dict_pos['Quaternion_y'] = cur_pos.pose.orientation.y
        dict_pos['Quaternion_z'] = cur_pos.pose.orientation.z
        dict_pos['Quaternion_w'] = cur_pos.pose.orientation.w
        q = np.array([dict_pos['Quaternion_w'],dict_pos['Quaternion_x'],dict_pos['Quaternion_y'],dict_pos['Quaternion_z']])
        euler_angle = self.quater_to_euler(q)

        self.mav_pose  =  np.array([dict_pos['Pos_x'],dict_pos['Pos_y'],dict_pos['Pos_z'],\
                            euler_angle[0],euler_angle[1],euler_angle[2]],np.float)
    # print ('mav_pose:',mav_pose)
        '''
        the coordinate between the path planner and gazebo is different
        '''
        pos0 = [self.mav_pose [0], self.mav_pose [1], self.mav_pose [2]] #position
        vel0 = self.path_handle.generate_vel_from_yaw(self.mav_pose[5])
        acc0 = [0, 0, 0] #acceleration

        
        self.goal_pose [0] = self.mav_pose [0] + self.pose['p_x']
        self.goal_pose [1] = self.mav_pose [1] + self.pose['p_y']
        self.goal_pose [2] = np.clip(self.mav_pose[2] + self.pose['p_z'],1.2,1.5)
        self.goal_pose [3] = self.mav_pose [5] + self.pose['r_z']

        self.goal_pose [3] = -360 + self.goal_pose [3] if self.goal_pose [3] > 180 else self.goal_pose [3]
        self.goal_pose [3] =  360 + self.goal_pose [3] if self.goal_pose [3] <-180 else self.goal_pose [3]

        posf = [self.goal_pose [0],self.goal_pose [1], self.goal_pose [2]]  # position

        velf = self.path_handle.generate_vel_from_yaw(self.goal_pose[3])  # velocity
        accf = [0, 0, 0]  # acceleration
        #self.Duration = self.pred_r*1.1
        self.optimal_path = self.path_handle.get_paths_list(pos0,vel0,acc0,posf,velf,accf,self.Duration)
        print ('~~~~~~~~~~~~~*************~~~~~~~~~~~~~~')
        print ('jump_goal:',self.goal_pose)
        
        
        next_x = 0
        next_y = 0
        next_z = 0
        theta = 0
        start_t = (self.Duration - self.path_queue_size* self.time_interval_execute)/1.2
        stop_t  = start_t + self.path_queue_size* self.time_interval_execute
        # yaw_delta = self.pose['r_z']/self.path_queue_size
        # print ("self.pose['r_z']",self.pose['r_z'])
        yaw_a = self.pose['r_z']/(self.yaw_sigmoid_func(stop_t -  self.time_interval_execute)-self.yaw_sigmoid_func(start_t)) #[),so stop_t - time_interval
        yaw_b = self.mav_pose[5] - yaw_a * self.yaw_sigmoid_func(start_t)

        theta = self.mav_pose[5]
        
        for t in np.arange(start_t, stop_t, self.time_interval_execute):
            pass
            '''
            break down the present path and use the replanning path
            '''
            if(self.path_buff.isFull() == True):
                
                break
                
            next_x = np.float(self.optimal_path.get_position(t)[0])
            next_y = np.float(self.optimal_path.get_position(t)[1])
            next_z = 1.2#np.float(self.optimal_path.get_position(t)[2])
            theta = yaw_a * self.yaw_sigmoid_func(t) + yaw_b#
            # theta = theta + yaw_delta

            '''
            deal with the point of -180->180
            '''
            theta = -360 + theta if theta > 180 else theta
            theta =  360 + theta if theta <-180 else theta
            lock.acquire()
            self.path_buff.add(np.array([next_x,next_y,next_z,theta]))
            lock.release()
        
    def generate_path(self):
       
        while(1):
            
            
            if(self.refresh_buff_flag  == True ):
                self.refresh_buff_flag = False
                self.path_queue.clear()
                self.path_buff.clear()
                self.update_path_func()
                for i in self.path_buff.queue(): self.path_queue.add(i)
            else :
                self.path_buff.clear()
                self.update_path_func()
           
                #self.path_queue.show()
            
            
               
       
            
        

    def pred_pose_callback(self,msg):
        
        self.pose ['p_x'] = msg.pose.position.x
        self.pose ['p_y'] = msg.pose.position.y
        self.pose ['p_z'] = msg.pose.position.z
        self.pose ['r_x'] = msg.pose.orientation.x
        self.pose ['r_y'] = msg.pose.orientation.y
        self.pose ['r_z'] = msg.pose.orientation.z
        self.pred_r = np.sqrt (pow(self.pose ['p_x'],2)+pow(self.pose ['p_y'],2)+pow(self.pose ['p_z'],2))
        self.update_path = True
             ##start the generate thread
        #print ('raw_pose',self.pose)
       

    def pass_through(self):
        global lock
        if self.optimal_path is not None:       
       
            while(self.path_queue.isEmpty() == False):
                if (self.path_queue.isEmpty() == False):
                    path_tmp = self.path_queue.get()
                    # print ('next_piont:',path_tmp)
                    # print ('~~~~~~~~~~~~~*************~~~~~~~~~~~~~~')
                    self.con.move(path_tmp[0],path_tmp[1],path_tmp[2],path_tmp[3],False)
                    time.sleep(0.02)
            self.refresh_buff_flag = True

    def run(self):    
        global jump_once

        '''
        execute the generated path 
        '''
        
        if (self.update_path):
            self.pass_through()

            #self.gate_num_pub.publish(jump_once)
if __name__== '__main__':   

    jump_once = 1

    # theta = 0
    # r = 1.5
    # c_x,c_y = 8,11
    # start_yaw = -30
    # bias_x,bias_y = -0.1,0.5
    # start_x = c_x -r +bias_x
    # start_y = c_y + bias_y
    # sin_theta = np.sin(np.deg2rad(theta))
    # cos_theta = np.cos(np.deg2rad(theta))
    # print (sin_theta,cos_theta)
    # next_x = r-r * cos_theta + start_x
    # next_y = -r * sin_theta + start_y
    lock = threading.Lock()
    mav = Execute_Class()
    '''
    init the mav position 
    '''
    for i in range(10):
        mav.con.move(0,0,1.2,0,False)
        time.sleep(0.02)
    time.sleep(10)

    path_thread =threading.Thread(target=mav.generate_path)
    path_thread.setDaemon(True)
    path_thread.start()
    while not rospy.is_shutdown():
        mav.run()
    
