import sys
sys.path.append('../')
import keras as K
from model import get_DronNet_model
from parse_data import Parse_helper
from generator import TrainImageGenerator, ValGenerator
from pathlib import Path
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adam
import tensorflow as tf
import os
import numpy as np
import _thread
#from show_gate_pose import Gate
import cv2
import time
class Predcit:
    def __init__(self,model_file = "checkpoints/weights.005-0.105.hdf5"):
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

if __name__== '__main__':
    pass #
    #model = get_DronNet_model(3)
    model = K.models.load_model(str(Path("checkpoints/weights.005-0.105.hdf5")))
    test_file = ["../../data/2019-06-11-16-23-15/"]
    image_paths=(list(Path(test_file[0]+'image/').glob("*.bmp")))
    image_paths= sorted(image_paths)
    print (image_paths)
    image_idx = 1
 
    
    #Gate_Handle = Gate()
    #set_pose = {'p_x':0,'p_y':0,'p_z':0,'r_x':0,'r_y':0,'r_z':0,\
    #            'p_x_gt':0,'p_y_gt':0,'p_z_gt':0,'r_x_gt':0,'r_y_gt':0,'r_z_gt':0}
    #try:
        #thread.start_new_thread(Gate_Handle.set_gate_pose , (set_pose,) )
    #    _thread.start_new_thread(Gate_Handle.start, () )
    #except:
    #    print ("Error: unable to start thread")

    for image_path in image_paths:
        parse = Parse_helper(test_file[0],image_path)
        pair_data = parse.read_pair()
        gate_num = parse.get_gate_num()
        image = pair_data['image']

        start = time.clock()
        pred = model.predict(np.expand_dims(image,0))

        elapsed = (time.clock() - start)
        print("Time used:",elapsed)
        #print (pair_data)
        pred = np.squeeze(pred)
        

        #set_pose['p_x'],set_pose['p_y'],set_pose['r_z']  = (p_y), (-p_x),yaw
 
        gt =  np.array([pair_data['pose']['x'],pair_data['pose']['y'],pair_data['pose']['z'],pair_data['pose']['yaw']],np.float)
        print('orig_pred_pose',pred)
        pred_dict = parse.get_predict_pose(pred)
        print ("gt_pose:",gt)
        print ("pred_pose:",np.array([pred_dict['pose']['x'],pred_dict['pose']['y'],pred_dict['pose']['z'],pred_dict['pose']['yaw']]))


        cv2.imshow("Image", parse.image)
        cv2.waitKey (0)
