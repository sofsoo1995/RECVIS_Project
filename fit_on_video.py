from os.path import join, exists, abspath, dirname
from os import makedirs
import logging
import sys, os

sys.path.append(os.path.expanduser('/home/ubuntu/pynb/project/smpl/'))
import cPickle as pickle
from time import time
from glob import glob
import argparse

import cv2
import numpy as np
import chumpy as ch

from opendr.camera import ProjectPoints
from lib.robustifiers import GMOf
from smpl_webuser.serialization import load_model
from smpl_webuser.lbs import global_rigid_transformation
from smpl_webuser.verts import verts_decorated
from lib.sphere_collisions import SphereCollisions
from lib.max_mixture_prior import MaxMixtureCompletePrior
from render_model import render_model
from fit_3d_new import *
import matplotlib.pyplot as plt

_LOGGER = logging.getLogger(__name__) 
input_arg = '../../db/manipulation_videos/barbell_0002/'
list_img = sorted(glob(join(input_arg+'frames/', '*[0-9].jpg')))
path_joint = input_arg+'joint/'
output = input_arg+"3d/continue/"
# Set up paths & load models.
# Assumes 'models' in the 'code/' directory where this file is in.
MODEL_DIR = join('/home/ubuntu/pynb/project/smplify_public/code', 'models')
# Model paths:
MODEL_NEUTRAL_PATH = join(
    MODEL_DIR, 'basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')
MODEL_FEMALE_PATH = join(
    MODEL_DIR, 'basicModel_f_lbs_10_207_0_v1.0.0.pkl')
MODEL_MALE_PATH = join(MODEL_DIR,
                       'basicmodel_m_lbs_10_207_0_v1.0.0.pkl')

    # paths to the npz files storing the regressors for capsules
SPH_REGS_NEUTRAL_PATH = join(MODEL_DIR,
                             'regressors_locked_normalized_hybrid.npz')
SPH_REGS_FEMALE_PATH = join(MODEL_DIR,
                            'regressors_locked_normalized_female.npz')
SPH_REGS_MALE_PATH = join(MODEL_DIR,
                          'regressors_locked_normalized_male.npz')

# parameter of the main
n_betas=10
flength=5000.
pix_thsh=25.
viz=False
num = 0


do_degrees=[0.]
gender='male'
model = load_model(MODEL_MALE_PATH)
sph_regs = np.load(SPH_REGS_MALE_PATH)
# First we do estimation for one frame
est = np.load(join(path_joint, str(num)+'.npz'))['arr_0']
img = cv2.imread(list_img[num])
joints = est[:2, :].T
conf = est[2, :]
# 
t0 = time()
params, vis, cam,try_both_orient, orient = run_single_fit(
                    img,
                    joints,
                    conf,
                    model,
                regs=sph_regs,
                n_betas=n_betas,
                flength=flength,
                pix_thsh=pix_thsh,
                scale_factor=2,
                viz=viz,
                do_degrees=do_degrees)

print(time()-t0)
with open(output+'%04d.pkl'%num, 'w') as outf:                   
    pickle.dump(params, outf)                                                   
if do_degrees is not None:
    cv2.imwrite(output+'%04d.png'%num, vis[0]) 

lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
shape = params['betas']
k=num+1
for ind, img_p in enumerate(list_img[k:]):
    print(k+ind)
    if(os.path.isfile(join(path_joint,str(ind+k)+'.npz'))):
        new_est = np.load(join(path_joint,str(ind+k)+'.npz'))['arr_0']

        # interesting part TODO : using optical flow to track the joint

        new_joints = new_est[:2, :].T
        new_conf = new_est[2,:]
        img = cv2.imread(img_p) 
        # TODO function with camera
        old_pose = params['pose']
        params, vis, cam,try_both_orient, orient = run_single_fit(
            img,
            new_joints,
            new_conf,
            model,
            regs=sph_regs,
            n_betas=n_betas,
            flength=flength,
            pix_thsh=pix_thsh,
            scale_factor=2,
            viz=viz,
            do_degrees=do_degrees,
            old_pose=old_pose,
            betas=shape,
            is_continue=True, cam=cam,
            body_orient=orient, try_both_orient=try_both_orient)

                

    with open(output+'%04d.pkl'%(ind+k), 'w') as outf:                                                                                                        
        pickle.dump(params, outf)                                                   
    if do_degrees is not None:
        cv2.imwrite(output+'%04d.png'%(ind+k), vis[0]) 
    
        
#continuity = lambda w:w*ch.linalg.norm(A.dot(sv.pose-pose_old))**2 
