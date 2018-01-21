from os.path import join, exists, abspath, dirname
from os import makedirs
import logging
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
from fit_3d import *


parser = argparse.ArgumentParser()
parser.add_argument('input_arg', type=str)
parser.add_argument('joint_arg', type=str)
parser.add_argument('output_arg', type=str)
args = parser.parse_args()




# Set up paths & load models.
# Assumes 'models' in the 'code/' directory where this file is in.
MODEL_DIR = join(abspath(dirname(__file__)), 'models')
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
# degree of view
do_degrees=[0.]
gender='male'
model = load_model(MODEL_MALE_PATH)
sph_regs = np.load(SPH_REGS_MALE_PATH)
est = np.load(args.joint_arg)['arr_0']
img_path = args.input_arg
out_path = args.output_arg+".pkl"
img = cv2.imread(img_path)
if img.ndim == 2:
    _LOGGER.warn("The image is grayscale!")
    img = np.dstack((img, img, img))
joints = est[:2, :].T
conf = est[2, :]
params, vis = run_single_fit(
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

with open(out_path, 'w') as outf:
    pickle.dump(params, outf)
if do_degrees is not None:
    cv2.imwrite(out_path.replace('.pkl', '.png'), vis[0])
