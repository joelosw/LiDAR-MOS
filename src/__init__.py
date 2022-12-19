import sys
import os
TRAIN_PATH = "./src/model/"
DEPLOY_PATH = "./deploy/"
dir_path = os.path.dirname(os.path.realpath(__file__))
RPATH = os.path.normpath(os.path.join(dir_path, "../"))
sys.path.insert(0, TRAIN_PATH)