#!/usr/bin/env python3
# Developed by Xieyuanli Chen
# This file is covered by the LICENSE file in the root of this project.
# Brief: This script generates residual images

import os
import sys
import yaml
import numpy as np
from tqdm import tqdm
from tqdm.auto import trange
import matplotlib.pyplot as plt
import cv2

from utils import load_poses, load_calib, load_files, load_vertex

try:
  from c_gen_virtual_scan import gen_virtual_scan as range_projection
except:
  print("Using clib by $export PYTHONPATH=$PYTHONPATH:<path-to-library>")
  print("Currently using python-lib to generate range images.")
  from utils import range_projection


def generate_res_images(pose_file, calib_file, scan_folder, residual_image_folder, visualization_folder, range_image_params, num_last_n): 
  #print(f'Looking for poses in {pose_file}')
  poses = np.array(load_poses(pose_file))
  inv_frame0 = np.linalg.inv(poses[0])
  
  # load calibrations
  
  T_cam_velo = load_calib(calib_file)
  T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
  T_velo_cam = np.linalg.inv(T_cam_velo)
  
  # convert kitti poses from camera coord to LiDAR coord
  new_poses = []
  for pose in poses:
    new_poses.append(T_velo_cam.dot(inv_frame0).dot(pose).dot(T_cam_velo))
  sensor_poses = np.array(new_poses)
  
  # load LiDAR scans
  
  scan_paths = load_files(scan_folder)
  
  # test for the first N scans
  if num_frames >= len(sensor_poses) or num_frames <= 0:
    #print('generate training data for all frames with number of: ', len(sensor_poses))
    pass
  else:
    sensor_poses = sensor_poses[:num_frames]
    scan_paths = scan_paths[:num_frames]
  
  height = range_image_params['height']
  width =  range_image_params['width']
  fov_up = range_image_params['fov_up']
  fov_down =  range_image_params['fov_down']
  max_range =  range_image_params['max_range']
  min_range =  range_image_params['min_range']
  
  # generate residual images for the whole sequence
  for frame_idx in trange(len(scan_paths), desc="Frames in sequence", leave=False):
    file_name = os.path.join(residual_image_folder, str(frame_idx).zfill(6))
    diff_image = np.full((range_image_params['height'], range_image_params['width']), 0,
                             dtype=np.float32)  # [H,W] range (0 is no data)
    
    # for the first N frame we generate a dummy file
    if frame_idx < num_last_n:
      np.save(file_name, diff_image)
    
    else:
      # load current scan and generate current range image
      current_pose = sensor_poses[frame_idx]
      current_scan = load_vertex(scan_paths[frame_idx])
      current_range = range_projection(current_scan.astype(np.float32),
                                       height, width, fov_up, fov_down, max_range, min_range)[:, :, 3].clip(0,max_range)
      # load last scan, transform into the current coord and generate a transformed last range image
      last_pose = sensor_poses[frame_idx - num_last_n]
      last_scan = load_vertex(scan_paths[frame_idx - num_last_n])
      last_scan_transformed = np.linalg.inv(current_pose).dot(last_pose).dot(last_scan.T).T
      last_range_transformed = range_projection(last_scan_transformed.astype(np.float32), height, width, fov_up, fov_down, max_range, min_range)[:, :, 3].clip(0,max_range)
      
      # generate residual image
      valid_mask = (current_range > min_range) & \
                   (current_range < max_range) & \
                   (last_range_transformed > min_range) & \
                   (last_range_transformed < max_range)
      difference = np.abs(current_range[valid_mask] - last_range_transformed[valid_mask])
      
      if normalize:
        difference = np.abs(current_range[valid_mask] - last_range_transformed[valid_mask]) / current_range[valid_mask]

      diff_image[valid_mask] = difference
      
      if debug:
        fig, axs = plt.subplots(3)
        diff_image_marked = np.zeros(list(last_range_transformed.shape) + [3])
        diff_image_marked[:,:,0] = 0.99
        diff_image_marked[valid_mask,0] = diff_image[valid_mask]
        diff_image_marked[valid_mask,1] = diff_image[valid_mask]
        diff_image_marked[valid_mask,2] = diff_image[valid_mask]
        axs[0].set_title("Last Scan transformed")
        axs[0].imshow(last_range_transformed)
        axs[1].set_title("Current Scan")
        axs[1].imshow(current_range)
        axs[2].set_title("Diff Image (Invalid Red)")
        axs[2].imshow(diff_image_marked, vmin=0, vmax=10)
        fig.tight_layout()
        plt.show()
        combined_image = np.concatenate([current_range, last_range_transformed, diff_image],axis=0)
        plt.imshow(combined_image)
        plt.show()


        
      if visualize:
        diff_image_marked = np.zeros(list(last_range_transformed.shape) + [3])
        diff_image_scaled = diff_image/diff_image.max()
        diff_image_marked[:,:,2] = 0.99
        diff_image_marked[valid_mask,0] = diff_image_scaled[valid_mask]
        diff_image_marked[valid_mask,1] = diff_image_scaled[valid_mask]
        diff_image_marked[valid_mask,2] = diff_image_scaled[valid_mask]
        image_name = os.path.join(visualization_folder, str(frame_idx).zfill(6) + "_diff.png")
        #cv2.imshow("Bla", (diff_image_marked*255).astype(np.uint8))
        #cv2.imwrite(image_name, (diff_image_marked*255).astype(np.uint8))
        valid_mask = (current_range > range_image_params['min_range']) & \
                      (current_range < range_image_params['max_range'])
        current_range_scaled = current_range/current_range.max()
        current_image_marked = np.zeros(list(last_range_transformed.shape) + [3])
        current_image_marked[:,:,2] = 0.99
        current_image_marked[valid_mask,0] = current_range_scaled[valid_mask]
        current_image_marked[valid_mask,1] = current_range_scaled[valid_mask]
        current_image_marked[valid_mask,2] = current_range_scaled[valid_mask]
        image_name = os.path.join(visualization_folder, str(frame_idx).zfill(6) + "_current.png")
        #cv2.imwrite(image_name, (current_image_marked*255).astype(np.uint8))
        valid_mask = (last_range_transformed > range_image_params['min_range']) & \
                      (last_range_transformed < range_image_params['max_range'])
        last_range_transformed_scaled = last_range_transformed/last_range_transformed.max()
        transformed_image_marked = np.zeros(list(last_range_transformed.shape) + [3])
        transformed_image_marked[:,:,2] = 0.99
        transformed_image_marked[valid_mask,0] = last_range_transformed_scaled[valid_mask]
        transformed_image_marked[valid_mask,1] = last_range_transformed_scaled[valid_mask]
        transformed_image_marked[valid_mask,2] = last_range_transformed_scaled[valid_mask]
        image_name = os.path.join(visualization_folder, str(frame_idx).zfill(6) + "_transformed.png")
        #cv2.imwrite(image_name, (transformed_image_marked*255).astype(np.uint8))

        combined_image = np.concatenate([current_range_scaled*255, last_range_transformed_scaled*255, diff_image_scaled*255],axis=0).astype(np.uint8)
        cv2.imwrite(os.path.join(visualization_folder, str(frame_idx).zfill(6) + '_combined.png'), combined_image)

      # save residual image
      np.save(file_name, diff_image)


if __name__ == '__main__':
  # load config file
  config_filename = 'config/master_config.yml'
  if len(sys.argv) > 1:
    config_filename = sys.argv[1]
  
  master_config = yaml.load(open(config_filename), Loader=yaml.FullLoader)
  sensor_name = master_config['dataset']['sensor']['name']

  res_config = master_config['residual_images']
  for num_last_n_idx in tqdm(res_config['num_last_n'], total=len(res_config['num_last_n']), desc="num_last_n"):
    for seq_idx in trange(len(res_config['seqs']), desc="Sequence in Num_last_n", leave=False):
      # specify parameters
      seq_num = res_config['seqs'][seq_idx]
      num_frames = res_config['num_frames']
      debug = res_config['debug']
      normalize = res_config['normalize']
      num_last_n = res_config['num_last_n'][num_last_n_idx]
      visualize = res_config['visualize']
      visualization_folder = res_config['visualization_folder'].replace('SEQ_NUM', str(seq_num)).replace('SENSOR', sensor_name)+str(num_last_n)
      
      # specify the output folders
      residual_image_folder = res_config['residual_image_folder'].replace('SEQ_NUM', str(seq_num)).replace('SENSOR', sensor_name)+str(num_last_n)
      if not os.path.exists(residual_image_folder):
        os.makedirs(residual_image_folder)
        
      if visualize:
        if not os.path.exists(visualization_folder):
          os.makedirs(visualization_folder)
      
      # load poses
      root_path = master_config['dataset']['root_folder']
      pose_file = os.path.join(root_path,seq_num,sensor_name, master_config['dataset']['pose_file'])
      calib_file = os.path.join(root_path,seq_num,sensor_name, master_config['dataset']['calib_file'])
      scan_folder = os.path.join(root_path,seq_num,sensor_name, master_config['dataset']['scan_folder'])
      range_image_params = master_config['dataset']['sensor']
      generate_res_images(pose_file, calib_file, scan_folder, residual_image_folder, visualization_folder, range_image_params, num_last_n)
