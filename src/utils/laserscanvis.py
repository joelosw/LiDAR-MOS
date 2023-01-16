#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import vispy
from vispy.scene import visuals, SceneCanvas
import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
import time


class LaserScanVis:
  """Class that creates and handles a visualizer for a pointcloud"""

  def __init__(self, scan, scan_names, label_names, offset=0,
               semantics=True, gt_names = None):
    self.scan = scan
    self.scan_names = scan_names
    self.label_names = label_names
    self.offset = offset
    self.semantics = semantics
    self.gt_names = gt_names

    self.reset()
    self.update_scan()

  def reset(self):
    """ Reset. """
    # last key press (it should have a mutex, but visualization is not
    # safety critical, so let's do things wrong)
    self.action = "no"  # no, next, back, quit are the possibilities

    # new canvas prepared for visualizing data
    self.canvas = SceneCanvas(keys='interactive', show=True)
    # interface (n next, b back, q quit, very simple)
    self.canvas.events.key_press.connect(self.key_press)
    self.canvas.events.draw.connect(self.draw)
    # grid
    self.grid = self.canvas.central_widget.add_grid()

    # laserscan part
    self.scan_view = vispy.scene.widgets.ViewBox(
        border_color='white', parent=self.canvas.scene)
    self.grid.add_widget(self.scan_view, 0, 0)
    self.scan_vis = visuals.Markers()
    self.scan_view.camera = 'turntable'
    self.scan_view.add(self.scan_vis)
    visuals.XYZAxis(parent=self.scan_view.scene)
    # add semantics
    if self.semantics:
      print("Using semantics in visualizer")
      self.sem_view = vispy.scene.widgets.ViewBox(
          border_color='white', parent=self.canvas.scene)
      self.grid.add_widget(self.sem_view, 0, 1)
      self.sem_vis = visuals.Markers()
      self.sem_view.camera = 'turntable'
      self.sem_view.add(self.sem_vis)
      visuals.XYZAxis(parent=self.sem_view.scene)
      self.sem_view.camera.link(self.scan_view.camera)

    if self.gt_names:
      print("Using gt_names in visualizer")
      self.gt_view = vispy.scene.widgets.ViewBox(
          border_color='white', parent=self.canvas.scene)
      self.grid.add_widget(self.gt_view, 0, 1 + int(self.semantics))
      self.gt_vis = visuals.Markers()
      self.gt_view.camera = 'turntable'
      self.gt_view.add(self.gt_vis)
      visuals.XYZAxis(parent=self.gt_view.scene)
      self.gt_view.camera.link(self.scan_view.camera)

    # img canvas size
    self.multiplier = 1
    self.canvas_W = 2048
    self.canvas_H = 128
    if self.semantics:
      self.multiplier += 1
    if self.gt_names:
      self.multiplier += 1

    # new canvas for img
    self.img_canvas = SceneCanvas(keys='interactive', show=True,
                                  size=(self.canvas_W, self.canvas_H * self.multiplier))
    self.img_canvas.events.resize.connect(self.resize_images)
    # grid
    self.img_grid = self.img_canvas.central_widget.add_grid()
    # interface (n next, b back, q quit, very simple)
    self.img_canvas.events.key_press.connect(self.key_press)
    self.img_canvas.events.draw.connect(self.draw)

    # add a view for the depth
    self.img_view = vispy.scene.widgets.ViewBox(
        border_color='white', parent=self.img_canvas.scene)
    #self.img_view.on_resize(self.resize_images)
    self.img_grid.add_widget(self.img_view, 0, 0)
    self.img_vis = visuals.Image(cmap='viridis')
    self.img_view.add(self.img_vis)

    # add semantics
    if self.semantics:
      self.sem_img_view = vispy.scene.widgets.ViewBox(
          border_color='white', parent=self.img_canvas.scene)
      self.img_grid.add_widget(self.sem_img_view, 1, 0)
      self.sem_img_vis = visuals.Image(cmap='viridis')
      self.sem_img_view.add(self.sem_img_vis)

    # add gt_names
    if self.gt_names:
      self.gt_img_view = vispy.scene.widgets.ViewBox(
          border_color='white', parent=self.img_canvas.scene)
      self.img_grid.add_widget(self.gt_img_view, 1+self.semantics, 0)
      self.gt_img_vis = visuals.Image(cmap='viridis')
      self.gt_img_view.add(self.gt_img_vis)

  def get_mpl_colormap(self, cmap_name):
    cmap = plt.get_cmap(cmap_name)

    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)

    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]

    return color_range.reshape(256, 3).astype(np.float32) / 255.0

  def update_scan(self):
    # first open data
    self.scan.open_scan(self.scan_names[self.offset], None, None)
    if self.semantics:
      self.scan.open_label(self.label_names[self.offset])
      self.scan.colorize()
    if self.gt_names:
      self.scan.open_gt_label(self.gt_names[self.offset])
      self.scan.gt_colorize()


    # then change names
    title = "scan " + self.scan_names[self.offset].split(os.sep)[-1] + f" in [{self.scan_names[0].split(os.sep)[-1][:-4]} - {self.scan_names[-1].split(os.sep)[-1][:-4]}]"
    self.canvas.title = "3D: " + title
    self.img_canvas.title = "Range: " + title

    # then do all the point cloud stuff

    # plot scan
    power = 16
    # print()
    range_data = np.copy(self.scan.unproj_range)
    # print(range_data.max(), range_data.min())
    range_data = range_data**(1 / power)
    # print(range_data.max(), range_data.min())
    viridis_range = ((range_data - range_data.min()) /
                     (range_data.max() - range_data.min()) *
                     255).astype(np.uint8)
    viridis_map = self.get_mpl_colormap("viridis")
    viridis_colors = viridis_map[viridis_range]
    self.scan_vis.set_data(self.scan.points,
                           face_color=viridis_colors[..., ::-1],
                           edge_color=viridis_colors[..., ::-1],
                           size=1)

    # plot semantics
    if self.semantics:
      self.sem_vis.set_data(self.scan.points,
                            face_color=self.scan.sem_label_color[..., ::-1],
                            edge_color=self.scan.sem_label_color[..., ::-1],
                            size=1)

    # plot gt_names
    if self.gt_names:
      self.gt_vis.set_data(self.scan.points,
                             face_color=self.scan.gt_label_color[..., ::-1],
                             edge_color=self.scan.gt_label_color[..., ::-1],
                             size=1)
      self.gt_img_vis.set_data(self.scan.proj_gt_color[..., ::-1])
      self.gt_img_vis.update()

    self.resize_images(None)

  def resize_images(self, event):
    #print("recieved resizing event: %s" % event)
    size = self.img_view.size
    data = np.copy(self.scan.proj_range)
    # print(data[data > 0].max(), data[data > 0].min())
    data[data > 0] = data[data > 0]**(1 / 16)
    data[data < 0] = data[data > 0].min()
    # print(data.max(), data.min())
    data = (data - data[data > 0].min()) / \
        (data.max() - data[data > 0].min())
    
    data = cv2.resize(data, dsize=[int(s) for s in size], interpolation=cv2.INTER_CUBIC)
    
    self.img_vis.set_data(data)
    self.img_vis.update()
    if self.semantics:
      size = self.sem_img_view.size
      data_sem = cv2.resize(self.scan.proj_sem_color[..., ::-1], dsize=[int(s) for s in size], interpolation=cv2.INTER_CUBIC)
      self.sem_img_vis.set_data(data_sem)
      self.sem_img_vis.update()
    if self.gt_names:
      size = self.gt_img_view.size
      data_gt = cv2.resize(self.scan.proj_gt_color[..., ::-1], dsize=[int(s) for s in size], interpolation=cv2.INTER_CUBIC)
      self.gt_img_vis.set_data(data_gt)
      self.gt_img_vis.update()



  # interface
  def key_press(self, event):
    self.canvas.events.key_press.block()
    self.img_canvas.events.key_press.block()
    if event.key == 'N':
      self.offset += 1
      if self.offset >= len(self.scan_names):
        self.offset = 0
      self.update_scan()
    elif event.key == 'B':
      self.offset -= 1
      if self.offset <= 0:
        self.offset = len(self.scan_names)-1
      self.update_scan()
    elif event.key == 'R':
      self.img_canvas.size=(self.canvas_W, self.canvas_H * self.multiplier)
      self.update_scan()
      
    elif event.key == 'Q' or event.key == 'Escape':
      self.destroy()

  def draw(self, event):
    if self.canvas.events.key_press.blocked():
      self.canvas.events.key_press.unblock()
    if self.img_canvas.events.key_press.blocked():
      self.img_canvas.events.key_press.unblock()

  def destroy(self):
    # destroy the visualization
    self.canvas.close()
    self.img_canvas.close()
    vispy.app.quit()

  def run(self):
    vispy.app.run()
