from laserscan import LaserScan
from laserscanvis import LaserScanVis
import numpy as np
import matplotlib.pyplot as plt
scan = LaserScan(project=True,H=128, fov_up=11.25, fov_down=-11.25, drop_zero_points=True)
path = "/home/joe46973/Masterarbeit/HelloWorld/diff_images/sensor_frame_bins/scan_0006897.bin"
scan.open_scan(path)
plt.imshow(scan.proj_range)
plt.show()
#scan.unproj_range = np.linalg.norm(scan.points, 2, axis=1)
#scan.do_range_projection()
vis = LaserScanVis(scan, scan_names=[path], label_names=[None], semantics=False)
vis.run()