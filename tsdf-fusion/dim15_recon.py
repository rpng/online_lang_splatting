"""Fuse 1000 RGB-D images from the 7-scenes dataset into a TSDF voxel volume with 2cm resolution.
"""

import time

import cv2
import numpy as np
from skimage import measure
import fusion3


prefix = '/media/saimouli/RPNG_FLASH_4/datasets/Replica2/vmap/room_0_small/imap/00'
#color_prefix = '/home/choyingw/Documents/GaussianGripMapping/datasets/Replica/room0_test/HR/colorized_1'
#color_prefix = '/home/choyingw/Documents/1029_15dim/GaussianGripMapping/results/datasets_Replica/2024-10-30-17-54-47/psnr/before_opt/lang'

# Download the langslam results from Dropbox and put it here
color_prefix = '/media/saimouli/Data6T/Replica/omni_data_result/room_0_small/2025-03-24-06-23-28/psnr/before_opt/lang'

if __name__ == "__main__":
  # ======================================================================================================== #
  # (Optional) This is an example of how to compute the 3D bounds
  # in world coordinates of the convex hull of all camera view
  # frustums in the dataset
  # ======================================================================================================== #
  print("Estimating voxel volume bounds...")
  n_imgs = 200
  cam_extr = np.loadtxt(f"{prefix}/traj_w_c.txt", delimiter=' ').reshape(-1, 4, 4)
  print(cam_extr.shape)
  cam_intr = np.array([
    [600.0, 0, 599.5],
    [0, 600.0, 339.5],
    [0, 0, 1],
  ])
  vol_bnds = np.zeros((3,2))
  for i in range(n_imgs):
    # langslam results save without those following frames
    if i == 0 or i % 5 != 0:
      continue
    # Read depth image and camera pose
    depth_im = cv2.imread(f"{prefix}/depth/depth_{i}.png",-1).astype(float)
    # Using depth map from original imap Replica_v2 dataset (depth in mm so scalw down 1000x)
    depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
    cam_pose = cam_extr[i]  # 4x4 rigid transformation matrix

    # Compute camera view frustum and extend convex hull
    view_frust_pts = fusion3.get_view_frustum(depth_im, cam_intr, cam_pose)
    vol_bnds[:,0] = np.minimum(vol_bnds[:,0], np.amin(view_frust_pts, axis=1))
    vol_bnds[:,1] = np.maximum(vol_bnds[:,1], np.amax(view_frust_pts, axis=1))
  # ======================================================================================================== #

  # ======================================================================================================== #
  # Integrate
  # ======================================================================================================== #
  # Initialize voxel volume
  print("Initializing voxel volume...")
  tsdf_vol = fusion3.TSDFVolume(vol_bnds, voxel_size=0.02)

  # Loop through RGB-D images and fuse them together
  t0_elapse = time.time()
  for i in range(0, n_imgs):

    if i == 0 or i % 5 != 0:
      continue
    print("Fusing frame %d/%d"%(i+1, n_imgs))

    # Read RGB-D image and camera pose
    try:
      color_image = np.load(f"{color_prefix}/{i}.npy")
    except:
      print("keyframe, skipped!")
      continue
    depth_im = cv2.imread(f"{prefix}/depth/depth_{i}.png",-1).astype(float)
    depth_im /= 1000.
    cam_pose = cam_extr[i]

    # Integrate observation into voxel volume (assume color aligned with depth)
    #print("min: ", color_image.max())
    #print("max: ", color_image.min())
    tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)

  fps = n_imgs / (time.time() - t0_elapse)
  print("Average FPS: {:.2f}".format(fps))
  
  # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
  print("Saving mesh to mesh.ply...")
  verts, faces, norms, colors = tsdf_vol.get_mesh()
  # Original floatingp-point semantic_mesh
  fusion3.meshwrite(f"{color_prefix}/../semantic_mesh.ply", verts, faces, norms, colors)

  # Colorized mesh to visually see consistency
  colors = (colors + 1) * 127.5
  fusion3.meshwrite_color(f"{color_prefix}/../semantic_mesh_color.ply", verts, faces, norms, colors)

  # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
  print("Saving point cloud to pc.ply...")
  point_cloud = tsdf_vol.get_point_cloud()
  fusion3.pcwrite(f"{color_prefix}/../semantic_pc.ply", point_cloud)