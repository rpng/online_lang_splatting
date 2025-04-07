"""Fuse 1000 RGB-D images from the 7-scenes dataset into a TSDF voxel volume with 2cm resolution.
"""

import time

import cv2
import numpy as np

import fusion


prefix = '/media/saimouli/RPNG_FLASH_4/datasets/Replica2/vmap/room_0/imap/00'
color_prefix = '/media/saimouli/RPNG_FLASH_4/datasets/Replica2/vmap/room_0/imap/00/semantic_color'
save_path = '/media/saimouli/Data6T/Replica/omni_data_result/room_0_small/2025-03-24-06-23-28/psnr/before_opt'

if __name__ == "__main__":
  # ======================================================================================================== #
  # (Optional) This is an example of how to compute the 3D bounds
  # in world coordinates of the convex hull of all camera view
  # frustums in the dataset
  # ======================================================================================================== #
  print("Estimating voxel volume bounds...")
  n_imgs = 10 #2000
  cam_extr = np.loadtxt(f"{prefix}/traj_w_c.txt", delimiter=' ').reshape(-1, 4, 4)
  cam_intr = np.array([
    [600.0, 0, 599.5],
    [0, 600.0, 339.5],
    [0, 0, 1],
  ])
  vol_bnds = np.zeros((3,2))
  for i in range(n_imgs):
    # Read depth image and camera pose
    depth_im = cv2.imread(f"{prefix}/depth/depth_{i}.png",-1).astype(float)
    depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
    cam_pose = cam_extr[i]  # 4x4 rigid transformation matrix

    # Compute camera view frustum and extend convex hull
    view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr, cam_pose)
    vol_bnds[:,0] = np.minimum(vol_bnds[:,0], np.amin(view_frust_pts, axis=1))
    vol_bnds[:,1] = np.maximum(vol_bnds[:,1], np.amax(view_frust_pts, axis=1))
  # ======================================================================================================== #

  # ======================================================================================================== #
  # Integrate
  # ======================================================================================================== #
  # Initialize voxel volume
  print("Initializing voxel volume...")
  tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.02)

  # Loop through RGB-D images and fuse them together
  t0_elapse = time.time()
  for i in range(0, n_imgs):
    print("Fusing frame %d/%d"%(i+1, n_imgs))

    # Read RGB-D image and camera pose
    color_image = cv2.cvtColor(cv2.imread(f"{color_prefix}/semantic_class_{i}.png"), cv2.COLOR_BGR2RGB)
    depth_im = cv2.imread(f"{prefix}/depth/depth_{i}.png",-1).astype(float)
    depth_im /= 1000.
    cam_pose = cam_extr[i]

    # Integrate observation into voxel volume (assume color aligned with depth)
    tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)

  fps = n_imgs / (time.time() - t0_elapse)
  print("Average FPS: {:.2f}".format(fps))

  # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
  print("Saving mesh to mesh.ply...")
  verts, faces, norms, colors = tsdf_vol.get_mesh()
  fusion.meshwrite(f"{save_path}/GT_semantic_mesh.ply", verts, faces, norms, colors)

  # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
  print("Saving point cloud to pc.ply...")
  point_cloud = tsdf_vol.get_point_cloud()
  fusion.pcwrite(f"{save_path}/GT_semantic_pc.ply", point_cloud)