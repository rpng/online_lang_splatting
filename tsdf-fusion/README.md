Preparation: Download Replica_v2 from imap. The files use the original imap replica v2's depth with scale in mm. Other versions are fine just need to make sure what the depth scale is for your data. 

Note: in each .py file, please read the comment and change path variables that match your local.

1. For dim-3 reconstruction, download Replica_v2 from imap and langsplat/ langslam dim-3 results and run python3 dim3_recon.py (change the path variable in the file)

2. For dim-15 reconstruction, download Replica_v2 from imap and langslam dim-15 results and run python3 dim15_recon.py 

Evaluation

1. Prepare colorized GT by python3 running save_semantic_colors_gt.py

2. To reconstruct TSDF for groundtruth, run python3 dim3_recon_gt.py (change the path variable in the file)

3. cd PytorchEMD; python3 setup.py

4. copy the compiled .so file to the tsdf-fusion folder (move one level up)

5. Evaluation: python3 3d_evaluation_and_visualize_langslam_dim15.py (langslam) or python3 3d_evaluation_and_visualize_langsplat.py (langsplat)
