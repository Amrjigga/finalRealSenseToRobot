# finalRealSenseToRobot

Checkpoint repo for RealSense D435i to Unitree G1 right palm/wrist orientation calibration in Isaac Lab.

Run command:

cd ~/IsaacLab_5
./isaaclab.sh -p scripts/calibrate_realsense_to_g1_right_palm_orientation_400.py \
  --device cuda:0 \
  --num_samples 400 \
  --rgb_width 640 \
  --rgb_height 480 \
  --fps 30 \
  --save_images \
  --use_imu_stabilization

Notes:
Use a mounted/rigid RealSense camera, not handheld. Save samples only when the hand visualization looks stable. IMU stabilization helps keep the hand visualization stable when the camera rotates slightly. Remaining bad frames are mostly depth/landmark palm flips.
