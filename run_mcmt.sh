
# Multi-camera Tracking (Tracklet Association)

CUDA_VISIBLE_DEVICES=2 python3 tools/multicam_association.py \
./test-site022 \
./experiments/mcmt/homography_list.pkl \
./experiments/mcmt/yolov7-w6-pose.pt \
--save_txt_path ./test-site022
