
# Single-camera Tracking on multicam

CUDA_VISIBLE_DEVICES=2 python3 tools/multicam_track.py video \
./test-site022 \
./experiments/yolox/yolox_x_8xb8-300e_coco.py \
./experiments/yolox/pretrained-YOLOX-X.pth \
--tp_weight ./experiments/trajectory_weight/tp_best_12-11.pth \
--save_result ./test-site022 --save_vid True --track_buffer 150 
