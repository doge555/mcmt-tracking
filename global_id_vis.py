import cv2
import numpy as np

def vis_global_id(video_path=str, id_info_path=str, output_video_path=str):
    color = [(0, 255, 0),(255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # print((int(width), int(height)))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (int(width), int(height)))
    bbox_info_list = []
    frame_num = 0
    
    with open(id_info_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            values = line.split(",")
            bbox_info_list.append(values)
    bbox_info_list = np.array(bbox_info_list, dtype=float)

    while True:
        ret, frame = cap.read()
        corr_bbox_info_list = bbox_info_list[bbox_info_list[:, 2]==frame_num]
        for corr_bbox_info in corr_bbox_info_list:
            cv2.rectangle(frame, (int(corr_bbox_info[3]), int(corr_bbox_info[4])), (int(corr_bbox_info[3]+corr_bbox_info[5]), int(corr_bbox_info[4]+corr_bbox_info[6])), color[int(corr_bbox_info[1]) % 5], 2)
            cv2.putText(frame, str(corr_bbox_info[1]), (int(corr_bbox_info[3]+corr_bbox_info[5]/2), int(corr_bbox_info[4]+corr_bbox_info[6]/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color[int(corr_bbox_info[1]) % 5], 2, cv2.LINE_AA)
        if not ret:
            break
        video_writer.write(frame)
        frame_num += 1
    
    video_writer.release()
    return 1

if __name__ == "__main__":
    result = vis_global_id(video_path="test-site022/c124/video.mp4", id_info_path="test-site022/c124/sorted_associated.txt", output_video_path="test-site022/c124/gobal_video.mp4")
    result = vis_global_id(video_path="test-site022/c125/video.mp4", id_info_path="test-site022/c125/sorted_associated.txt", output_video_path="test-site022/c125/gobal_video.mp4")
    result = vis_global_id(video_path="test-site022/c126/video.mp4", id_info_path="test-site022/c126/sorted_associated.txt", output_video_path="test-site022/c126/gobal_video.mp4")
    result = vis_global_id(video_path="test-site022/c127/video.mp4", id_info_path="test-site022/c127/sorted_associated.txt", output_video_path="test-site022/c127/gobal_video.mp4")
    result = vis_global_id(video_path="test-site022/c128/video.mp4", id_info_path="test-site022/c128/sorted_associated.txt", output_video_path="test-site022/c128/gobal_video.mp4")
    result = vis_global_id(video_path="test-site022/c129/video.mp4", id_info_path="test-site022/c129/sorted_associated.txt", output_video_path="test-site022/c129/gobal_video.mp4")
    if result == 1:
        print("finish generating!!!")        
    