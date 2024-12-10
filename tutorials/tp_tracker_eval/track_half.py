import os
import logging
import argparse
import motmetrics as mm
import torch
import numpy as np
from yolox.tracker.tp_tracker import TPTracker
from log import logger
from timer import Timer
from evaluation import Evaluator
from tools.model import SocialImplicit
from tools.CFG import CFG


def mkdir_if_missing(dir):
    os.makedirs(dir, exist_ok=True)

def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def eval_seq(opt, det_list, data_type, result_filename, frame_rate=30):
    '''
       Processes the video sequence given and provides the output of tracking result (write the results in video file)

       It uses JDE model for getting information about the online targets present.

       Parameters
       ----------
       opt : Namespace
             Contains information passed as commandline arguments.

       dataloader : LoadVideo
                    Instance of LoadVideo class used for fetching the image sequence and associated data.

       data_type : String
                   Type of dataset corresponding(similar) to the given video.

       result_filename : String
                         The name(path) of the file for storing results.
       frame_rate : int
                    Frame-rate of the given video.

       Returns
       -------
       (Returns are not significant here)
       frame_id : int
                  Sequence number of the last sequence
       '''
    tp_model = SocialImplicit(spatial_input=CFG["spatial_input"],
                              spatial_output=CFG["spatial_output"],
                              temporal_input=CFG["temporal_input"],
                              temporal_output=CFG["temporal_output"],
                              bins=CFG["bins"],
                              noise_weight=CFG["noise_weight"]).cuda()
    tp_model.load_state_dict(torch.load(opt.weights))
    tp_model.cuda().double()
    tp_model.eval()
    tracker = TPTracker(tp_model, opt, frame_rate=frame_rate)
    timer = Timer()
    results = []
    frame_id = 0
    for _, det_array in enumerate(det_list):
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1./max(1e-5, timer.average_time)))

        # run tracking
        timer.tic()
        online_targets = tracker.update(det_array, None)
        online_tlwhs = []
        online_ids = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            if tlwh[2] * tlwh[3] > opt.min_box_area:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
        timer.toc()
        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))
        frame_id += 1
    # save results
    write_results(result_filename, results, data_type)
    return frame_id, timer.average_time, timer.calls

def read_det_file(det_file):
    with open(det_file, 'r') as file:
        lines = file.readlines()

    data_dict = {}

    for line in lines:
        values = list(map(float, line.strip().split(',')))
        key = int(values[0])
        det_info = [values[2], values[3], values[2]+values[4], values[3]+values[5],  values[6]]
        if key in data_dict:
            data_dict[key].append(det_info)
        else:
            data_dict[key] = [det_info]
        result_list = []
    
    for key in data_dict:
        result_list.append(np.array(data_dict[key]))
    return result_list

def main(opt, data_root='/MOT16/train', seqs=['MOT16-02'], exp_name='demo'):
    logger.setLevel(logging.INFO)
    result_root = 'results'
    print(result_root)
    mkdir_if_missing(result_root)
    data_type = 'mot'

    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    for seq in seqs:

        logger.info('start seq: {}'.format(seq))
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        det_filename = os.path.join(data_root, seq, 'det', 'det.txt')
        det_list = read_det_file(det_filename)
        nf, ta, tc = eval_seq(opt, det_list, data_type, result_filename, frame_rate=30)
        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)

        # eval
        logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(data_root, seq, data_type)
        accs.append(evaluator.eval_file(result_filename))

    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    Evaluator.save_summary(summary, os.path.join(result_root, 'summary_{}.xlsx'.format(exp_name)))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='track.py')
    parser.add_argument('--weights', type=str, default='weight/tp_best_11-26.pth', help='path to weights file')
    parser.add_argument('--track-buffer', type=int, default=30, help='tracking buffer')
    parser.add_argument("--track_thresh", type=float, default=0.8, help="tracking confidence threshold")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    opt = parser.parse_args()
    print(opt, end='\n\n')
 
    seqs_str = '''MOT16-02
                    MOT16-04
                    MOT16-05
                    MOT16-09
                    MOT16-10
                    MOT16-11
                    MOT16-13'''
    data_root = 'MOT16/train'
    seqs = [seq.strip() for seq in seqs_str.split()]

    main(opt,
         data_root=data_root,
         seqs=seqs,
         exp_name=opt.weights.split('/')[-1]
        )
