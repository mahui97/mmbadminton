from genericpath import exists
import os
import json
from tkinter import image_names
import cv2
import mmcv
import numpy as np
import ntpath
from mmskeleton.apis.estimation import init_pose_estimator, inference_pose_estimator
from mmskeleton.utils import call_obj
from mmskeleton.datasets import skeleton
from multiprocessing import current_process, Process, Manager
from mmskeleton.utils import cache_checkpoint
from mmskeleton.court import court_model
from mmcv.utils import ProgressBar

pose_estimators = dict()


def worker(inputs, results, gpu, detection_cfg, estimation_cfg):
    worker_id = current_process()._identity[0] - 1
    global pose_estimators
    if worker_id not in pose_estimators:
        pose_estimators[worker_id] = init_pose_estimator(
            detection_cfg, estimation_cfg, device=gpu)
    while True:
        idx, image = inputs.get()

        # end signal
        if image is None:
            return

        res = inference_pose_estimator(pose_estimators[worker_id], image)
        res['frame_index'] = idx
        results.put(res)

def get_all_files(path):
    allfile = []
    for dirpath, dirnames, filenames in os.walk(path):
        for dir in dirnames:
            allfile.append(os.path.join(dirpath, dir))
        for name in filenames:
            allfile.append(os.path.join(dirpath, name))
    allfile = list(filter(lambda x: x.find(".mp4") >= 0, allfile))
    return allfile

def build_court(detection_cfg,
          estimation_cfg,
          tracker_cfg,
          image_dir,
          out_dir,
          gpus=1,
          worker_per_gpu=1,
          video_max_length=10000,
          category_annotation=None):

    # cache_checkpoint(detection_cfg.checkpoint_file)
    # cache_checkpoint(estimation_cfg.checkpoint_file)
    if tracker_cfg is not None:
        raise NotImplementedError

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    inputs = Manager().Queue(video_max_length)
    results = Manager().Queue(video_max_length)

    num_worker = gpus * worker_per_gpu
    procs = []
    for i in range(num_worker): 
        p = Process(
            target=worker,
            args=(inputs, results, i % gpus, detection_cfg, estimation_cfg))
        procs.append(p)
        p.start()
    
    image_file_list = get_all_files(image_dir)
    prog_bar = ProgressBar(len(image_file_list))
    for image_file in image_file_list:
        image_name = image_file.split('/')[-1].split('.')[0]
        if image_name != 'ff_b_01':  
            continue
        reader = cv2.imread(image_file) 
        frame_court_model = court_model(reader, image_name)
        prog_bar.update()

    # send end signals
    for p in procs:
        inputs.put((-1, None))
    # wait to finish
    for p in procs:
        p.join()

    print('\nBuild court dataset to {}.'.format(out_dir))
    return

def build(detection_cfg,
          estimation_cfg,
          tracker_cfg,
          video_dir,
          out_dir,
          gpus=1,
          worker_per_gpu=1,
          video_max_length=10000,
          category_annotation=None):

    cache_checkpoint(detection_cfg.checkpoint_file)
    cache_checkpoint(estimation_cfg.checkpoint_file)
    if tracker_cfg is not None:
        raise NotImplementedError

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    if category_annotation is None:
        video_categories = dict()
    else:
        with open(category_annotation) as f:
            video_categories = json.load(f)['annotations']

    inputs = Manager().Queue(video_max_length)
    results = Manager().Queue(video_max_length)

    num_worker = gpus * worker_per_gpu
    procs = []
    for i in range(num_worker):
        p = Process(
            target=worker,
            args=(inputs, results, i % gpus, detection_cfg, estimation_cfg))
        procs.append(p)
        p.start()
    
    video_file_list = get_all_files(video_dir)
    prog_bar = ProgressBar(len(video_file_list))
    for video_file in video_file_list:
        video_name = video_file.split('/')[-1].split('.')[0]
        # if video_name != 'fc_a_02':
        #     continue
        action = video_name.split('_')[0]
        category_id = video_categories[action][
            'category_id'] if action in video_categories else -1
        if category_id == -1:
            continue
        # reader = mmcv.VideoReader(os.path.join(video_dir, video_file))
        reader = mmcv.VideoReader(video_file)
        
        video_frames = reader[:video_max_length]
        annotations = []
        num_keypoints = -1
        invalid_frames = []
        frame_court_model = court_model(video_frames[0], video_name)
        for i, image in enumerate(video_frames):
            inputs.put((i, image))

        for i in range(len(video_frames)):
            t = results.get()
            if not t['has_return']:
                continue

            num_person = len(t['joint_preds'])
            assert len(t['person_bbox']) == num_person
            
            # in_court[j]=0: this person is not a player
            # in_court[j]=1: a player we need
            in_court = np.zeros((num_person))
            ankles = np.ones((2, 2))
            if num_person == 1:
                in_court[0] = 1
                valid_player, ankles = frame_court_model.in_court(t['joint_preds'][0][15:17, :])
            else:
                bbox_size = 2147483648
                last_person_id = -1
                
                for j in range(num_person):
                    valid_player = frame_court_model.in_half_countor(t['person_bbox'][j])

                    # valid_player, std_ankle = frame_court_model.in_court(t['joint_preds'][j][15:17, :])

                    in_court[j] = 1 if valid_player == True else 0
                    if in_court[j] == 1:
                        bspace = (t['person_bbox'][j][2] - t['person_bbox'][j][0]) * (t['person_bbox'][j][3] - t['person_bbox'][j][1])
                        if bbox_size <= bspace:
                            in_court[j] = 0
                        else:
                            bbox_size = bspace
                            # ankles = std_ankle
                            if last_person_id > -1:
                                in_court[last_person_id] = 0
                            last_person_id = j
                
            if np.sum(in_court) != 1:
                invalid_frames.append(i)
                continue
            j_value = np.argwhere(in_court == 1).reshape(-1).tolist()
            for j in j_value:
            # for j in range(num_person):
                keypoints = [[p[0], p[1], round(s[0], 2)] for p, s in zip(
                    t['joint_preds'][j].round().astype(int).tolist(), t[
                        'joint_scores'][j].tolist())]
                num_keypoints = len(keypoints)
                # location = ankles.mean(axis=0).tolist()
                
                person_info = dict(
                    person_bbox=t['person_bbox'][j].round().astype(int)
                    .tolist(),
                    frame_index=t['frame_index'],
                    id=0,
                    person_id=None,
                    # location=location, # 坐标
                    keypoints=keypoints)
                annotations.append(person_info)

        # output results
        annotations = sorted(annotations, key=lambda x: x['frame_index'])
        info = dict(
            video_name=video_name,
            resolution=reader.resolution,
            num_frame=len(video_frames),
            num_keypoints=num_keypoints,
            keypoint_channels=['x', 'y', 'score'],
            version='1.0')
        video_info = dict(
            info=info, category_id=category_id, annotations=annotations)
        with open(os.path.join(out_dir, video_name + '.json'), 'w') as f:
            json.dump(video_info, f)
        if invalid_frames != []:
            invalid_str = video_name + ' ' + str(invalid_frames) + '\n'
            with open('invalid_frames.txt', 'a') as f:
                f.write(invalid_str)
        prog_bar.update()

    # send end signals
    for p in procs:
        inputs.put((-1, None))
    # wait to finish
    for p in procs:
        p.join()

    print('\nBuild skeleton dataset to {}.'.format(out_dir))
    return video_info

def build_matchai(detection_cfg,
          estimation_cfg,
          tracker_cfg,
          video_dir,
          label_dir,
          out_dir,
          gpus=1,
          worker_per_gpu=1,
          video_max_length=3600,
          fps=30,
          category_annotation=None):

    cache_checkpoint(detection_cfg.checkpoint_file)
    cache_checkpoint(estimation_cfg.checkpoint_file)
    if tracker_cfg is not None:
        raise NotImplementedError

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    if category_annotation is None:
        video_categories = dict()
    else:
        with open(category_annotation) as f:
            video_categories = json.load(f)['annotations']
    
    inputs = Manager().Queue(video_max_length)
    results = Manager().Queue(video_max_length)

    num_worker = gpus * worker_per_gpu
    procs = []
    for i in range(num_worker):
        p = Process(
            target=worker,
            args=(inputs, results, i % gpus, detection_cfg, estimation_cfg))
        procs.append(p)
        p.start()
    
    video_file_list = get_all_files(video_dir)
    prog_bar = ProgressBar(2000)
    for video_file in video_file_list:
        video_name = video_file.split('/')[-1].split('.')[0] # example: 7_i
        index = video_name.split('_')[1]
        if int(index) < 83 or int(index) > 107 or index == '91':
            continue # 7_83 ~ 7_90
        # if index in ['51', '64', '80', '81', '82', '91']:
        #     continue
        subdir = os.path.join(out_dir, video_name)
        if not os.path.isdir(subdir):
            os.makedirs(subdir)
        
        reader = mmcv.VideoReader(video_file)
        with open(os.path.join(label_dir, 'final_'+index+'.json'), 'r') as f:
            shot_json = json.load(f)
        if isinstance(shot_json, dict):
            for key in shot_json:
                sub_video_json = shot_json[key]
                # sub_video = reader[sub_video_json['start_time']*fps:sub_video_json['end_time']*fps]
                # each shot_key is a video
                for shot_key in sub_video_json['shots']:
                    shot = sub_video_json['shots'][shot_key]
                    
                    category_id = video_categories[shot['Major_shot']]['category_id'] if shot['Major_shot'] in video_categories else -1
                    if category_id == -1:
                        continue

                    start_frame = int(shot['start_frame_shot'] * fps)
                    end_frame = int(shot['end_frame_shot'] * fps + 1)
                    video_frames = reader[start_frame:end_frame]
                    
                    annotations = []
                    num_keypoints = -1
                    
                    
                    for i, image in enumerate(video_frames):
                        inputs.put((i, image))

                    for i in range(len(video_frames)):
                        t = results.get()
                        if not t['has_return']:
                            continue

                        num_person = len(t['joint_preds'])
                        assert len(t['person_bbox']) == num_person
                        
                        # find proper person of this action
                        valid_index = -1
                        for j in range(num_person):
                            person_bbox=t['person_bbox'][j].round().astype(int)
                            person_center = person_bbox[:4].reshape((2,2)).mean(axis=0)
                            if person_center[0] < 176 or person_center[0] > 625:
                                continue
                            if shot['player_played'] == 'bottom' and person_center[1] > 400:
                                valid_index = j
                            if shot['player_played'] == 'top' and person_center[1] > 165 and person_center[1] < 401:
                                valid_index = j
                        if valid_index == -1:
                            continue
                        
                        j = valid_index
                        keypoints = [[p[0], p[1], round(s[0], 2)] for p, s in zip(
                            t['joint_preds'][j].round().astype(int).tolist(), t[
                                'joint_scores'][j].tolist())]
                        num_keypoints = len(keypoints)
                        
                        person_info = dict(
                            person_bbox=t['person_bbox'][j].round().astype(int)
                            .tolist(),
                            frame_index=t['frame_index'],
                            id=0,
                            person_id=None,
                            keypoints=keypoints)
                        annotations.append(person_info)

                    # output results
                    annotations = sorted(annotations, key=lambda x: x['frame_index'])
                    
                    v_name = video_name + '_' + key + '_' + shot_key
                    info = dict(
                        video_name=v_name,
                        resolution=reader.resolution,
                        num_frame=len(video_frames),
                        num_keypoints=num_keypoints,
                        keypoint_channels=['x', 'y', 'score'],
                        version='1.0')
                    video_info = dict(
                        info=info, category_id=category_id, annotations=annotations)
                    with open(os.path.join(subdir, v_name + '.json'), 'w') as f:
                        json.dump(video_info, f)
                prog_bar.update()

    # send end signals
    for p in procs:
        inputs.put((-1, None))
    # wait to finish
    for p in procs:
        p.join()

    print('\nBuild skeleton dataset to {}.'.format(out_dir))
    return video_info

import torch
import ffmpy

def cut_a_video(ffmpeg_args, fps=30):
    
    start_frame = int(ffmpeg_args['shot']['start_frame_shot'] * fps)
    end_frame = int(ffmpeg_args['shot']['end_frame_shot'] * fps + 1)
    
    if not os.path.isdir(ffmpeg_args['out_dir']):
        os.makedirs(ffmpeg_args['out_dir'])
    # example: 1_top_SMASH.mp4
    cmd = 'ffmpeg -i ' + ffmpeg_args['full_video_name'] + ' -vf "select=between(n\\,' + str(start_frame) + '\\,' + str(end_frame) + ')" -y -acodec copy '\
         + os.path.join(ffmpeg_args['out_dir'], ffmpeg_args['out_file_name'])
    os.system(cmd)

def worker_cut_matchai(inputs, gpu):
    while True:
        args = inputs.get()

        # end signal
        if args is None:
            return

        cut_a_video(args, fps=60)

def cut_matchai(detection_cfg,
          estimation_cfg,
          tracker_cfg,
          video_dir,
          label_dir,
          out_dir,
          gpus=1,
          worker_per_gpu=1,
          video_max_length=10000,
          fps=30):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    inputs = Manager().Queue(video_max_length)

    num_worker = gpus * worker_per_gpu
    procs = []
    for i in range(num_worker):
        p = Process(
            target=worker_cut_matchai,
            args=(inputs, i % gpus))
        procs.append(p)
        p.start()
    
    video_file_list = os.listdir(video_dir)
    prog_bar = ProgressBar(len(video_file_list))
    
    for video_file in video_file_list:
        index = video_file.split('.')[0].split('_')[1]
        # 7_76 have not finished completely
        if index in ['6', '8', '10', '14', '24', '47', '50', '65', '105', '51', '64', '80', '81', '82', '91']:
            continue
        with open(os.path.join(label_dir, 'final_'+index+'.json'), 'r') as f:
            shot_json = json.load(f)
        if isinstance(shot_json, dict):
            for key in shot_json:
                sub_video_json = shot_json[key]
                for shot_key in sub_video_json['shots']:
                    shot = sub_video_json['shots'][shot_key]
                    args = {
                        "full_video_name": video_dir + '/' + video_file,
                        "shot": shot,
                        "out_dir": os.path.join(out_dir, '7_' + index),
                        "out_file_name": index + '_' + shot['player_played'] + '_' + shot['Major_shot'] + '_' + key + '_' + shot_key + '.mp4'
                    }
                    inputs.put(args)
                    
        prog_bar.update()

    # send end signals
    for p in procs:
        inputs.put((-1, None))
    # wait to finish
    for p in procs:
        p.join()

    print('\nBuild matchai shot videos to {}.'.format(out_dir))
    return        


def f(data):
    fmap = data['data'] * mask
    for _ in range(fmap.ndim - 1):
        fmap = fmap.sum(1)
    fmap = fmap / np.sum(mask)
    return fmap


def dataset_analysis(dataset_cfg, mask_channel=2, workers=16, batch_size=16):
    dataset = call_obj(**dataset_cfg)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers)

    prog_bar = ProgressBar(len(dataset))
    for k, (data, mask) in enumerate(data_loader):
        assert mask.size(1) == 1
        n = data.size(0)
        c = data.size(1)
        if k == 0:
            means = [[] for i in range(c)]
            stds = [[] for i in range(c)]
        mask = mask.expand(data.size()).type_as(data)
        data = data * mask
        sum = data.reshape(n * c, -1).sum(1)
        num = mask.reshape(n * c, -1).sum(1)
        mean = sum / num
        diff = (data.reshape(n * c, -1) - mean.view(n * c, 1)) * mask.view(
            n * c, -1)
        std = ((diff**2).sum(1) / num)**0.5
        mean = mean.view(n, c)
        std = std.view(n, c)
        for i in range(c):
            m = mean[:, i]
            m = m[~torch.isnan(m)]
            if len(m) > 0:
                means[i].append(m.mean())
            s = std[:, i]
            s = s[~torch.isnan(s)]
            if len(s) > 0:
                stds[i].append(s.mean())
        for i in range(n):
            prog_bar.update()
    means = [np.mean(m) for m in means]
    stds = [np.mean(s) for s in stds]
    print('\n\nDataset analysis result:')
    print('\tmean of channels : {}'.format(means))
    print('\tstd of channels  : {}'.format(stds))