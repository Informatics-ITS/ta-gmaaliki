# This script produces tracklets given tracking results and original sequence frame as RGB images.
import argparse
from torchreid.utils import FeatureExtractor

import os
from tqdm import tqdm
from loguru import logger
from PIL import Image

import pickle
import numpy as np
import glob

import torch
import torchvision.transforms as T
from .Tracklet import Tracklet

def generate_tracklets(
    model_path: str, # Specify the path to reid model's checkpoint file (default is ../reid_checkpoints/sports_model.pth.tar-60)
    data_path: str, # Specify directory of the clip's dataset (e.g. SoccerNet/tracking-2023/test/SNMOT-116).
    pred_file: str, # Specify the tracker file for a clip
    output_dir: str, # Specify the output for the .pkl file
):
    # load feature extractor:
    val_transforms = T.Compose([
            T.Resize([256, 128]),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor = FeatureExtractor(
        model_name='osnet_x1_0',
        model_path = model_path,
        device=device
    )

    os.makedirs(output_dir, exist_ok=True)

    imgs = sorted(glob.glob(os.path.join(data_path, 'img1', '*')))   # assuming data is organized in MOT convention
    track_res = np.genfromtxt(pred_file,dtype=float, delimiter=',')

    last_frame = int(track_res[-1][0])
    seq_tracks = {}

    for frame_id in range(1, last_frame+1):
        if frame_id%100 == 0:
            logger.info(f'Processing frame {frame_id}/{last_frame}')

        # query all track_res for current frame
        inds = track_res[:,0] == frame_id
        frame_res = track_res[inds]
        img = Image.open(imgs[int(frame_id)-1])
        
        input_batch = None    # input batch to speed up processing
        tid2idx = {}
        
        # NOTE MOT annotation format:
        # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        for idx, (frame, track_id, l, t, w, h, score, obj_class, _, _) in enumerate(frame_res):
            # Update tracklet with detection
            bbox = [l, t, w, h]
            if track_id not in seq_tracks:
                seq_tracks[track_id] = Tracklet(track_id, frame, score, bbox, obj_class=obj_class)
            else:
                seq_tracks[track_id].append_det(frame, score, bbox, obj_class=obj_class)
            tid2idx[track_id] = idx

            im = img.crop((l, t, l+w, t+h)).convert('RGB')
            im = val_transforms(im).unsqueeze(0)
            if input_batch is None:
                    input_batch = im
            else:
                input_batch = torch.cat([input_batch, im], dim=0)
        
        if input_batch is not None:
            features = extractor(input_batch)    # len(features) == len(frame_res)
            feats = features.cpu().detach().numpy()
            
            # update tracklets with feature
            for tid, idx in tid2idx.items():
                feat = feats[tid2idx[tid]]
                feat /= np.linalg.norm(feat)
                seq_tracks[tid].append_feat(feat)
        else:
            print(f"No detection at frame: {frame_id}")
    
    # save seq_tracks into pickle file
    seq = os.path.splitext(os.path.basename(pred_file))[0]
    track_output_path = os.path.join(output_dir,  f'{seq}.pkl')
    with open(track_output_path, 'wb') as f:
        pickle.dump(seq_tracks, f)
    logger.info(f"save tracklets info to {track_output_path}")
