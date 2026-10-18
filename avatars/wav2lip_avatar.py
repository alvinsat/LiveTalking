###############################################################################
#  Copyright (C) 2024 LiveTalking@lipku https://github.com/lipku/LiveTalking
#  email: lipku@foxmail.com
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  
#       http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###############################################################################
#
#  Wav2Lip 数字人 — 迁移自 lipreal.py + lipasr.py
#

import math
import torch
import numpy as np

import os
import time
import cv2
import glob
import pickle

import queue
from queue import Queue
from threading import Thread, Event
import torch.multiprocessing as mp

from avatars.audio_features.mel import MelASR
import asyncio
from av import AudioFrame, VideoFrame
from avatars.wav2lip.models import Wav2Lip
from avatars.base_avatar import BaseAvatar

from tqdm import tqdm
from utils.logger import logger
from utils.image import read_imgs, mirror_index
from utils.device import initialize_device
from utils.openvino import compile_torch_model
from registry import register

device = initialize_device()
logger.info('PyTorch backend device available: {}'.format(device))

def _load(checkpoint_path, openvino=False):
    if not openvino and device.type == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_model(path, opt=None):
    model = Wav2Lip()
    logger.info("Load checkpoint from: {}".format(path))
    use_openvino = opt is not None and opt.inference_backend == 'openvino'
    checkpoint = _load(path, openvino=use_openvino)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    if use_openvino:
        batch_size = opt.batch_size
        examples = (
            torch.zeros(batch_size, 1, 80, 16),
            torch.zeros(batch_size, 6, 256, 256),
        )
        return compile_torch_model(
            model, examples, opt.openvino_device, "wav2lip",
            source_paths=(path,),
            static_input_shapes={
                index: tuple(example.shape)
                for index, example in enumerate(examples)
            },
        )

    model = model.to(device)
    return model.eval()

def load_avatar(avatar_id, opt=None):
    avatar_path = f"./data/avatars/{avatar_id}"
    full_imgs_path = f"{avatar_path}/full_imgs" 
    face_imgs_path = f"{avatar_path}/face_imgs" 
    coords_path = f"{avatar_path}/coords.pkl"
    
    with open(coords_path, 'rb') as f:
        coord_list_cycle = pickle.load(f)
    frame_list_cycle = None
    input_img_list = glob.glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    frame_list_cycle = read_imgs(input_img_list)
    input_face_list = glob.glob(os.path.join(face_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    input_face_list = sorted(input_face_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    face_list_cycle = read_imgs(input_face_list)

    return frame_list_cycle,face_list_cycle,coord_list_cycle

@torch.no_grad()
def warm_up(batch_size,model,modelres):
    # 预热函数
    logger.info('warmup model...')
    inference_device = torch.device('cpu') if getattr(model, 'is_openvino', False) else device
    img_batch = torch.ones(batch_size, 6, modelres, modelres).to(inference_device)
    mel_batch = torch.ones(batch_size, 1, 80, 16).to(inference_device)
    model(mel_batch, img_batch)

@register("avatar", "wav2lip")
class LipReal(BaseAvatar):
    @torch.no_grad()
    def __init__(self, opt, model, avatar):
        super().__init__(opt)

        #self.fps = opt.fps # 20 ms per frame
        
        # self.batch_size = opt.batch_size
        # self.idx = 0
        # self.res_frame_queue = Queue(self.batch_size*2)
        self.model = model

        self.frame_list_cycle,self.face_list_cycle,self.coord_list_cycle = avatar

        self.asr = MelASR(opt,self)
        self.asr.warm_up()
    
    def inference_batch(self, index, audiofeat_batch):
        # 这里的 index 是针对当前 avatar 的索引
        # 返回一个 batch 的推理结果，batch 大小由 self.batch_size 决定
        length = len(self.face_list_cycle)
        img_batch = []
        for i in range(self.batch_size):
            idx = mirror_index(length, index + i)
            face = self.face_list_cycle[idx]
            img_batch.append(face)
        img_batch, audiofeat_batch = np.asarray(img_batch), np.asarray(audiofeat_batch)

        if getattr(self.model, 'is_openvino', False):
            batch, height, width, _ = img_batch.shape
            model_images = np.empty((batch, 6, height, width), dtype=np.float32)
            face_images = np.transpose(img_batch, (0, 3, 1, 2))
            model_images[:, :3] = face_images
            model_images[:, :3, height // 2:, :] = 0
            model_images[:, 3:] = face_images
            model_images /= 255.0
            model_audio = np.ascontiguousarray(
                audiofeat_batch[:, np.newaxis, :, :],
                dtype=np.float32,
            )
        else:
            img_masked = img_batch.copy()
            img_masked[:, face.shape[0] // 2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            audiofeat_batch = np.reshape(
                audiofeat_batch,
                [len(audiofeat_batch), audiofeat_batch.shape[1], audiofeat_batch.shape[2], 1],
            )
            model_images = torch.FloatTensor(
                np.transpose(img_batch, (0, 3, 1, 2))
            ).to(device)
            model_audio = torch.FloatTensor(
                np.transpose(audiofeat_batch, (0, 3, 1, 2))
            ).to(device)

        with torch.no_grad():
            pred = self.model(model_audio, model_images)
        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
        return pred

    def paste_back_frame(self,pred_frame,idx:int):
        bbox = self.coord_list_cycle[idx]
        combine_frame = self.frame_list_cycle[idx].copy()
        y1, y2, x1, x2 = bbox
        res_frame = cv2.resize(pred_frame.astype(np.uint8),(x2-x1,y2-y1))
        combine_frame[y1:y2, x1:x2] = res_frame
        return combine_frame
