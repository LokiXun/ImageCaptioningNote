# encoding: utf-8
"""
Function: mPLUG 模型推理
@author: LokiXun
@contact: 2682414501@qq.com
"""
import argparse
import os
import yaml
import language_evaluation
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any, Union

import torch
from torchvision import transforms
import torch.nn as nn
from torch.cuda.amp import autocast
import torch.backends.cudnn as cudnn
from PIL import Image
import cv2
import numpy as np

from utils_module.logging_utils_local import get_logger
from dataset import get_transform
from models.model_caption_mplug import MPLUG
from models.vit import interpolate_pos_embed, resize_pos_embed
from models.tokenization_bert import BertTokenizer
import utils
from dataset.utils import save_result
from dataset import create_dataset, create_sampler, create_loader, coco_collate_fn
from scheduler import create_scheduler
from optim import create_optimizer, create_two_optimizer
from utils_module.const import DataPreprocessError, ModelInferenceError

logger = get_logger(name="mPLUG_imageCaptioning_inference")

base_path = Path(__file__).resolve().parent
model_config_filepath = base_path.joinpath("configs/caption_mplug_base.yaml").as_posix()
assert Path(model_config_filepath).exists(), f"model yaml file not exists! {model_config_filepath}"

with open(model_config_filepath, "r", encoding="u8") as fp:
    config = yaml.load(fp, Loader=yaml.SafeLoader)
    config['text_encoder'] = config['bert_base_uncased_dir']
    config['text_decoder'] = config['bert_base_uncased_dir']
    config["beam_size"] = config['beam_size']
    print(f"============= model config \n {config}")


class DataPreprocess:
    def __init__(self, cfg):
        """图像预处理"""
        self.config = cfg
        _, _, self.test_transform = get_transform(config=self.config)

    @staticmethod
    def process_single_image(torch_transform: transforms.Compose, image_path: str) \
            -> Tuple[bool, Union[torch.Tensor, str]]:
        """load image & transform the image"""
        # load image to ndarray
        try:
            assert Path(image_path).is_file(), f"file not exists"
            image: Image.Image = Image.open(image_path)
            image = image.convert("RGB")
            logger.info(f"image_path={image_path}, mode={image.mode}")

            # process
            image: torch.Tensor = torch_transform(image)
            return True, image
        except Exception as e:
            error_message = f"process_single_image error={e}, image_path={image_path}"
            logger.exception(error_message)
            return False, error_message

    def run(self, images_path_list: List[str]) -> Tuple[bool, List[Tuple[str, Union[str, torch.Tensor]]]]:
        """
        process images
        Args:
            images_path_list: Image filepath

        Returns:

        """
        error_image_list: List[Tuple[str, str]] = []
        result_image_list: List[Tuple[str, str]] = []
        try:
            assert images_path_list, f"no inputs image to process!images_path_list={images_path_list}"

            for single_image_path in images_path_list:
                success_flag, image_result = self.process_single_image(torch_transform=self.test_transform,
                                                                       image_path=single_image_path)
                if not success_flag:
                    error_image_list.append((single_image_path, image_result))
                    continue

                result_image_list.append((single_image_path, image_result))

            if not result_image_list:
                raise DataPreprocessError(f"all image failed")

            return True, result_image_list
        except Exception as e:
            error_message = f"images_path_list={images_path_list} process error!"
            logger.exception(error_message)
            return False, error_image_list


class ModelInference:
    def __init__(self, cfg: dict):
        self.config = cfg
        self.device = "cpu"  # "cuda" if torch.cuda.is_available() else "cpu"

        # load model
        self.text_encoder = 'bert-base-uncased'
        print(f"self.text_encoder={self.config['text_encoder']}")
        self.tokenizer = BertTokenizer.from_pretrained(self.text_encoder)

        self.model_checkpoints_path: str = cfg['pretrained_model_checkpoints']
        assert Path(
            self.model_checkpoints_path).is_file(), f"model_checkpoints_path not exists!{self.model_checkpoints_path}"
        self.caption_model = self._load_model()
        self.caption_model.eval()

    def _load_model(self):
        model = MPLUG(config=self.config, tokenizer=self.tokenizer)
        model = model.to(self.device)
        logger.info(f"initialize model success!")

        assert Path(self.model_checkpoints_path).is_file(), \
            f"model_checkpoints_path not exists!{self.model_checkpoints_path}"
        checkpoint = torch.load(self.model_checkpoints_path, map_location='cpu')
        try:
            state_dict = checkpoint['model']
        except Exception as e:
            print(f"load checkpoints error={e}")
            state_dict = checkpoint['module']

        # reshape positional embedding to accomodate for image resolution change
        if self.config["clip_name"] == "ViT-B-16":
            num_patches = int(self.config["image_res"] * self.config["image_res"] / (16 * 16))
        elif self.config["clip_name"] == "ViT-L-14":
            num_patches = int(self.config["image_res"] * self.config["image_res"] / (14 * 14))
        pos_embed = nn.Parameter(torch.zeros(num_patches + 1, 768).float())

        pos_embed = resize_pos_embed(state_dict['visual_encoder.visual.positional_embedding'].unsqueeze(0),
                                     pos_embed.unsqueeze(0))
        state_dict['visual_encoder.visual.positional_embedding'] = pos_embed

        # not evaluate
        for key in list(state_dict.keys()):
            if ('fusion' in key or 'bert' in key) and 'decode' not in key:
                encoder_key = key.replace('fusion.', '').replace('bert.', '')
                state_dict[encoder_key] = state_dict[key]
                del state_dict[key]
        msg = model.load_state_dict(state_dict, strict=False)
        print(f'load checkpoint from {self.model_checkpoints_path}')
        print(msg)
        return model

    def run(self, image_result_list: List[Tuple[str, torch.Tensor]]):
        """inference"""
        assert image_result_list, f"inference data is empty!image_result_list={image_result_list}"

        batch_predict_result_list = []
        with torch.no_grad() and autocast():
            for image_no, (image_file_path, image) in enumerate(image_result_list):

                image = image.to(self.device, non_blocking=True)
                image = torch.unsqueeze(image, dim=0)
                image_no = [image_no]
                caption = [self.config['eos']]
                question_input = [self.config['bos'] + " "]
                caption = self.tokenizer(caption, padding='longest', truncation=True,
                                         max_length=self.config['max_input_length'],
                                         return_tensors="pt").to(self.device)
                question_input = self.tokenizer(question_input, padding='longest', truncation=True,
                                                max_length=self.config['max_input_length'],
                                                return_tensors="pt").to(self.device)

                topk_ids, topk_probs = self.caption_model(image, question_input, caption, train=False)

                for image_id, topk_id, topk_prob in zip(image_no, topk_ids, topk_probs, ):
                    ans = self.tokenizer.decode(topk_id[0]).replace("[SEP]", "").replace("[CLS]", ""). \
                        replace("[PAD]", "").strip()
                    batch_predict_result_list.append({"question_id": image_id, "pred_caption": ans, })
        return batch_predict_result_list


def main(inputs_image_path_list):
    """对传入的多张图像，模型推理得到 captions, 显示效果"""
    # 0. 待推理数据
    if not inputs_image_path_list:
        logger.info(f"Inputs Empty!inputs_image_path_list={inputs_image_path_list}")
        return True

    data_preprocess = DataPreprocess(cfg=config)
    model_inference = ModelInference(cfg=config)
    # 数据预处理
    success_flag, process_result = data_preprocess.run(inputs_image_path_list)
    if not success_flag:
        raise DataPreprocessError(f"process_result={process_result}")

    # 1. inference
    batch_inference_result = model_inference.run(image_result_list=process_result)
    logger.info(f"inference success! batch_predict_result_list={batch_inference_result}")
    print(batch_inference_result)

    # 2. visualize
    pass


if __name__ == '__main__':
    image_coco2014_base_path = Path(
        r"C:\Users\Loki\workspace\Hobby_CVinternship\dataset_object_detection\coco2014\val2014_img/").resolve()
    image_path_list: List[str] = [
        image_coco2014_base_path.joinpath("COCO_val2014_000000000395.jpg").as_posix(),
        image_coco2014_base_path.joinpath("COCO_val2014_000000000564.jpg").as_posix(),
    ]
    main(image_path_list)
