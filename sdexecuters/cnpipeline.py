import argparse
import ast

import cv2
import numpy as np
from PIL import Image
from diffusers.utils import load_image

from adapalette import ada_palette

# Parse command line arguments

parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True, help='Output file path')
parser.add_argument('--input', required=True, help='Input file path')
parser.add_argument('--model', required=True, help='Model ID')
parser.add_argument('--model_kwargs', required=False, help='Model kwargs')
args = parser.parse_args()

args_dict = vars(args)

ada_palette = np.array(ada_palette)


# <editor-fold desc="converters">
def cannyConvert(image, model_kwargs):
    image = np.array(image)

    min_threshold = int(model_kwargs.get('min_threshold', 100))
    max_threshold = int(model_kwargs.get('max_threshold', 200))

    image = cv2.Canny(image, min_threshold, max_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    control_image = Image.fromarray(image)

    return control_image


def mlsdConvert(image, model_kwargs):
    from controlnet_aux import MLSDdetector
    processor = MLSDdetector.from_pretrained('lllyasviel/ControlNet')
    control_image = processor(image)
    return control_image


def scribbleConvert(image, model_kwargs):
    from controlnet_aux import HEDdetector
    processor = HEDdetector.from_pretrained('lllyasviel/Annotators')
    control_image = processor(image, scribble=True)
    return control_image


def lineartConvert(image, model_kwargs):
    from controlnet_aux import LineartAnimeDetector
    processor = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")
    control_image = processor(image)
    return control_image


def normalConvert(image, model_kwargs):
    from controlnet_aux import NormalBaeDetector
    processor = NormalBaeDetector.from_pretrained('lllyasviel/Annotators')
    control_image = processor(image)
    return control_image


def shuffleConvert(image, model_kwargs):
    from controlnet_aux import ContentShuffleDetector
    processor = ContentShuffleDetector()
    control_image = processor(image)
    return control_image


def poseConvert(image, model_kwargs):
    from controlnet_aux import OpenposeDetector
    processor = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
    control_image = processor(image, hand_and_face=True)
    return control_image


def segConvert(image, model_kwargs):
    import torch
    from transformers import AutoImageProcessor, UperNetForSemanticSegmentation

    image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
    image_segmentor = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small")
    pixel_values = image_processor(image, return_tensors="pt").pixel_values
    with torch.no_grad():
        outputs = image_segmentor(pixel_values)
    seg = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)  # height, width, 3
    for label, color in enumerate(ada_palette):
        color_seg[seg == label, :] = color
    color_seg = color_seg.astype(np.uint8)
    control_image = Image.fromarray(color_seg)
    return control_image


def depthConvert(image, model_kwargs):
    from transformers import pipeline
    depth_estimator = pipeline('depth-estimation')
    image = depth_estimator(image)['depth']
    image = np.array(image)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    depth_image = Image.fromarray(image)
    return depth_image


def resize_for_condition_image(input_image: Image, resolution: int):
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(round(H / 64.0)) * 64
    W = int(round(W / 64.0)) * 64
    img = input_image.resize((W, H), resample=Image.LANCZOS)
    return img


# </editor-fold>

controlNet = {"lllyasviel/control_v11p_sd15_canny": cannyConvert,
              "lllyasviel/control_v11e_sd15_ip2p": None,
              "lllyasviel/control_v11p_sd15_inpaint": None,
              "lllyasviel/control_v11p_sd15_mlsd": mlsdConvert,
              "lllyasviel/control_v11f1p_sd15_depth": depthConvert,
              "lllyasviel/control_v11p_sd15_normalbae": normalConvert,
              "lllyasviel/control_v11p_sd15_seg": segConvert,
              "lllyasviel/control_v11p_sd15_lineart": lineartConvert,
              "lllyasviel/control_v11p_sd15s2_lineart_anime": lineartConvert,
              "lllyasviel/control_v11p_sd15_openpose": poseConvert,
              "lllyasviel/control_v11p_sd15_scribble": scribbleConvert,
              "lllyasviel/control_v11p_sd15_softedge": None,
              "lllyasviel/control_v11e_sd15_shuffle": shuffleConvert
              }

image = load_image(args.input)
outputImage = controlNet[args.model](image, ast.literal_eval(args.model_kwargs))
outputImage.save(args.output)
