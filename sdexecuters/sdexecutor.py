# Import necessary modules
from diffusers import ControlNetModel, UniPCMultistepScheduler, StableDiffusionPipeline, \
    StableDiffusionControlNetPipeline
import torch
import argparse
import ast
import random
from diffusers.utils import load_image

# Set up command line argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True, help='Output file path')
parser.add_argument('--model', required=True, help='Model ID')
parser.add_argument('--prompt', required=True, help='Prompt text')
parser.add_argument('--width', required=False, help='width of the output image', type=int)
parser.add_argument('--height', required=False, help='height of the output image', type=int)
parser.add_argument('--seed', required=False, help='seed -1 is random', type=int)
parser.add_argument('--negative_prompt', required=False, help='Negative prompt text')
parser.add_argument('--controlNet', required=False, help='pass the control net')
parser.add_argument('--num_inference_steps', required=False, help='Num inference steps', type=int)
args = parser.parse_args()

# Convert args to dictionary for easier manipulation
args_dict = vars(args)

# Handle random seed for generator
seed = args_dict.pop('seed')
seed = random.randint(0, 999999) if seed == -1 else seed
generator = torch.Generator(device="cuda").manual_seed(seed)
args_dict['generator'] = generator

# Load model
model = args_dict.pop('model')
cache_dir = 'D:/stable-diffusion/pip_cach/hub/'

# Check if control net is provided
controlNets = args_dict.pop('controlNet')
if controlNets:
    # Process control nets if present
    controlNets = ast.literal_eval(controlNets)
    controlNetTypes, images, weights = zip(*controlNets)
    # add the weights
    args_dict['controlnet_conditioning_scale'] = weights
    # Load control net images
    images = [load_image(i.replace('\\', '/')) for i in images]
    args_dict['image'] = images

    # Load control net models
    controlnetModels = [ControlNetModel.from_pretrained(c, torch_dtype=torch.float16) for c in controlNetTypes]
    # Initialize pipeline with control nets
    pipe = StableDiffusionControlNetPipeline.from_pretrained(model, controlnet=controlnetModels,
                                                             torch_dtype=torch.float16, cache_dir=cache_dir)
else:
    # Initialize pipeline without control nets
    pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16, cache_dir=cache_dir)

# Transfer the pipeline model to GPU
pipe = pipe.to("cuda")
pipe.enable_xformers_memory_efficient_attention()
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# Generate output image
output = args_dict.pop('output')
image_tensor = pipe(**args_dict).images[0]

# Save the output image tensor to file
image_tensor.save(output)
