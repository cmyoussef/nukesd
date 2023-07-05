import argparse
import ast
import random
import sys
import inspect

import torch
from diffusers import (ControlNetModel, UniPCMultistepScheduler, StableDiffusionPipeline,
                       StableDiffusionControlNetPipeline, StableDiffusionImg2ImgPipeline,
                       StableDiffusionDepth2ImgPipeline, StableDiffusionInpaintPipeline,
                       StableDiffusionSAGPipeline, StableDiffusionUpscalePipeline,
                       StableDiffusionImageVariationPipeline, DiffusionPipeline, RePaintPipeline,
                       RePaintScheduler)
from diffusers.utils import load_image


class StableDiffusionGenerator:
    def __init__(self, paths, cache_dir):
        self.paths = paths
        self.cache_dir = cache_dir
        self.args_dict = {}
        self.add_paths()

    def add_paths(self):
        for path in self.paths:
            if path not in sys.path:
                sys.path.append(path)

    def setup_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--output', required=True, help='Output file path')
        parser.add_argument('--model', required=True, help='Model ID')
        parser.add_argument('--prompt', required=True, help='Prompt text')
        parser.add_argument('--negative_prompt', required=False, help='Negative prompt text')
        parser.add_argument('--width', required=False, help='width of the output image', type=int)
        parser.add_argument('--height', required=False, help='height of the output image', type=int)
        parser.add_argument('--controlNet', required=False, help='pass the control net')
        parser.add_argument('--seed', required=False, help='seed -1 is random', type=int)
        parser.add_argument('--num_images_per_prompt', required=False,
                            help='The number of images to generate per prompt', type=int)
        parser.add_argument('--guidance_scale', required=False, help='Guidance scale', type=float)
        parser.add_argument('--num_inference_steps', required=False,
                            help='The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.',
                            type=int)
        parser.add_argument('--latents', required=False,
                            help='Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image generation')
        parser.add_argument('--img2img', required=False,
                            help='lets you pass a text prompt and an initial image to condition the generation of new images using Stable Diffusion.')
        parser.add_argument('--use_depth', required=False,
                            help='lets you pass a text prompt and an initial image to condition the \
                                    generation of new images as well as a depth_map to preserve the images’ structure.',
                            type=bool)
        parser.add_argument('--inPaint', required=False,
                            help='lets you edit specific parts of an image by providing a mask and a text prompt using Stable Diffusion.')
        parser.add_argument('--rePaintPipeline', required=False,
                            help='Free-form inpainting is the task of adding new content to an image in the regions specified by an arbitrary binary mask.')
        parser.add_argument('--paintByExample', required=False,
                            help='lets you edit specific parts of an image by providing a mask and a text prompt using Stable Diffusion.')
        parser.add_argument('--SAG', required=False,
                            help='Self-Attention Guidance (SAG) uses the intermediate self-attention maps of diffusion models to enhance their stability and efficacy. Specifically, \
                                  SAG adversarially blurs only the regions that diffusion models attend to at each iteration and guides them accordingly.')
        parser.add_argument('--upScale', required=False,
                            help='can be used to enhance the resolution of input images by a factor of 4.')
        parser.add_argument('--imageVariation', required=False,
                            help='lets you generate variations from an input image using Stable Diffusion. It uses a fine-tuned version of Stable Diffusion model.')
        parser.add_argument('--unsupported_args', required=False, help='unsupported args based on each model.')
        return parser

    def parse_args(self):
        parser = self.setup_parser()
        args = parser.parse_args()
        self.args_dict = vars(args)

    def process_args(self):
        # Convert args to dictionary for easier manipulation

        # Handle random seed for generator
        seed = self.args_dict.pop('seed', -1)
        seed = random.randint(0, 999999) if seed == -1 else seed
        generator = torch.Generator(device="cuda").manual_seed(seed)
        self.args_dict['generator'] = generator

        # Handle unsupported arguments
        unsupported_args = ast.literal_eval(self.args_dict.pop('unsupported_args', []))
        for a in unsupported_args:
            if a in self.args_dict:
                self.args_dict.pop(a)

        return self.args_dict

    def load_model(self):
        self.model = self.args_dict.pop('model')

    def create_pipeline(self, args_dict=None):

        args_dict = args_dict or self.args_dict
        # Extract specific argument values and remove them from args_dict
        controlNets = args_dict.pop('controlNet')
        img2img = args_dict.pop('img2img')
        img2img_depth = args_dict.pop('use_depth', False)
        inPaint = args_dict.pop('inPaint')
        rePaintPipeline = args_dict.pop('rePaintPipeline')
        sag = args_dict.pop('SAG')
        upScale = args_dict.pop('upScale')
        paintByExample = args_dict.pop('paintByExample')
        imageVariation = args_dict.pop('imageVariation')

        # Determine pipeline type and create appropriate pipeline
        if controlNets and not img2img:
            # Handle control nets case
            self.control_nets_pipeline(controlNets)
        elif img2img and not controlNets:
            # Handle img2img case
            self.img2img_pipeline(img2img, img2img_depth)
        elif inPaint:
            # Handle inPaint case
            self.inpaint_pipeline(inPaint)
        elif rePaintPipeline:
            # Handle rePaintPipeline case
            self.repaint_pipeline(rePaintPipeline)
        elif paintByExample:
            # Handle paintByExample case
            self.paint_by_example_pipeline(paintByExample)
        elif sag:
            # Handle SAG case
            self.sag_pipeline()
        elif upScale:
            # Handle upScale case
            self.upscale_pipeline(upScale)
        elif imageVariation:
            # Handle imageVariation case
            self.image_variation_pipeline(imageVariation)
        else:
            # Default pipeline case
            self.default_pipeline()
        return self.pipe

    def generate_output(self):
        # Transfer the pipeline model to GPU
        self.pipe = self.pipe.to("cuda")
        self.pipe.enable_xformers_memory_efficient_attention()
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)

        # Get a list of argument names in the __init__ of self.pipe
        call_args = inspect.getfullargspec(self.pipe.__call__).args

        # Remove 'self' from the list as it's not an argument that can be passed
        if 'self' in call_args:
            call_args.remove('self')

        # Create a new dictionary containing only the keys that are in pipe_args
        relevant_args = {k: v for k, v in self.args_dict.items() if k in call_args}

        outputs_batch = ast.literal_eval(self.args_dict.pop('output'))
        # Generate output image
        print('pipe_args')
        print(call_args)
        print('-' * 10)
        print('relevant_args')
        print(relevant_args)
        print('-' * 10)
        print('self.args_dict')
        print(self.args_dict)
        print('-' * 10)

        for i, outputs in enumerate(outputs_batch):
            images_tensor = self.pipe(**self.args_dict).images
            for image_tensor, output in zip(images_tensor, outputs):
                # Save the output image tensor to file
                image_tensor.save(output)

    def control_nets_pipeline(self, controlNets):
        controlNets = ast.literal_eval(controlNets)
        controlNetTypes, images, weights = zip(*controlNets)
        self.args_dict['controlnet_conditioning_scale'] = weights
        images = [load_image(i.replace('\\', '/')) for i in images]
        self.args_dict['image'] = images
        controlnetModels = [ControlNetModel.from_pretrained(c, torch_dtype=torch.float16) for c in controlNetTypes]
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(self.model, controlnet=controlnetModels,
                                                                      torch_dtype=torch.float16,
                                                                      cache_dir=self.cache_dir,
                                                                      safety_checker=None)

    def img2img_pipeline(self, img2img, img2img_depth):
        img2img_path, strength = ast.literal_eval(img2img)
        self.args_dict['strength'] = strength
        self.args_dict['image'] = load_image(img2img_path)

        if img2img_depth:
            self.model = 'stabilityai/stable-diffusion-2-depth'
            self.pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(self.model, torch_dtype=torch.float16,
                                                                         cache_dir=self.cache_dir,
                                                                         safety_checker=None)
        else:
            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(self.model, torch_dtype=torch.float16,
                                                                       cache_dir=self.cache_dir,
                                                                       safety_checker=None)

    def inpaint_pipeline(self, inPaint):
        init_image, mask_image = [load_image(i.replace('\\', '/')) for i in ast.literal_eval(inPaint)]
        self.args_dict['image'] = init_image
        self.args_dict['mask_image'] = mask_image
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(self.model, torch_dtype=torch.float16,
                                                                   cache_dir=self.cache_dir,
                                                                   safety_checker=None)

    def repaint_pipeline(self, rePaintPipeline):
        init_image, mask_image, jump_length, jump_n_sample, eta = ast.literal_eval(rePaintPipeline)
        init_image, mask_image = [load_image(i.replace('\\', '/')) for i in [init_image, mask_image]]
        self.args_dict['image'] = init_image
        self.args_dict['mask_image'] = mask_image
        self.args_dict.pop('num_images_per_prompt')
        self.args_dict.pop('guidance_scale')
        self.args_dict.pop('generator')
        self.args_dict['jump_length'] = jump_length
        self.args_dict['jump_n_sample'] = jump_n_sample
        self.args_dict['eta'] = eta

        scheduler = RePaintScheduler.from_pretrained("google/ddpm-ema-celebahq-256")
        self.pipe = RePaintPipeline.from_pretrained("google/ddpm-ema-celebahq-256", scheduler=scheduler)

    def paint_by_example_pipeline(self, paintByExample):
        init_image, mask_image, example_image = [load_image(i.replace('\\', '/')) for i in ast.literal_eval(paintByExample)]
        self.args_dict['image'] = init_image
        self.args_dict['mask_image'] = mask_image
        self.args_dict['example_image'] = example_image
        self.pipe = DiffusionPipeline.from_pretrained(self.model, torch_dtype=torch.float16,
                                                      cache_dir=self.cache_dir,
                                                      safety_checker=None)

    def sag_pipeline(self):
        self.pipe = StableDiffusionSAGPipeline.from_pretrained(self.model, torch_dtype=torch.float16,
                                                               cache_dir=self.cache_dir,
                                                               safety_checker=None)

    def upscale_pipeline(self, upScale):
        # self.imageSize = int(upScale)
        self.args_dict['image'] = load_image(upScale)
        self.pipe = StableDiffusionUpscalePipeline.from_pretrained(self.model, torch_dtype=torch.float16,
                                                                   cache_dir=self.cache_dir,
                                                                   safety_checker=None)

    def image_variation_pipeline(self, imageVariation):
        self.args_dict['image'] = load_image(imageVariation)
        self.pipe = StableDiffusionImageVariationPipeline.from_pretrained(self.model, torch_dtype=torch.float16,
                                                                          cache_dir=self.cache_dir,
                                                                          safety_checker=None)

    def default_pipeline(self):
        self.pipe = StableDiffusionPipeline.from_pretrained(self.model, torch_dtype=torch.float16,
                                                            cache_dir=self.cache_dir,
                                                            safety_checker=None)


if __name__ == "__main__":
    paths = ['D:/stable-diffusion/stable-diffusion-webui/',
             'D:/stable-diffusion/stable-diffusion-webui/repositories/stable-diffusion-stability-ai']
    cache_dir = 'D:/stable-diffusion/pip_cach/hub/'

    generator = StableDiffusionGenerator(paths, cache_dir)
    generator.parse_args()
    args = generator.process_args()
    generator.load_model()
    generator.create_pipeline(args)
    generator.generate_output()
