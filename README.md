# nukesd - Stable Diffusion Implementation in Nuke

![cover](./icons/cover.gif)

`nukesd` is a package that brings stable diffusion implementation to Nuke, enhancing your compositing and visual effects pipeline.

## Installation

To integrate `nukesd` into Nuke, you will need to modify your `main.py` file located inside the `.nuke` directory.

First, ensure that Nuke has access to your gizmos path by appending the path where you installed the package to `sys.path`.

## Usage

After setting up the path, you need to add the following lines to your `main.py`:

```python
import cnnuke
import sdnuke

from sdnuke import *
from cnnuke import *

m =toolbar.addMenu("SD", icon="sd.png")
#
m.addCommand("sd_CN", "cnnuke.create_cn_gizmo_instance()")
m.addCommand("sd_text2img", "StableDiffusionGenerate()")
m.addCommand("sd_text2imgSAG", "StableDiffusionSAGPipeline()")
m.addCommand("sd_img2img", "StableDiffusionImg2Img()")
m.addCommand("sd_inPaint", "StableDiffusionInpaintPipeline()")
m.addCommand("sd_paintByExample", "StableDiffusionPaintByExample()")
m.addCommand("sd_variation", "StableDiffusionImageVariationPipeline()")
m.addCommand("sd_rePaint", "StableDiffusionRePaintPipelinePipeline()")
m.addCommand("sd_upScale", "StableDiffusionUpscalePipeline()")
```
This code will add a new menu ("SD") to your Nuke toolbar. The menu includes two commands:
- "nukeSD" to create a new instance of the Stable Diffusion gizmo
- "nukeCN" to create a new instance of the Control Net gizmo

## Dependencies
Ensure to install all the necessary packages listed in the `requirements.txt` file using the command:

```bash
pip install -r requirements.txt
```

## Contact
If you encounter any issues or have questions, feel free to open an issue on this GitHub repository.
