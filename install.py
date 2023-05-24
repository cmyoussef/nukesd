#!/usr/bin/env python3
import subprocess

subprocess.run(["pip", "install", "torch==2.0.0+cu118", "torchvision==0.15.1+cu118", "--extra-index-url", "https://download.pytorch.org/whl/cu118"])
subprocess.run(["pip", "install", "-r", "requirements.txt"])
