# flake8: noqa
# This file is used for deploying replicate models
# running: cog predict -i img=@inputs/whole_imgs/10045.png -i version='v1.4' -i scale=2
# push: cog push r8.im/tencentarc/gfpgan
# push (backup): cog push r8.im/xinntao/gfpgan

import os
import time

os.system('python setup.py develop')
os.system('pip install realesrgan')

import cv2
import shutil
import tempfile
import torch
import requests
import av
import numpy as np
from basicsr.archs.srvgg_arch import SRVGGNetCompact

from gfpgan import GFPGANer

try:
    from cog import BasePredictor, Input, Path
    from realesrgan.utils import RealESRGANer
except Exception:
    print('please install cog and realesrgan package')

def vid2frames(vidPath, framesOutPath):
    vidcap = cv2.VideoCapture(vidPath)
    success,image = vidcap.read()
    frame = 1
    while success:
      yield image
      success,image = vidcap.read()
      frame += 1


class Predictor(BasePredictor):

    def setup(self):
        os.makedirs('output', exist_ok=True)
        # download weights
        if not os.path.exists('gfpgan/weights/realesr-general-x4v3.pth'):
            os.system(
                'wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth -P ./gfpgan/weights'
            )
        if not os.path.exists('gfpgan/weights/GFPGANv1.2.pth'):
            os.system(
                'wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.2.pth -P ./gfpgan/weights')
        if not os.path.exists('gfpgan/weights/GFPGANv1.3.pth'):
            os.system(
                'wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth -P ./gfpgan/weights')
        if not os.path.exists('gfpgan/weights/GFPGANv1.4.pth'):
            os.system(
                'wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth -P ./gfpgan/weights')
        if not os.path.exists('gfpgan/weights/RestoreFormer.pth'):
            os.system(
                'wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth -P ./gfpgan/weights'
            )

        # background enhancer with RealESRGAN
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        model_path = 'gfpgan/weights/realesr-general-x4v3.pth'
        half = True if torch.cuda.is_available() else False
        self.upsampler = RealESRGANer(
            scale=4, model_path=model_path, model=model, tile=0, tile_pad=10, pre_pad=0, half=half)

        # Use GFPGAN for face enhancement
        self.face_enhancer = GFPGANer(
            model_path='gfpgan/weights/GFPGANv1.4.pth',
            upscale=2,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=self.upsampler)
        self.current_version = 'v1.4'

    def predict(
            self,
            video: Path = Input(description='Input'),
            version: str = Input(
                description='GFPGAN version. v1.3: better quality. v1.4: more details and better identity.',
                choices=['v1.2', 'v1.3', 'v1.4', 'RestoreFormer'],
                default='v1.4'),
            scale: float = Input(description='Rescaling factor', default=2),
    ) -> Path:
        weight = 0.5
        print(video, version, scale, weight)
        try:

            input_container = av.open(str(video))
            input_stream = input_container.streams.video[0]

            width, height = input_stream.width, input_stream.height
            fps = input_stream.average_rate

            print("video width: ", width)
            print("video height: ", height)
            print("fps: ", fps)

            new_width = int(width * scale * 2)
            new_height = int(height * scale * 2)

            print("output video width: ", new_width)
            print("output video height: ", new_height)

            output_container = av.open('out.mp4', mode='w')
            output_stream = output_container.add_stream('mpeg4', rate=fps)
            output_stream.width = new_width
            output_stream.height = new_height
            output_stream.pix_fmt = 'yuv420p'

            print("output path: ", 'out.mp4')

            for frame in input_container.decode(input_stream):

                img = frame.to_ndarray(format='bgr24')

                if len(img.shape) == 3 and img.shape[2] == 4:
                    img_mode = 'RGBA'
                elif len(img.shape) == 2:
                    img_mode = None
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                else:
                    img_mode = None

                h, w = img.shape[0:2]
                if h < 300:
                    img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)

                if self.current_version != version:
                    if version == 'v1.2':
                        self.face_enhancer = GFPGANer(
                            model_path='gfpgan/weights/GFPGANv1.2.pth',
                            upscale=2,
                            arch='clean',
                            channel_multiplier=2,
                            bg_upsampler=self.upsampler)
                        self.current_version = 'v1.2'
                    elif version == 'v1.3':
                        self.face_enhancer = GFPGANer(
                            model_path='gfpgan/weights/GFPGANv1.3.pth',
                            upscale=2,
                            arch='clean',
                            channel_multiplier=2,
                            bg_upsampler=self.upsampler)
                        self.current_version = 'v1.3'
                    elif version == 'v1.4':
                        self.face_enhancer = GFPGANer(
                            model_path='gfpgan/weights/GFPGANv1.4.pth',
                            upscale=2,
                            arch='clean',
                            channel_multiplier=2,
                            bg_upsampler=self.upsampler)
                        self.current_version = 'v1.4'
                    elif version == 'RestoreFormer':
                        self.face_enhancer = GFPGANer(
                            model_path='gfpgan/weights/RestoreFormer.pth',
                            upscale=2,
                            arch='RestoreFormer',
                            channel_multiplier=2,
                            bg_upsampler=self.upsampler)

                try:
                    print("calling enhancer...")
                    _, _, output = self.face_enhancer.enhance(
                        img, has_aligned=False, only_center_face=False, paste_back=True, weight=weight)

                    print("got enhanced output")
                    print("output shape: ", output.shape)
                    print("output type: ", output.dtype)
                except RuntimeError as error:
                    print('Error', error)

                try:
                    if scale != 2:
                        interpolation = cv2.INTER_AREA if scale < 2 else cv2.INTER_LANCZOS4
                        h, w = img.shape[0:2]
                        output = cv2.resize(output, (int(w * scale / 2), int(h * scale / 2)), interpolation=interpolation)
                except Exception as error:
                    print('wrong scale input.', error)

                # Ensure the upscaled frame is in uint8 format
                if output.dtype != np.uint8:
                    print("converting output")
                    output = (output * 255).astype(np.uint8)

                print("writing frame to video...")
                output_frame = av.VideoFrame.from_ndarray(output, format='bgr24')
                output_container.mux(output_stream.encode(output_frame))

                print("wrote frame to video")

            output_container.close()

            return 'out.mp4'
        except Exception as error:
            print('global exception: ', error)
        finally:
            clean_folder('output')


def clean_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
