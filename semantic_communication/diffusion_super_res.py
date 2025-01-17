from diffusers import LDMSuperResolutionPipeline, StableDiffusionUpscalePipeline
import diffusers
import warnings, random, torch
warnings.filterwarnings(action='ignore', message=r'Passing.*is deprecated', category=FutureWarning)

diffusers.logging.set_verbosity_error()
diffusers.logging.disable_progress_bar()


class DiffusionSuperRes():
    def __init__(self, device):
        self.idx = 0
        self.model_id = "CompVis/ldm-super-resolution-4x-openimages"
        pipe = LDMSuperResolutionPipeline.from_pretrained(self.model_id)
        self.device = device
        self.pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        
    def inference(self, image, num_inference_steps=100):
        image = self.pipe(image, num_inference_steps=num_inference_steps, eta=1).images[0]
        return image


class DiffusionUpscaler():
    def __init__(self, device):
        self.idx = 0
        self.model_id = "stabilityai/stable-diffusion-x4-upscaler"
        pipe = StableDiffusionUpscalePipeline.from_pretrained(
            self.model_id, revision="fp16", torch_dtype=torch.float16
        )
        self.device = device
        self.pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        
    def inference(self, prompt, image, num_inference_steps=100):
        image = self.pipe(prompt=prompt, image=image, num_inference_steps=num_inference_steps, eta=1).images[0]
        return image
    