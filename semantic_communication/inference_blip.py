import torch
from torchvision import transforms
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration, Blip2Processor, Blip2Model, Blip2ForConditionalGeneration

class BLIP():
    def __init__(self, device):
        # Load the BLIP model configuration and model
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = model.to(device)

    def caption_image(self, device, image):
        # Load and preprocess the image
        # Preprocess the image and convert the input tensor to torch.float16
        inputs = self.processor(image, return_tensors="pt").to(device)

        # Generate captions without gradient computation
        generated_ids = self.model.generate(**inputs, max_new_tokens=100)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text

class BLIPVQA():
    def __init__(self, device):
        # Load the BLIP model configuration and model
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b-coco")
        model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b-coco", torch_dtype=torch.float16)
        self.model = model.to(device)

    def describe_image(self, device, vqa, image):
        # Load and preprocess the image
        # Preprocess the image and convert the input tensor to torch.float16
        inputs = self.processor(image, vqa, return_tensors="pt").to(device)

        # Generate captions without gradient computation
        inputs = self.processor(images=image, text=vqa, return_tensors="pt").to(device, torch.float16)

        outputs = self.model(**inputs)
        print(outputs)


class BLIP2():
    def __init__(self, device):
        # Load the BLIP model configuration and model
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2ForConditionalGeneration.from_pretrained(
                    "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
                )
        self.model = model.to(device)

    def caption_image(self, device, image):
        # Load and preprocess the image
        # Preprocess the image and convert the input tensor to torch.float16
        inputs = self.processor(image, return_tensors="pt").to(device, torch.float16)

        # Generate captions without gradient computation
        generated_ids = self.model.generate(**inputs, max_new_tokens=200)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text
