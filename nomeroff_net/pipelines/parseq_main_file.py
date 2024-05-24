import torch
from PIL import Image
from nomeroff_net.pipelines.parseq_main_file_1 import SceneTextDataModule
from .parseq_main_file_1 import SceneTextDataModule

# Load model and image transforms
def OCR_parseq(image : Image):
    parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()
    
    img_transform = SceneTextDataModule.get_transform(parseq.hparams.img_size)

    # Preprocess. Model expects a batch of images with shape: (B, C, H, W)
    image = img_transform(image).unsqueeze(0)

    logits = parseq(image)
    # Greedy decoding
    temperature = -1
    pred = logits.softmax(temperature)
    label, confidence = parseq.tokenizer.decode(pred)
    return label[0]

# Use case
# img = Image.open('/Users/maratorozaliev/Desktop/autoriaNumberplateOcrKg-2022-11-30/test/img/8155782_0_2.png').convert('RGB')
# print(OCR_parseq(image=img))
