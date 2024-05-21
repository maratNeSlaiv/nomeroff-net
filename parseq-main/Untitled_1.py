import torch
from PIL import Image
from strhub.data.module import SceneTextDataModule

# Load model and image transforms
def OCR_predict(parseq, image : Image):
    parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()
    img_transform = SceneTextDataModule.get_transform(parseq.hparams.img_size)

    # Preprocess. Model expects a batch of images with shape: (B, C, H, W)
    img = img_transform(img).unsqueeze(0)

    logits = parseq(img)
    logits.shape  # torch.Size([1, 26, 95]), 94 characters + [EOS] symbol

    # Greedy decoding
    pred = logits.softmax(-1)
    label, confidence = parseq.tokenizer.decode(pred)
    return label, confidence
