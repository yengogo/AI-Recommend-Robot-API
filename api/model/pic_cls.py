from PIL import Image
# import requests
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
import torch
import os
from glob import glob
import json
from tqdm import tqdm
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ChineseCLIPModel.from_pretrained(
    "OFA-Sys/chinese-clip-vit-base-patch16").to(device)
processor = ChineseCLIPProcessor.from_pretrained(
    "OFA-Sys/chinese-clip-vit-base-patch16")

with open('src/totoal_categorylist.json', 'r', encoding='utf-8') as catfile:
    catlist = json.load(catfile)


def zeroshot_cls(image):
    labels = {k: v for v, k in enumerate(catlist)}
    idx_to_label = {v: k for k, v in labels.items()}
    inputs = processor(images=image, return_tensors="pt").to(device)
    image_features = model.get_image_features(**inputs)
    image_features = image_features / \
        image_features.norm(p=2, dim=-1, keepdim=True)  # normalize

    # compute text features
    inputs = processor(text=catlist, padding=True,
                       return_tensors="pt").to(device)
    text_features = model.get_text_features(**inputs)
    text_features = text_features / \
        text_features.norm(p=2, dim=-1, keepdim=True)  # normalize

    # compute image-text similarity scores
    inputs = processor(text=catlist, images=image,
                       return_tensors="pt", padding=True).to(device)
    outputs = model(**inputs)
    # this is the image-text similarity score
    logits_per_image = outputs.logits_per_image
    # probs: [[1.2686e-03, 5.4499e-02, 6.7968e-04, 9.4355e-01]]
    probs = logits_per_image.softmax(dim=1)
    # print(probs)
    y = torch.argmax(probs, dim=1)
    final_label = [idx_to_label[i] for i in y.detach().cpu().numpy()]

    return final_label
