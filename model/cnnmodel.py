import torch
import json
import torchvision
from PIL import Image
import numpy as np

# читаем изображение
def read_and_preprocess(path: str):
    # img = torchvision.io.read_image(path) / 255.
    # Read image
    img = torch.permute(torch.Tensor(np.array(Image.open(path))) /255., (2, 0, 1))
    print(img.shape)

    
    # print(type(img))
    resize = torchvision.transforms.Resize((227, 227))
    img = resize(img)
    # print(img)
    return img

def load_classes():
    # читаем файл с классами imagenet
    with open('model/imagenet_class_index.json', 'r') as file:
        classes = file.read()
    classes = json.loads(classes)
    return classes

def get_predictions(model, img, classes):
    # получаем предсказания
    idx = torch.argmax(model(img.unsqueeze(0)), dim=1).numpy()
    # декодируем
    result = classes[str(idx[0])][1]
    return result

def predict(model, filepath):
    img = read_and_preprocess(filepath)
    classes = load_classes()
    img = read_and_preprocess(filepath)
    # print(img.shape)
    pred = get_predictions(model, img, classes)
    return pred
    
