import torch
import torchvision
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_cnn_model():
    cnn_model = torchvision.models.resnet18(pretrained=True)
    cnn_model.eval()
    return cnn_model

def load_text_model():
    model_checkpoint = 'cointegrated/rubert-tiny-sentiment-balanced'
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    text_model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
    text_model.eval()
    return text_model, tokenizer