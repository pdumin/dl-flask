import torch

def get_sentiment(model, inputs):
    proba = torch.sigmoid(model(**inputs).logits).cpu().detach().numpy()[0]
    return model.config.id2label[proba.argmax()]
