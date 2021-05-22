import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

import tqdm
import argparse

import time

from emotion_classification.dataset.emotion_dataset import PredictDataSet
from emotion_classification.model.model import *

classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


def predict(device, image_path, weight_path, batch_size):
    if device == '':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device: {}'.format(device))

    transform = transforms.Compose([transforms.Resize([48, 48]), transforms.ToTensor()])

    dataset = PredictDataSet(image_path, transform=transform)
    print('DataSet: PredictDataSet')
    print(f'DataSet size: {len(dataset)}')

    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = darknet(num_classes=7, num_blocks=[1, 2, 8])
    model.to(device)

    model.load_state_dict(torch.load(weight_path, map_location=device))

    model.eval()

    pbar = tqdm.tqdm(dl)

    preds = []

    since = time.time()
    for x in pbar:
        x = x.to(device)
        out = model(x)

        pred = torch.argmax(out, dim=1)
        preds += pred.tolist()
    print('Time: {:.2f}s'.format(time.time() - since))
    preds = [classes[p] for p in preds]
    return preds


if __name__ == '__main__':
    image_path = 'inference/images'
    weight_path = 'weights/model_epoch_53.pt'
    batch_size = 8
    pred = predict('cpu', image_path, weight_path, batch_size)
    print(pred)
