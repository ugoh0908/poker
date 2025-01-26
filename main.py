import tkinter as tk
from threading import Thread
import pygetwindow as gw
import pytesseract
import time
from torchvision import models
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import os
import random
from poker_calculate_logic import poker_calculate_logic
from image_create import image_create
from Logic.trimming_image import trimming_image
from Const.mainconst import mainconst
from Logic.change_card_name import change_card_name
from Flow.main_flow import main_flow


# Tesseractのパスを設定（インストール済みであることが前提）
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# カードのクラス数を明示的に指定（例: 52）
num_classes = 52
# 1. モデルを定義し、学習済み重みを読み込む
model = models.resnet18(pretrained=False)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)
model.load_state_dict(torch.load("card_classifier.pth", map_location="cpu"))
model.eval()

# 2. 同じ前処理を定義
test_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def main():

    instance_main_flow = main_flow()  # main_flow クラスのインスタンスを作成
    instance_main_flow.main_flow()


if __name__ == "__main__":
    main()
