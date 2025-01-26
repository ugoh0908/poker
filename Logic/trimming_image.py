import os
from PIL import Image
from torchvision import models
import torch.nn as nn
import torch
import torchvision.transforms as transforms


class trimming_image:
    def extract_card_text(
        self, image, region, test_transform, model, class_names, screenshot_save_path
    ):
        # 指定した領域をトリミング
        cropped_image = image.crop(region)
        cropped_image.save("hand.png")
        # 3. 画像を読み込み → 前処理 → 推論
        print("Current working directory:", os.getcwd())
        img_path = "hand.png"
        img = Image.open(img_path).convert("RGB")
        input_tensor = test_transform(img).unsqueeze(0)  # (N,C,H,W) にする

        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            # predicted はクラスID（整数）

        # 推論結果をクラス名に変換
        predicted_class = class_names[predicted.item()]

        # 切り取ったスクリーンショットを保存する
        target_path = os.path.join(screenshot_save_path, predicted_class)
        if not os.path.exists(target_path):
            os.makedirs(target_path)

        existing_files = os.listdir(target_path)
        numbering = len(existing_files) + 1

        file_name = f"{numbering}.png"
        file_path = os.path.join(target_path, file_name)
        cropped_image.save(file_path, "PNG")

        # クラス名を返す
        return predicted_class
