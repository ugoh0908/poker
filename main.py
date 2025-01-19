import pygetwindow as gw
import pyautogui
from PIL import Image
import pytesseract
import time
from torchvision import models
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import os


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
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225])
])

# 3. クラス名を表示
class_names = [
"10_clover",
"10_diamond",
"10_herat",
"10_spade",
"1_clover",
"1_diamond",
"1_herat",
"1_spade",
"2_clover",
"2_diamond",
"2_herat",
"2_spade",
"3_clover",
"3_diamond",
"3_herat",
"3_spade",
"4_clover",
"4_diamond",
"4_herat",
"4_spade",
"5_clover",
"5_diamond",
"5_herat",
"5_spade",
"6_clover",
"6_diamond",
"6_herat",
"6_spade",
"7_clover",
"7_diamond",
"7_herat",
"7_spade",
"8_clover",
"8_diamond",
"8_herat",
"8_spade",
"9_clover",
"9_diamond",
"9_herat",
"9_spade",
"J_clover",
"J_diamond",
"J_herat",
"J_spade",
"K_clover",
"K_diamond",
"K_herat",
"K_spade",
"Q_clover",
"Q_diamond",
"Q_herat",
"Q_spade"]


def get_screenshot(window_title):
    # 指定したウィンドウを探す
    windows = gw.getWindowsWithTitle(window_title)
    if not windows:
        print("指定されたウィンドウが見つかりません。")
        return None
    window = windows[0]

    # ウィンドウの位置とサイズを取得
    x, y, width, height = window.left, window.top, window.width, window.height

    # スクリーンショットを撮影
    screenshot = pyautogui.screenshot(region=(x, y, width, height))
    return screenshot

def extract_card_text(image, region):
    # 指定した領域をトリミング
    cropped_image = image.crop(region)
    cropped_image.save("test_eight_of_spades.png")
    # 3. 画像を読み込み → 前処理 → 推論
    print("Current working directory:", os.getcwd())
    img_path = "test_eight_of_spades.png"
    img = Image.open(img_path).convert("RGB")  
    input_tensor = test_transform(img).unsqueeze(0)  # (N,C,H,W) にする

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        # predicted はクラスID（整数）

    # 推論結果をクラス名に変換
    predicted_class = class_names[predicted.item()]
    
    # クラス名を返す
    return predicted_class

    
def show_cropped_image(image, region):
    # 指定した領域をトリミング
    cropped_image = image.crop(region)

def main():
    window_title = "MEmu"  # エミュレータのウィンドウタイトル
    card_region =  (260, 975, 380, 1125)   # カードが表示されるエリアを調整

    while True:
        screenshot = get_screenshot(window_title)
        if screenshot:
            
            card_text = extract_card_text(screenshot, card_region)
            if card_text:
                
                print("現在のカード:", card_text)
        time.sleep(51)  # 1秒ごとに更新

if __name__ == "__main__":
    main()