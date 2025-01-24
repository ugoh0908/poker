import tkinter as tk
from threading import Thread
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
from treys import Deck, Evaluator, Card
import random


# Tesseractのパスを設定（インストール済みであることが前提）
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# スクショ保存するパス設定
screenshot_save_path = r"D:\開発\poker\screenshot"


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
"10_heart",
"10_spade",
"1_clover",
"1_diamond",
"1_heart",
"1_spade",
"2_clover",
"2_diamond",
"2_heart",
"2_spade",
"3_clover",
"3_diamond",
"3_heart",
"3_spade",
"4_clover",
"4_diamond",
"4_heart",
"4_spade",
"5_clover",
"5_diamond",
"5_heart",
"5_spade",
"6_clover",
"6_diamond",
"6_heart",
"6_spade",
"7_clover",
"7_diamond",
"7_heart",
"7_spade",
"8_clover",
"8_diamond",
"8_heart",
"8_spade",
"9_clover",
"9_diamond",
"9_heart",
"9_spade",
"J_clover",
"J_diamond",
"J_heart",
"J_spade",
"K_clover",
"K_diamond",
"K_heart",
"K_spade",
"Q_clover",
"Q_diamond",
"Q_heart",
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
    target_path =os.path.join(screenshot_save_path,predicted_class)
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    
    existing_files = os.listdir(target_path)
    numbering=len(existing_files)+1

    file_name=f"{numbering}.png"
    file_path=  os.path.join(target_path,file_name)
    cropped_image.save(file_path,"PNG")



    
    # クラス名を返す
    return predicted_class

    
def show_cropped_image(image, region):
    # 指定した領域をトリミング
    cropped_image = image.crop(region)

def main():
    window_title = "MEmu"  # エミュレータのウィンドウタイトル
    card_region_1st =  (260, 975, 380, 1125)   # カードが表示されるエリアを調整
    card_region_2nd =  (380, 975, 500, 1125)   # カードが表示されるエリアを調整

    screenshot = get_screenshot(window_title)
    if screenshot:
            
        card_text_1st = change_card_name(extract_card_text(screenshot, card_region_1st))
        card_text_2nd = change_card_name(extract_card_text(screenshot, card_region_2nd))
   
        print("現在のカード:1枚目[", card_text_1st,"]、2枚目[",card_text_2nd,"]")

        if(card_text_1st !=card_text_2nd):
            hole_cards = [card_text_1st, card_text_2nd]
            num_opponents = 1
            trials = 10000
    
            win_rate, tie_rate = estimate_win_probability(hole_cards, num_opponents, trials)
            print(f"ホールカード: {hole_cards}, 相手人数: {num_opponents}, 試行回数: {trials}")
            print(f"推定勝率: {win_rate:.2f}%")
            result_label.config(text=f"結果: {win_rate}")
            print(f"推定引き分け率: {tie_rate:.2f}%")

def change_card_name(origin_name):
    split_name=origin_name.split("_")
    if(split_name[0]=="1"):
        split_name[0]="A"
    elif(split_name[0]=="10"):
        split_name[0]="T"
    if(split_name[1]=="diamond"):
        split_name[1]="d"
    elif(split_name[1]=="heart"):
        split_name[1]="h"
    elif(split_name[1]=="clover"):
        split_name[1]="c"
    else:
        split_name[1]="s"
    return  split_name[0]+split_name[1]

def estimate_win_probability(hole_cards_str, num_opponents=3, trials=10000):
    """
    モンテカルロ法を用いて、与えられたホールカードの勝率を推定する。
    
    Parameters:
    -----------
    hole_cards_str : list of str
        例) ["As", "Kd"] のように、'rank' + 'suit' 形式で与える。
        rankは 2,3,4,5,6,7,8,9,T,J,Q,K,A
        suitは s(スペード), h(ハート), d(ダイヤ), c(クラブ)
    num_opponents : int
        相手プレイヤーの人数
    trials : int
        シミュレーション回数
    
    Returns:
    --------
    win_rate : float
        自分のハンドが勝つ確率（%表記）
    tie_rate : float
        自分のハンドが他プレイヤーと引き分けになる確率（%表記）
    """
    
    evaluator = Evaluator()
    
    # 自分のホールカードをCard型に変換
    my_hole_cards = [Card.new(c) for c in hole_cards_str]

    win_count = 0
    tie_count = 0
    
    for _ in range(trials):
        # デッキを作成して、自分のホールカードを取り除く
        deck = Deck()
        for c in my_hole_cards:
            deck.cards.remove(c)
        
        # 相手プレイヤーたちのホールカードをランダムに取得
        opponents_hole_cards = []
        for _ in range(num_opponents):
            opp_cards = [deck.draw(1)[0], deck.draw(1)[0]]
            opponents_hole_cards.append(opp_cards)
        
        # コミュニティカードを5枚取得
        community_cards = [deck.draw(1)[0] for _ in range(5)]
        
        # 自分の評価値（数値が小さいほど強いハンド）
        my_score = evaluator.evaluate(my_hole_cards, community_cards)
        
        # 各相手の評価値を計算
        opponent_scores = [
            evaluator.evaluate(opp_hole, community_cards) for opp_hole in opponents_hole_cards
        ]
        
        # 自分の評価値と相手評価値を比較
        better_count = sum(1 for s in opponent_scores if s < my_score)   # 自分より強い相手
        same_count   = sum(1 for s in opponent_scores if s == my_score)  # 同点の相手
        
        # 自分より強い相手がいなければ → 勝ち or 引き分け
        if better_count == 0:
            # 同点の相手がいれば引き分けの可能性
            if same_count > 0:
                tie_count += 1
            else:
                win_count += 1
    
    win_rate = (win_count / trials) * 100
    tie_rate = (tie_count / trials) * 100
    
    return win_rate, tie_rate

if __name__ == "__main__":
    main()

def thread_main():
    try:
        main()
    except Exception as e:
        print(f"エラーが発生しました: {e}")

def create_gui():
    root = tk.Tk()
    root.title("ポーカー解析ツール")
    root.geometry("400x200")

    label = tk.Label(root,text="ポーカー解析ツールを開始するにはボタンを押してください")
    label.pack(pady=20)

    # ボタン作成
    button - tk.Button(root, text="解析開始",command=lambda: Thread(target=threaded_main).start())
    button.pack(pady=20)
    
    result_label = tk.Label(root,text="結果: 未計算")
    result_label.pack(pady=10)

    root.mainloop()