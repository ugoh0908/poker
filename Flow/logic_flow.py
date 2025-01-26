import pygetwindow as gw
from Logic.trimming_image import trimming_image
from Const.mainconst import mainconst
from poker_calculate_logic import poker_calculate_logic
import pyautogui
from torchvision import models
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from Logic.change_card_name import change_card_name

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


class logic_flow:

    def get_screenshot(slef):
        # 指定したウィンドウを探す
        windows = gw.getWindowsWithTitle(mainconst.window_title)
        if not windows:
            print("指定されたウィンドウが見つかりません。")
            return None
        window = windows[0]

        # ウィンドウの位置とサイズを取得
        x, y, width, height = window.left, window.top, window.width, window.height

        # スクリーンショットを撮影
        screenshot = pyautogui.screenshot(region=(x, y, width, height))
        return screenshot

    def logic_flow(self):
        screenshot = self.get_screenshot()

        from main import main

        if screenshot:
            instance_trimming_image = trimming_image()
            instance_change_card_name = change_card_name()
            card_text_1st = instance_change_card_name.change_card_name(
                instance_trimming_image.extract_card_text(
                    screenshot,
                    mainconst.card_region_1st,
                    test_transform,
                    model,
                    mainconst.class_names,
                    mainconst.screenshot_save_path,
                )
            )
            card_text_2nd = instance_change_card_name.change_card_name(
                instance_trimming_image.extract_card_text(
                    screenshot,
                    mainconst.card_region_2nd,
                    test_transform,
                    model,
                    mainconst.class_names,
                    mainconst.screenshot_save_path,
                )
            )

            print("現在のカード:1枚目[", card_text_1st, "]、2枚目[", card_text_2nd, "]")

            if card_text_1st != card_text_2nd:
                hole_cards = [card_text_1st, card_text_2nd]
                num_opponents = 1
                trials = 10000

                instance_poker_calculate_logic = poker_calculate_logic()

                win_rate, tie_rate = (
                    instance_poker_calculate_logic.estimate_win_probability(
                        hole_cards, num_opponents, trials
                    )
                )
                print(
                    f"ホールカード: {hole_cards}, 相手人数: {num_opponents}, 試行回数: {trials}"
                )
                print(f"推定勝率: {win_rate:.2f}%")
                print(f"推定引き分け率: {tie_rate:.2f}%")

                return f"ホールカード: {hole_cards}, 勝率: {win_rate:.2f}%, 引き分け率: {tie_rate:.2f}%"
