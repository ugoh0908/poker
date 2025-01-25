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
from image_create import image_create
from Const.mainconst import mainconst
from Logic.change_card_name import change_card_name
from Thread.main_thread import main_thread


class main_gui:
    def create_gui():
        result_label = None

        root = tk.Tk()
        root.title("ポーカー解析ツール")
        root.geometry("400x200")

        label = tk.Label(
            root, text="ポーカー解析ツールを開始するにはボタンを押してください"
        )
        label.pack(pady=20)

        # ボタン作成
        instance_main_thread = main_thread()
        button = tk.Button(
            root,
            text="解析開始",
            command=lambda: Thread(
                target=instance_main_thread.main_thread(result_label)
            ).start(),
        )
        button.pack(pady=20)

        result_label = tk.Label(root, text="結果: 未計算")
        result_label.pack(pady=10)

        root.mainloop()
        # GUI作成はここまで

        instance_image_create = image_create()
        screenshot = image_create.get_screenshot(mainconst.window_title)
