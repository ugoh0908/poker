import pygetwindow as gw
import pyautogui
from PIL import Image
import pytesseract
import time

# Tesseractのパスを設定（インストール済みであることが前提）
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

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
    # OCRでカード情報を読み取る
    card_text = pytesseract.image_to_string(cropped_image, lang='eng')
    return card_text.strip()

    
def show_cropped_image(image, region):
    # 指定した領域をトリミング
    cropped_image = image.crop(region)
    cropped_image.show()  # 切り取った画像を表示

def main():
    window_title = "MEmu"  # エミュレータのウィンドウタイトル
    card_region =  (260, 975, 500, 1125)   # カードが表示されるエリアを調整

    while True:
        screenshot = get_screenshot(window_title)
        if screenshot:
            
            card_text = extract_card_text(screenshot, card_region)
            show_cropped_image(screenshot, card_region)
            if card_text:
                
                print("現在のカード:", card_text)
        time.sleep(51)  # 1秒ごとに更新

if __name__ == "__main__":
    main()
