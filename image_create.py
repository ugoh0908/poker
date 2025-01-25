import pygetwindow as gw
import pyautogui


class image_create:
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
