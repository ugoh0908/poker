from Flow.logic_flow import logic_flow


class main_thread:
    def main_thread(self, result_label):
        try:
            instance_logic_flow = logic_flow()
            result = instance_logic_flow.logic_flow()
            result_label.after(0, lambda: result_label.config(text=f"結果: {result}"))
        except Exception as e:
            result_label.after(0, lambda e=e: result_label.config(text=f"エラー: {e}"))
            print(e)
