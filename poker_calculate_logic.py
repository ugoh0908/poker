from treys import Deck, Evaluator, Card

class poker_calculate_logic:
    def estimate_win_probability(self,hole_cards_str, num_opponents, trials):
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