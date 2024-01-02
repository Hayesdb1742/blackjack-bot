from hands import Card, BlackjackHand


def decide_hit_or_stay(player_hand, dealer_upcard):
    player_value = player_hand.get_value()

    if player_hand.is_pair():
        # Pair strategy
        if player_hand.cards[0].rank in ['Ace', '8']:
            return "split"
        elif player_hand.cards[0].rank == '10':
            return "stay"

    if player_hand.is_soft():
        # Soft hand strategy
        if player_value == 13 and dealer_upcard in [5, 6]:
            return "double_down"
        elif player_value == 14 and dealer_upcard in [4, 5, 6]:
            return "double_down"
        elif player_value == 15 and dealer_upcard in [4, 5, 6]:
            return "double_down"
        elif player_value == 16 and dealer_upcard in [4, 5, 6]:
            return "double_down"
        elif player_value == 17 and dealer_upcard in [3, 4, 5, 6]:
            return "double_down"
        elif player_value == 18 and dealer_upcard in [2, 7, 8]:
            return "stay"
        elif player_value == 19 or player_value == 20:
            return "stay"

    # Hard hand strategy
    if player_value <= 8 or player_value >= 17:
        return "stay"
    elif 9 <= player_value <= 11 and dealer_upcard in [2, 3, 4, 5, 6]:
        return "double_down"
    else:
        return "hit"

# Example usage:
player_hand = BlackjackHand()
dealer_upcard = 6
player_hand.add_card(Card('6', 'Hearts'))
player_hand.add_card(Card('6', 'Diamonds'))

decision = decide_hit_or_stay(player_hand, dealer_upcard)
print("Decision:", decision)
