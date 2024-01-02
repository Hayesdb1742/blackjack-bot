class BlackjackHand:
    def __init__(self):
        self.cards = []
        self.value = 0
        self.is_bust = False
        self.is_blackjack = False

    def add_card(self, card):
        self.cards.append(card)
        self.calculate_value()
        if self.value == 21 and len(self.cards) == 2:
            self.is_blackjack = True
        if self.value > 21:
            self.is_bust = True

    def calculate_value(self):
        self.value = 0
        num_aces = 0

        for card in self.cards:
            self.value += card.get_value()

            # Count Aces separately to handle their special value
            if card.rank == 'Ace':
                num_aces += 1

        # Adjust the value of Aces to minimize the risk of busting
        while num_aces > 0 and self.value > 21:
            self.value -= 10
            num_aces -= 1

    def is_soft(self):
        return 'Ace' in [card.rank for card in self.cards] and self.value < 21

    def is_pair(self):
        return len(self.cards) == 2 and self.cards[0].rank == self.cards[1].rank

    def is_busted(self):
        return self.is_bust

    def clear_hand(self):
        self.cards = []
        self.value = 0
        self.is_bust = False
        self.is_blackjack = False

    def __str__(self):
        return ', '.join([str(card) for card in self.cards])

    def get_cards(self):
        return self.cards

    def get_value(self):
        return self.value

    def split_hand(self):
        if self.is_pair():
            split_card = self.cards.pop(1)
            new_hand = BlackjackHand()
            new_hand.add_card(split_card)
            return new_hand

    def double_down(self, card):
        self.add_card(card)


class Card:
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit

    def get_value(self):
        if self.rank in ['Jack', 'Queen', 'King']:
            return 10
        elif self.rank == 'Ace':
            return 11
        else:
            return int(self.rank)

    def __str__(self):
        return f"{self.rank} of {self.suit}"


# Example usage:
hand = BlackjackHand()
card1 = Card('Ace', 'Spades')
card2 = Card('10', 'Hearts')
card3 = Card('6', 'Diamonds')

hand.add_card(card1)
hand.add_card(card2)
hand.add_card(card3)

print("Hand:", hand)
print("Total Value:", hand.get_value())
print("Is Soft:", hand.is_soft())
print("Is Pair:", hand.is_pair())
print("Is Busted:", hand.is_busted())
print("Is Blackjack:", hand.is_blackjack)

# Example of splitting a pair
if hand.is_pair():
    new_hand = hand.split_hand()
    print("Original Hand after Split:", hand)
    print("New Hand after Split:", new_hand)

    

    



    
