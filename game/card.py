from enum import Enum

class Const:
    CARD_NUM = 52
    MIN_CARD_ID = 2
    MAX_CARD_ID = 53

class Suit(Enum):
    SPADE = 0
    CLUB = 1
    DIAMOND = 2
    HEART = 3

class Rank(Enum):
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14

class Card:
    is_head: bool
    suit: Suit
    rank: Rank
    id: int
    owner: int

    def __init__(self, rank: Rank, suit: Suit):
        self.suit = suit
        self.rank = rank
        self.id = self.get_card_id()
        self.is_head = False
        self.owner = -1

    def __str__(self):
        return f"{self.rank.name} of {self.suit.name}"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if isinstance(other, Card):
            return self.rank == other.rank and self.suit == other.suit
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if isinstance(other, Card):
            return self.id < other.id
        return False

    def __le__(self, other):
        if isinstance(other, Card):
            return self.id <= other.id
        return False

    def __gt__(self, other):
        if isinstance(other, Card):
            return self.id > other.id
        return False

    def __ge__(self, other):
        if isinstance(other, Card):
            return self.id >= other.id
        return False

    def __hash__(self):
        return hash((self.rank, self.suit))

    def get_card_id(self) -> int:
        """Returns the card index (13 * Suit + Rank)."""
        return 13 * self.suit.value + self.rank.value

    def get_value(self) -> int:
        """Returns the point value of the card according to Dummy game rules."""
        if self.rank == Rank.QUEEN and self.suit == Suit.SPADE:
            return 50
        elif self.rank == Rank.TWO and self.suit == Suit.CLUB:
            return 50
        elif self.rank == Rank.ACE:
            return 15
        elif self.rank in [Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING]:
            return 10
        else:  # 2-9
            return 5
    
    def update_rank(self, new_rank: Rank):
        """Updates the rank of the card."""
        self.rank = new_rank
        self.id = self.get_card_id()
    
    def update_suit(self, new_suit: Suit):
        """Updates the suit of the card."""
        self.suit = new_suit
        self.id = self.get_card_id()

    def is_speto(self) -> bool:
        """Checks if the card is a speto card."""
        return (self.rank == Rank.TWO and self.suit == Suit.CLUB) or (self.rank == Rank.QUEEN and self.suit == Suit.SPADE)
        
    @staticmethod
    def get_card_from_id(card_id: int) -> 'Card':
        """Returns the card corresponding to the given card ID."""
        suit = Suit((card_id - 2) // 13)
        temp_rank = card_id % 13
        if temp_rank == 0:
            temp_rank = 13
        elif temp_rank == 1:
            temp_rank = 14
        rank = Rank(temp_rank)
        return Card(rank, suit)

    @staticmethod
    def is_same_rank(card1: 'Card', card2: 'Card') -> bool:
        """Checks if two cards have the same rank."""
        return card1.rank == card2.rank

    @staticmethod
    def is_same_suit(card1: 'Card', card2: 'Card') -> bool:
        """Checks if two cards have the same suit."""
        return card1.suit == card2.suit

    @staticmethod
    def is_same_card(card1: 'Card', card2: 'Card') -> bool:
        """Checks if two cards are the same."""
        return card1.id == card2.id

    @staticmethod
    def is_valid_card_id(card_id: int) -> bool:
        """Checks if the given card ID is valid."""
        return Const.MIN_CARD_ID <= card_id <= Const.MAX_CARD_ID

    @staticmethod
    def is_valid_card(card: 'Card') -> bool:
        """Checks if the given card is valid."""
        return Card.is_valid_card_id(card.id)


class CardList:
    cards: set[Card]

    def __init__(self, cards: set[Card] = None):
        if cards is None:
            cards = set()
        self.cards = cards

    def __str__(self):
        return f"{self.cards}"

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.cards)

    def __iter__(self):
        return iter(self.cards)

    def __contains__(self, item):
        return item in self.cards

    def add(self, card: Card):
        self.cards.add(card)

    def remove(self, card: Card):
        self.cards.remove(card)

    def clear(self):
        self.cards.clear()

    def copy(self):
        return CardList(self.cards.copy())

    def update(self, cards: set[Card]):
        self.cards.update(cards)

    def has_card(self, card: Card):
        return card in self.cards

    def has_two_club(self):
        return self.has_card(Card(Rank.TWO, Suit.CLUB))

    def has_queen_spade(self):
        return self.has_card(Card(Rank.QUEEN, Suit.SPADE))

    def has_speto(self):
        return self.has_two_club() or self.has_queen_spade()