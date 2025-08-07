from card import Suit, Rank, Card, CardList
from form import CARDS_MAP, Form, Layoff
from extra_point import ExtraPointType, ExtraPoint, ExtraPointValueType
from enum import Enum

class PlayerIndex(Enum):
    DISCARD_PILE = -1
    STOCK_PILE = -2
    PLAYER_0 = 0
    PLAYER_1 = 1
    PLAYER_2 = 2
    PLAYER_3 = 3

class Player:
    id: PlayerIndex
    cards: CardList
    dropped_forms: list[Form]
    layoffs: list[Layoff]
    score: int
    extra_points: list[ExtraPoint]
    knock_card: Card | None
    is_active: bool
    can_show_speto: bool

    def __init__(self, i: PlayerIndex):
        self.id = i
        self.cards = CardList()
        self.dropped_forms = []
        self.layoffs = []
        self.score = 0
        self.extra_points = []
        self.knock_card = None
        self.is_active = False
        self.can_show_speto = True

    def reset(self):
        self.cards.clear()
        self.dropped_forms.clear()
        self.layoffs.clear()
        self.score = 0
        self.extra_points.clear()
        self.knock_card = None
        self.is_active = False
        self.can_show_speto = True

    def __str__(self):
        if self.id.value < 0:
            return f"{self.id.name}: {self.cards}"
        else:
            return f"{self.id.name}: {self.cards}\nscore: {self.score}"

    def __repr__(self):
        return self.__str__()

    def calculate_score(self):
        """Calculate total score from cards in dropped forms and layoffs."""
        self.score = self.calculate_dropped_forms_score() + self.calculate_layoffs_score()
        self.score += self.knock_card.get_value() if self.knock_card else 0

        for extra_point in self.extra_points:
            self.score = extra_point.update_score(self.score)
        return self.score

    def calculate_dropped_forms_score(self) -> int:
        """Calculate score from cards in dropped forms that belong to this player."""
        score = 0
        for form in self.dropped_forms:
            for card in form.cards:
                if card.owner == self.id.value:
                    score += card.get_value()
        return score

    def calculate_layoffs_score(self) -> int:
        """Calculate score from cards in layoffs where form owner is not this player."""
        score = 0
        for layoff in self.layoffs:
            if layoff.form.owner != self.id.value:
                for card in layoff.cards:
                    score += card.get_value()
        return score

    def calculate_hand_cards_score(self) -> int:
        """Calculate score from cards in hand that belong to this player."""
        return sum(card.get_value() for card in self.cards)

    def calculate_end_score(self) -> int:
        self.score = self.calculate_dropped_forms_score() + self.calculate_layoffs_score() - self.calculate_hand_cards_score()
        self.score += self.knock_card.get_value() if self.knock_card else 0
        for extra_point in self.extra_points:
            self.score = extra_point.update_score(self.score)
        return self.score