from .card import Suit, Rank, Card, CardList
import itertools


class CardsMap:
    """2D cards map 13(ranks)x4(suits) for easy to found all form in a CardList"""
    map: list[list[bool]] = None

    def __init__(self):
        self.clear()

    def add_card(self, c):
        self.map[c.rank.value - 2][c.suit.value] = True

    def remove_card(self, c):
        self.map[c.rank.value - 2][c.suit.value] = False

    def has_card(self, rank, suit):
        return self.map[rank - 2][suit]

    def clear(self):
        self.map = [[False] * len(Suit) for _ in range(len(Rank))]

    def update(self, cards: set[Card]):
        for c in cards:
            self.add_card(c)

    def get_all_forms(self) -> list[CardList]:
        forms = []
        # check all runs in all rows (consecutive ranks, same suits)
        for suit_index in range(len(Suit)):
            form_count = 0
            for rank_index in range(len(Rank)):
                if self.map[rank_index][suit_index]:
                    form_count += 1
                    if form_count >= 3:
                        for i in range(3, form_count + 1):
                            forms.append(CardList(set([Card(Rank(j + 2), Suit(suit_index)) for j in range(rank_index - i + 1, rank_index + 1)])))
                else: form_count = 0
        # check all sets in all columns (same ranks)
        for rank_index in range(len(Rank)):
            cards_in_rank = []
            for suit_index in range(len(Suit)):
                if self.map[rank_index][suit_index]:
                    cards_in_rank.append(Card(Rank(rank_index + 2), Suit(suit_index)))
            if len(cards_in_rank) >= 3:
                for r in range(3, len(cards_in_rank) + 1):
                    for combo in itertools.combinations(cards_in_rank, r):
                        forms.append(CardList(set(combo)))
        return forms


CARDS_MAP = CardsMap()


class Form(CardList):
    id: int
    owner: int
    is_set: bool
    is_run: bool
    extend_cards: CardList

    def __init__(self, cards: set[Card] = None):
        super().__init__(cards)
        self.id = 0
        self.owner = 0
        self.is_set = False
        self.is_run = False
        self.extend_cards = CardList()
        self._update_form()

    def copy(self):
        form = Form(self.cards.copy())
        form.id = self.id
        form.owner = self.owner
        form.is_set = self.is_set
        form.is_run = self.is_run
        return form

    def add(self, c: Card):
        super().add(c)
        self._update_form()

    def remove(self, c: Card):
        super().remove(c)
        self._update_form()

    def clear(self):
        super().clear()
        self.is_set = False
        self.is_run = False

    def update(self, cards: set[Card]):
        self.cards.clear()
        super().update(cards)
        self._update_form()

    def extend(self, cards: set[Card]):
        super().update(cards)
        self.extend_cards.update(cards)
        self._update_form()

    def can_extend(self, cards: set[Card]) -> bool:
        """Check if the given cards can be extended to form this form."""
        new_form = self.copy()
        new_form.extend(cards)
        if self.is_set:
            return new_form.is_set
        if self.is_run:
            return new_form.is_run
        return False

    def _update_form(self):
        if not self.cards or len(self.cards) < 3:
            self.is_set = False
            self.is_run = False
            return
        is_set, is_run = Form.is_set_or_run_form(self.cards)
        self.is_set = is_set
        self.is_run = is_run

    @staticmethod
    def is_valid_form(cards: set[Card]) -> bool:
        is_set, is_run = Form.is_set_or_run_form(cards)
        return is_set or is_run

    @staticmethod
    def is_set_or_run_form(cards: set[Card]) -> (bool, bool):
        """Check if the given cards form a valid set or run.

        Args:
            cards (set[Card]): A set of cards to check

        Returns:
            tuple(bool, bool): A tuple of (is_set, is_run) where:
                - is_set: True if cards form a set (same rank)
                - is_run: True if cards form a run (consecutive ranks, same suit)
        """
        if not cards or len(cards) < 3:
            return False, False
        ranks = [c.rank.value for c in cards]
        rank_set = set(ranks)
        suits = [c.suit.value for c in cards]
        suit_set = set(suits)
        is_set = len(rank_set) == 1
        is_run = len(suit_set) == 1 and len(rank_set) == len(ranks) and max(rank_set) - min(rank_set) + 1 == len(rank_set)
        return is_set, is_run


class Layoff:
    form: Form
    cards: CardList

    def __init__(self, cards: CardList, form: Form):
        self.form = form
        self.cards = cards

    def __str__(self):
        return f"{self.cards} -> {self.form}"

    def __repr__(self):
        return self.__str__()

    def can_layoff(self) -> bool:
        """Check if the form can be layed off from this layoff."""
        return self.form.can_extend(self.cards.cards)

    def do_layoff(self):
        """Lay off the form from this layoff."""
        self.form.extend(self.cards.cards)
