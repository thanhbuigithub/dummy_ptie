from enum import Enum, auto

from .card import Card, CardList
from .form import Form, Layoff

class ActionType(Enum):
    """Represents the types of actions that can be taken in the game."""
    INIT = auto()        # Represents initial game state
    DRAW = auto()        # Draw a card from the stock pile
    PICK = auto()        # Pick a card from the discard pile
    MELD = auto()        # Create a meld and place it on the table
    LAYOFF = auto()      # Add one or more cards to an existing meld
    DISCARD = auto()     # Discard a card to the discard pile
    SHOW_SPETO = auto()  # Show Q♠ and 2♣ to get extra points
    KNOCK = auto()       # Go out by discarding the last card

class Action:
    player_id: int
    action_type: ActionType

    def __init__(self, player_id: int, action_type: ActionType):
        self.player_id = player_id
        self.action_type = action_type

    def __str__(self):
        return f"{self.player_id} {self.action_type}"

class InitAction(Action):

    def __init__(self):
        super().__init__(-1, ActionType.INIT)

    def __str__(self):
        return f"{self.action_type}"

class DrawAction(Action):

    def __init__(self, player_id: int):
        super().__init__(player_id, ActionType.DRAW)

    def __str__(self):
        return f"{self.player_id} {self.action_type}"

class PickAction(Action):
    card: Card
    form: Form
    is_pick_last: bool
    owner_pick_last: int
    got_cards: CardList

    def __init__(self, player_id: int, card: Card, card_list: CardList):
        super().__init__(player_id, ActionType.PICK)
        self.card = card
        self.form = Form(card_list.cards)
        self.is_pick_last = False
        self.owner_pick_last = -1
        self.got_cards = CardList()

    def __str__(self):
        return f"{self.player_id} {self.action_type} | pick: {self.card} -> form: {self.form}"

class MeldAction(Action):
    form: Form

    def __init__(self, player_id: int, card_list: CardList):
        super().__init__(player_id, ActionType.MELD)
        self.form = Form(card_list.cards)
        self.form.owner = player_id

    def __str__(self):
        return f"{self.player_id} {self.action_type} | form: {self.form}"

class LayoffAction(Action):
    layoff: Layoff

    def __init__(self, player_id: int, card_list: CardList, form: Form):
        super().__init__(player_id, ActionType.LAYOFF)
        self.layoff = Layoff(card_list, form)

    def __str__(self):
        return f"{self.player_id} {self.action_type} | layoff: {self.layoff}"

class DiscardAction(Action):
    card: Card
    is_stupid: bool

    def __init__(self, player_id: int, card: Card):
        super().__init__(player_id, ActionType.DISCARD)
        self.card = card

    def __str__(self):
        return f"{self.player_id} {self.action_type} | {self.card}"

class ShowSpetoAction(Action):
    def __init__(self, player_id: int):
        super().__init__(player_id, ActionType.SHOW_SPETO)

    def __str__(self):
        return f"{self.player_id} {self.action_type}"

class KnockAction(Action):
    card: Card

    def __init__(self, player_id: int, card: Card):
        super().__init__(player_id, ActionType.KNOCK)
        self.card = card

    def __str__(self):
        return f"{self.player_id} {self.action_type} | {self.card}"