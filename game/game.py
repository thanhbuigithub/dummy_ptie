import numpy as np
import random
import copy
from enum import Enum, auto
from .card import Card, CardList, Suit, Rank
from .player import Player, PlayerIndex
from .form import Form, CARDS_MAP, Layoff
from .extra_point import ExtraPoint, ExtraPointType
from .action import Action, ActionType, InitAction, DrawAction, PickAction, DiscardAction, LayoffAction, MeldAction, KnockAction, ShowSpetoAction


class GamePhase(Enum):
    """Represents the current phase of a player's turn."""
    PICK_OR_DRAW = auto()
    DISCARD_MELD_LAYOFF = auto()

class GameState(Enum):
    """Represents the overall state of the game."""
    NOT_START = auto()
    STARTED = auto()
    FINISHED = auto()

class TurnDirection(Enum):
    CLOCKWISE = auto()
    COUNTER_CLOCKWISE = auto()


BASE_CARDS_LIST: list[Card] = [Card.get_card_from_id(card) for card in range(2, 54)]

class DummyGame:
    id: str
    chairs: list[Player]
    players: list[Player]
    discard_pile: list[Card]
    stock_pile: list[Card]
    head_card: Card | None = None
    game_state: GameState
    phase: GamePhase
    turn_direction: TurnDirection
    start_player_index: int
    current_turn: int
    current_player: Player | None
    forms: list[Form]
    turn_actions: list[Action]
    can_dark_knock: list[bool]

    def __init__(self, id: str):
        self.id = id
        self.chairs = [Player(PlayerIndex.PLAYER_0), Player(PlayerIndex.PLAYER_1), Player(PlayerIndex.PLAYER_2), Player(PlayerIndex.PLAYER_3)]
        self.players = []
        self.discard_pile = []
        self.stock_pile = []
        self.head_card = None
        self.game_state = GameState.NOT_START
        self.phase = GamePhase.PICK_OR_DRAW
        self.turn_direction = TurnDirection.CLOCKWISE
        self.start_player_index = -1
        self.current_turn = 0
        self.current_player = None
        self.forms = []
        self.turn_actions = []
        self.can_dark_knock = []

    def reset(self):
        for chair in self.chairs:
            chair.reset()

        self.players = []
        self.discard_pile = []
        self.stock_pile = []
        self.head_card = None
        self.game_state = GameState.NOT_START
        self.phase = GamePhase.PICK_OR_DRAW
        self.turn_direction = TurnDirection.CLOCKWISE
        self.start_player_index = -1
        self.current_turn = 0
        self.current_player = None
        self.forms = []
        self.turn_actions = []
        self.can_dark_knock = []
        
    def start_game(self, turn_direction: TurnDirection, active_players=None, start_chair_index=None):
        if active_players is None:
            active_players = [True, True, True, True]
        if len(active_players) != 4:
            raise ValueError("Invalid number of active players. Must be 4. Example: [True, False, True, False]")
        n_players = sum(1 for player in active_players if player)
        if n_players < 2 or n_players > 4:
            raise ValueError("Number of players must be between 2 and 4")
        for i, player in enumerate(self.chairs):
            player.is_active = active_players[i]
            if active_players[i]:
                self.players.append(player)
        if start_chair_index is None:
            start_chair_index = active_players.index(True)
        if start_chair_index < 0 or start_chair_index > 3 or active_players[start_chair_index] == False:
            raise ValueError("Invalid start player index")
        if turn_direction not in [TurnDirection.CLOCKWISE, TurnDirection.COUNTER_CLOCKWISE]:
            raise ValueError("Invalid turn direction")

        self.game_state = GameState.STARTED
        self.turn_direction = turn_direction
        self.start_player_index = self.players.index(self.chairs[start_chair_index])

    def deal_cards(self):
        if self.game_state != GameState.STARTED:
            raise ValueError("Game is not in progress")

        cards = BASE_CARDS_LIST.copy()
        random.shuffle(cards)

        n_players = len(self.players)
        cards_per_player = 11 if n_players == 2 else (9 if n_players == 3 else 7)

        for player in self.players:
            player.cards.update(cards[:cards_per_player])
            for card in player.cards:
                card.owner = player.id.value
            cards = cards[cards_per_player:]

        for card in cards:
            card.owner = PlayerIndex.STOCK_PILE.value
        self.stock_pile = cards
        self.head_card = self.stock_pile.pop()
        self.discard_pile = [self.head_card]
        self.head_card.owner = PlayerIndex.DISCARD_PILE.value

    def save_initial_state(self):
        self._save_current_state(InitAction())

    def next_turn(self):
        if self.game_state != GameState.STARTED:
            raise ValueError("Game is not in progress")
        if len(self.stock_pile) <= 0:
            self.end_game()
            return
        n_players = len(self.players)
        self.current_turn += 1
        index_delta = 1 if self.turn_direction == TurnDirection.CLOCKWISE else -1
        self.current_player = self.players[(self.current_turn - 1 + self.start_player_index) * index_delta % n_players]
        self.turn_actions = []
        self.can_dark_knock = [len(player.dropped_forms) <= 0 for player in self.players]
        self.next_pick_or_draw_phase()

    def next_pick_or_draw_phase(self):
        if self.game_state != GameState.STARTED:
            raise ValueError("Game is not in progress")
        self.phase = GamePhase.PICK_OR_DRAW
        
    def next_discard_meld_layoff_phase(self):
        if self.game_state != GameState.STARTED:
            raise ValueError("Game is not in progress")
        self.phase = GamePhase.DISCARD_MELD_LAYOFF

    def end_game(self):
        self.game_state = GameState.FINISHED
        for player in self.players:
            if (len(player.dropped_forms) <= 0):
                player.extra_points.append(ExtraPoint(ExtraPointType.BURN_STUPID, player.id.value))
            player.calculate_end_score()

    def take_action(self, action: Action):
        if self.game_state != GameState.STARTED:
            raise ValueError("Game is not in progress")
        player = self.get_player_by_id(action.player_id)
        if not player:
            raise ValueError("Invalid player ID")
        
        match action.action_type:
            case ActionType.SHOW_SPETO:
                self.player_show_speto(player, action)
            case ActionType.DRAW:
                self.player_draw(player, action)
            case ActionType.PICK:
                self.player_pick(player, action)
            case ActionType.DISCARD:
                self.player_discard(player, action)
            case ActionType.LAYOFF:
                self.player_layoff(player, action)
            case ActionType.MELD:
                self.player_meld(player, action)
            case ActionType.KNOCK:
                self.player_knock(player, action)
            case _:
                raise ValueError("Invalid action type")
        self.turn_actions.append(action)
        for player in self.players:
            player.calculate_score()
        match action.action_type:
            case ActionType.DRAW | ActionType.PICK:
                self.next_discard_meld_layoff_phase()
            case ActionType.DISCARD:
                self.next_turn()
            case ActionType.KNOCK:
                self.end_game()
        
        # Save current state before applying the action
        self._save_current_state(action)

    def _save_current_state(self, action: Action):
        if 'game_history_manager' in globals():
            game_history_manager.save_game_history(self.id, action)

    def restore_to_state(self, history_index: int):
        if 'game_history_manager' in globals():
            game_history_manager.restore_game_history(self.id, history_index)

    def player_show_speto(self, player: Player, action: ShowSpetoAction):
        player.extra_points.append(ExtraPoint(ExtraPointType.SHOW_SPETO, player.id.value))
        player.can_show_speto = False

    def player_pick(self, player: Player, action: PickAction):
        pick_index = self.discard_pile.index(action.card)
        if pick_index < 0:
            raise ValueError(f"Card {Card} not in discard pile {self.discard_pile}")
        if pick_index == len(self.discard_pile) - 1:
            action.is_pick_last = True
            action.owner_pick_last = self.discard_pile[pick_index].owner
        pick_cards = [card for card in self.discard_pile[pick_index:] if card not in action.form.cards]
        self.discard_pile = self.discard_pile[:pick_index]
        action.form.id = len(self.forms)
        action.form.owner = player.id.value
        player.dropped_forms.append(action.form)
        self.forms.append(action.form)
        for card in action.form.cards:
            if player.cards.has_card(card):
                player.cards.remove(card)
        player.cards.update(set(pick_cards))
        action.got_cards.clear()
        action.got_cards.update(set(pick_cards))
        # check pick head card
        if self.head_card and self.head_card == action.card:
            player.extra_points.append(ExtraPoint(ExtraPointType.PICK_HEAD, player.id.value))
        # check stupid
        if action.form.has_card(self.head_card):
            stupid_player_ids = []
            for card in action.form.cards:
                if card.owner != player.id.value and card.owner != PlayerIndex.DISCARD_PILE.value and card.owner != PlayerIndex.STOCK_PILE.value:
                    stupid_player_ids.append(card.owner)
            if len(stupid_player_ids) > 0:
                player_id = stupid_player_ids.pop()
                stupid_player = self.get_player_by_id(player_id)
                if stupid_player:
                    stupid_player.extra_points.append(ExtraPoint(ExtraPointType.TAUNTING_STUPID, player_id))
        if action.form.has_speto():
            stupid_player_ids = []
            for card in action.form.cards:
                if card.owner != player.id.value and card.owner != PlayerIndex.DISCARD_PILE.value and card.owner != PlayerIndex.STOCK_PILE.value:
                    stupid_player_ids.append(card.owner)
            if len(stupid_player_ids) > 0:
                player_id = stupid_player_ids.pop()
                stupid_player = self.get_player_by_id(player_id)
                if stupid_player:
                    stupid_player.extra_points.append(ExtraPoint(ExtraPointType.TAUNTING_STUPID, player_id))
        # update all cards owner
        for card in pick_cards:
            card.owner = player.id.value
        for card in action.form.cards:
            card.owner = player.id.value

    def player_draw(self, player: Player, action: DrawAction):
        if len(self.stock_pile) == 0:
            raise ValueError("No cards in stock pile")
        card = self.stock_pile.pop()
        player.cards.add(card)
        card.owner = player.id.value

    def player_discard(self, player: Player, action: DiscardAction):
        action.card.owner = player.id.value
        self.discard_pile.append(action.card)
        player.cards.remove(action.card)
        for card in self.discard_pile[:-len(self.players)]:
            card.owner = PlayerIndex.DISCARD_PILE.value
        # check stupid
        CARDS_MAP.clear()
        CARDS_MAP.update(self.discard_pile)
        all_discard_pile_forms = CARDS_MAP.get_all_forms()
        is_dummy_discard = False
        if len(all_discard_pile_forms) > 0:
            for form in all_discard_pile_forms:
                if form.has_card(action.card):
                    is_dummy_discard = True
        for form in self.forms:
            if form.can_extend({action.card}):
                is_dummy_discard = True
        if is_dummy_discard:
            action.is_stupid = True
            player.extra_points.append(ExtraPoint(ExtraPointType.DUMMY_STUPID, player.id.value))

    def player_layoff(self, player: Player, action: LayoffAction):
        for card in action.layoff.cards:
            player.cards.remove(card)
            card.owner = player.id.value
        action.layoff.form.extend(action.layoff.cards.cards)
        layoff_score = sum(card.get_value() for card in action.layoff.cards)

        # check stupid
        form_owner = self.get_player_by_id(action.layoff.form.owner)
        if action.layoff.cards.has_speto():
            if action.layoff.form.is_set and action.layoff.form.owner != player.id.value:
                form_owner.extra_points.append(ExtraPoint(ExtraPointType.BE_LAYOFF_SPETO, form_owner.id.value))
            elif action.layoff.form.is_run:
                form_cards = sorted(action.layoff.form.cards)
                form_cards_reverse = sorted(action.layoff.form.cards, reverse=True)
                if action.layoff.form.has_card(Card(Rank.TWO, Suit.CLUB)):
                    form_origin_cards = [card for card in form_cards if card not in action.layoff.form.extend_cards]
                    max_card_rank = max(card.rank.value for card in form_origin_cards)
                    for card in form_cards:
                        if card.rank.value <= max_card_rank and card.owner != player.id.value and card.owner >= 0:
                            card_owner = self.get_player_by_id(card.owner)
                            if card_owner:
                                card_owner.extra_points.append(
                                    ExtraPoint(ExtraPointType.BE_LAYOFF_SPETO, card_owner.id.value))
                                break
                if action.layoff.form.has_card(Card(Rank.QUEEN, Suit.SPADE)):
                    form_origin_cards = [card for card in form_cards_reverse if card not in action.layoff.form.extend_cards]
                    min_card_rank = min(card.rank.value for card in form_origin_cards)
                    for card in form_cards_reverse:
                        if card.rank.value >= min_card_rank and card.owner != player.id.value and card.owner >= 0:
                            card_owner = self.get_player_by_id(card.owner)
                            if card_owner:
                                card_owner.extra_points.append(
                                    ExtraPoint(ExtraPointType.BE_LAYOFF_SPETO, card_owner.id.value))
                                break

    def player_meld(self, player: Player, action: MeldAction):
        for card in action.form.cards:
            player.cards.remove(card)
        action.form.id = len(self.forms)
        action.form.owner = player.id.value
        for card in action.form.cards:
            card.owner = player.id.value
        player.dropped_forms.append(action.form)
        self.forms.append(action.form)

    def player_knock(self, player: Player, action: KnockAction):
        player.knock_card = action.card
        action.card.owner = player.id.value
        self.discard_pile.append(action.card)
        player.extra_points.append(ExtraPoint(ExtraPointType.NORMAL_KNOCK, player.id.value))
        is_dark_knock = self.can_dark_knock[self.players.index(player)]
        is_suit_knock = False
        if len(player.dropped_forms) > 0:
            first_suit = next(iter(player.dropped_forms[0].cards)).suit
            is_suit_knock = all(
                form.is_run and all(card.suit == first_suit for card in form.cards)
                for form in player.dropped_forms
            )
        is_dummy_knock = False
        for form in self.forms:
            if form.can_extend({action.card}):
                is_dummy_knock = True
        if is_dummy_knock:
            player.extra_points.append(ExtraPoint(ExtraPointType.DUMMY_KNOCK, player.id.value))
        if is_dark_knock:
            player.extra_points.append(ExtraPoint(ExtraPointType.DARK_KNOCK, player.id.value))
        if is_suit_knock:
            player.extra_points.append(ExtraPoint(ExtraPointType.SUIT_KNOCK, player.id.value))
        player.cards.remove(action.card)
        
        # Check stupid
        for action in self.turn_actions:
            if action.action_type == ActionType.PICK:
                pick_action: PickAction = action
                if pick_action.is_pick_last and pick_action.owner_pick_last != player.id.value and pick_action.owner_pick_last != PlayerIndex.DISCARD_PILE.value and pick_action.owner_pick_last != PlayerIndex.STOCK_PILE.value:
                    stupid_player = self.get_player_by_id(pick_action.owner_pick_last)
                    stupid_player.extra_points.append(ExtraPoint(ExtraPointType.STUPID_STUPID, stupid_player.id.value))

    def get_player_by_id(self, player_id: int):
        for player in self.players:
            if player.id.value == player_id:
                return player
        return None

    def get_all_possible_action_for_player(self, player: Player):
        if self.game_state != GameState.STARTED:
            return []
        action = []
        if self.phase == GamePhase.PICK_OR_DRAW:
            # Check SHOW_SPETO action
            if player.can_show_speto and self.current_turn <= len(self.players) and player.cards.has_two_club() and player.cards.has_queen_spade():
                action.append(ShowSpetoAction(player.id.value))

            # Add DRAW action
            action.append(DrawAction(player.id.value))

            # Check PICK action
            CARDS_MAP.clear()
            cards_array = self.discard_pile + list(player.cards)
            CARDS_MAP.update(cards_array)
            all_forms = CARDS_MAP.get_all_forms()
            for form in all_forms:
                # Kiểm tra xem lá bài có trong discard_pile không
                discard_pile_index = []
                for card in form:
                    index = self.discard_pile.index(card) if card in self.discard_pile else -1
                    discard_pile_index.append(index)
                
                discard_pile_card_index = [index for index in discard_pile_index if index >= 0]
                min_discard_pile_card_index = min(discard_pile_card_index) if len(discard_pile_card_index) > 0 else -1
                num_player_card = sum(1 for index in discard_pile_index if index < 0)
                if min_discard_pile_card_index >= 0 and num_player_card >= 1:
                    num_card_got_picked = len(form) - min_discard_pile_card_index
                    if num_card_got_picked + len(player.cards) > len(form):
                        action.append(PickAction(player.id.value, self.discard_pile[min_discard_pile_card_index], form))

        if self.phase == GamePhase.DISCARD_MELD_LAYOFF:
            # Check knock action:
            if len(player.cards) == 1:
                action.append(KnockAction(player.id.value, next(iter(player.cards.cards))))

            # Add DISCARD action
            if len(player.cards) > 1:
                for card in player.cards:
                    action.append(DiscardAction(player.id.value, card))

            # Add MELD action
            if len(player.cards) > 3 and len(player.dropped_forms) > 0:
                CARDS_MAP.clear()
                CARDS_MAP.update(player.cards)
                all_forms_can_meld = CARDS_MAP.get_all_forms()
                for form in all_forms_can_meld:
                    if len(form) < len(player.cards):
                        action.append(MeldAction(player.id.value, form))

            # Add LAYOFF action
            if len(player.cards) > 1 and len(player.dropped_forms) > 0:
                for card in player.cards:
                    for form in self.forms:
                        if form.can_extend({card}):
                            action.append(LayoffAction(player.id.value, CardList({card}), form))

        return action

class DummyGameHistory:
    game: DummyGame
    game_id: str
    action_history: list[Action]
    state_history: list[dict]
    current_history_index: int

    def __init__(self, game_id: str, game: DummyGame):
        self.game_id = game_id
        self.game = game
        self.action_history = []
        self.state_history = []
        self.current_history_index = -1

    def _save_current_state(self, action: Action):
        """Save the current game state to history before applying an action."""
        # Create a deep copy of the current state
        state = copy.deepcopy(self.game)
        
        # When saving a new action and we're not at the end of the history,
        # truncate the history at current position before adding
        if self.current_history_index < len(self.action_history) - 1:
            self.action_history = self.action_history[:self.current_history_index + 1]
            self.state_history = self.state_history[:self.current_history_index + 1]
        
        self.action_history.append(action)
        self.state_history.append(state)
        self.current_history_index = len(self.action_history) - 1

    def restore_to_state(self, history_index: int):
        """Restore the game to a previous state in the history."""
        if history_index < 0 or history_index >= len(self.state_history):
            raise ValueError(f"Invalid history index: {history_index}")
        
        state = self.state_history[history_index]
        self.game.__dict__ = copy.deepcopy(state.__dict__)
        
        # Update the current history index
        self.current_history_index = history_index
        print(f"current_history_index: {self.current_history_index}")
        print(f"action_history: {len(self.action_history)}")

class DummyGameHistoryManager:
    game: dict[str, DummyGameHistory]

    def __init__(self):
        self.game = {}

    def add_game(self, game_id: str, game: DummyGame):
        self.game[game_id] = DummyGameHistory(game_id, game)

    def set_game_history(self, game_id: str, game_history: DummyGameHistory):
        self.game[game_id] = game_history

    def get_game_history(self, game_id: str):
        if game_id not in self.game:
            return None
        return self.game[game_id]

    def save_game_history(self, game_id: str, action: Action):
        game_history = self.get_game_history(game_id)
        if game_history:
            game_history._save_current_state(action)

    def restore_game_history(self, game_id: str, history_index: int):
        game_history = self.get_game_history(game_id)
        if game_history:
            game_history.restore_to_state(history_index)

    def get_current_history_index(self, game_id: str):
        game_history = self.get_game_history(game_id)
        if game_history:
            return game_history.current_history_index
        return -1
    
    def get_action_history(self, game_id: str):
        game_history = self.get_game_history(game_id)
        if game_history:
            return game_history.action_history
        return []
    
    def get_state_history(self, game_id: str):
        game_history = self.get_game_history(game_id)
        if game_history:
            return game_history.state_history
        return []
    
    def get_game(self, game_id: str):
        game_history = self.get_game_history(game_id)
        if game_history:
            return game_history.game
        return None

game_history_manager = DummyGameHistoryManager()

class GameScenario:
    id: str
    name: str
    description: str
    setup_function: callable

    def __init__(self, id: str, name: str, description: str, setup_function: callable):
        self.id = id
        self.name = name
        self.description = description
        self.setup_function = setup_function


# Define some example scenarios
def setup_speto_scenario(game: DummyGame):
    # Reset the game to a clean state
    game.reset()
    
    # Start a game with 4 players in clockwise direction
    game.start_game(TurnDirection.CLOCKWISE, [True, True, True, True], 0)
    
    # Setup specific cards for players to demonstrate SPETO
    player0 = game.players[0]
    player0.cards.add(Card(Rank.TWO, Suit.CLUB))  # 2♣
    player0.cards.add(Card(Rank.QUEEN, Suit.SPADE))  # Q♠

    # Deal some random cards to all players
    cards = [Card.get_card_from_id(card_id) for card_id in range(2, 54)]
    cards = [card for card in cards if card not in player0.cards]
    
    for player in game.players:
        card_count = 7 if player != player0 else 5  # player0 already has 2 cards
        for _ in range(card_count):
            if cards:
                card = cards.pop()
                player.cards.add(card)

    for player in game.players:
        for card in player.cards:
            card.owner = player.id.value
    
    # Setup stock pile and discard pile
    game.stock_pile = cards
    for card in game.stock_pile:
        card.owner = PlayerIndex.STOCK_PILE.value
        
    # Put a card on the discard pile
    if game.stock_pile:
        game.head_card = game.stock_pile.pop()
        game.discard_pile = [game.head_card]
        game.head_card.owner = PlayerIndex.DISCARD_PILE.value
    
    # Set the game state and phase
    game.game_state = GameState.STARTED
    game.phase = GamePhase.PICK_OR_DRAW
    game.current_turn = 0
    game.current_player = game.players[0]
    game.can_dark_knock = [True for _ in game.players]

    game.save_initial_state()
    
    return game

def setup_knock_scenario(game: DummyGame):
    # Reset the game to a clean state
    game.reset()
    
    # Start a game with 4 players in clockwise direction
    game.start_game(TurnDirection.CLOCKWISE, [True, True, True, True], 0)
    
    # Setup a scenario where player0 can knock right away
    player0 = game.players[0]
    
    # Add a form to player0's hand that can be melded
    player0.cards.add(Card(Rank.SEVEN, Suit.HEART))
    player0.cards.add(Card(Rank.EIGHT, Suit.HEART))
    player0.cards.add(Card(Rank.NINE, Suit.HEART))
    player0.cards.add(Card(Rank.SEVEN, Suit.CLUB))
    player0.cards.add(Card(Rank.EIGHT, Suit.CLUB))
    player0.cards.add(Card(Rank.NINE, Suit.CLUB))
    
    # Add one extra card that can be used to knock
    player0.cards.add(Card(Rank.TEN, Suit.SPADE))

    head_card = Card(Rank.TEN, Suit.CLUB)
    
    # Deal some random cards to other players
    cards = [Card.get_card_from_id(card_id) for card_id in range(2, 54)]
    cards = [card for card in cards if card not in player0.cards and card != head_card]
    
    for player in game.players[1:]:
        for _ in range(7):
            if cards:
                card = cards.pop()
                card.owner = player.id.value
                player.cards.add(card)
    
    # Setup stock pile and discard pile
    game.stock_pile = cards
    for card in game.stock_pile:
        card.owner = PlayerIndex.STOCK_PILE.value
        
    # Put a card on the discard pile
    if game.stock_pile:
        game.head_card = head_card
        game.discard_pile = [game.head_card]
        game.head_card.owner = PlayerIndex.DISCARD_PILE.value
    
    # Set the game state and phase
    game.game_state = GameState.STARTED
    game.phase = GamePhase.PICK_OR_DRAW
    game.current_turn = 0
    game.current_player = game.players[0]
    game.can_dark_knock = [True for _ in game.players]

    game.save_initial_state()
    
    return game


class Scenario:
    SpetoScenario = GameScenario("speto", "SPETO Scenario", "Player 0 has both 2♣ and Q♠ to demonstrate SPETO action", setup_speto_scenario)
    KnockScenario = GameScenario("knock", "Quick Knock Scenario", "Player 0 has cards to meld and knock right away", setup_knock_scenario)


# Global game history manager instance
game_history_manager = DummyGameHistoryManager()