from __future__ import annotations
from scripts_of_tribute.base_ai import BaseAI
from scripts_of_tribute.board import GameState, EndGameState
from scripts_of_tribute.enums import MoveEnum, PlayerEnum
from scripts_of_tribute.move import BasicMove, SimpleCardMove, MakeChoiceMoveUniqueEffect
import time
import random
import numpy as np
from typing import Callable
import re

# tournament addition
from ..extensions import safe_play

def Linear(x: np.ndarray) -> np.ndarray:
    return x

def ThanH(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def Sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def LeakyReLU(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(x > 0, x, alpha * x)

def ELU(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def Softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

ACTIVATION_FUNCTION_NAME_MAP = {
    "linear": Linear,
    "tanh": ThanH,
    "sigmoid": Sigmoid,
    "leaky_relu": LeakyReLU,
    "elu": ELU,
}

GameStateEvaluator = Callable[[GameState], np.ndarray]

def HandStatistics(game_state: GameState, regex):
    cards = game_state.current_player.draw_pile + game_state.current_player.hand + game_state.current_player.cooldown_pile
    pattern = re.compile(rf"({regex}) (\d+)")

    valid_card = [
        (card, int(pattern.match(effect).group(2)))
        for card in cards
        for effect in card.effects
        if pattern.match(effect)
    ]
    if valid_card is [] or valid_card is None:
        print("no valid card founded")
        return 0,0,0,0

    cards_sorted = sorted(valid_card, key=lambda x: x[1], reverse=True)

    top_5_hand = sum(n for _, n in cards_sorted[:5])
    bottom_5_hand = sum(n for _, n in cards_sorted[-5:])
    average_singleCard = sum(n for _, n in cards_sorted) / len(cards_sorted) if cards_sorted else 0
    average_Hand = (top_5_hand + bottom_5_hand)/ 2

    return top_5_hand,bottom_5_hand, average_Hand,average_singleCard

def CalculateMaxMinAverageCoin(game_state: GameState):
    return HandStatistics(game_state,"GAIN_COIN")

def CalculateMaxMinAveragePowerAndPrestige( game_state: GameState):
    return HandStatistics(game_state, "GAIN_POWER|GAIN_PRESTIGE")

def CalculateFavor(game_state:GameState, player_id):
    patron_favor = game_state.patron_states.patrons.items()
    favor = 0
    for patron_id, player_enum in patron_favor:
        if player_enum == PlayerEnum.NO_PLAYER_SELECTED:
            continue # if the patron favor is neutral, ignore it
        elif player_enum == player_id:
            favor = favor + 1
        else:
            favor = favor - 1
    return favor

def CalculateCoinLeft(game_state: GameState):
    return game_state.current_player.coins

#MINMAX HAND VALUE RATING
def MMHVR_values(game_state: GameState ) ->np.ndarray:
        top_hand_coin,bottom_hand_coin, average_Hand_coin, average_singleCard_coin = CalculateMaxMinAverageCoin(game_state)
        top_hand_PEP,bottom_hand_PEP, average_Hand_PEP, average_singleCard_PEP = CalculateMaxMinAveragePowerAndPrestige(game_state)
        favor     = CalculateFavor(game_state, game_state.current_player.player_id)
        coin_left = CalculateCoinLeft(game_state)
        prestige  = game_state.current_player.prestige
        power     = game_state.current_player.power


        param   = np.array([np.log(top_hand_coin),bottom_hand_coin, average_Hand_coin, average_singleCard_coin,
                            top_hand_PEP**1.3,bottom_hand_PEP, average_Hand_PEP, average_singleCard_PEP,
                            prestige, power,
                            np.sign(favor) * favor**2, -coin_left, -game_state.enemy_player.prestige**1.1])

        return param

def CalculateWeightedUtility(utility_function:GameStateEvaluator, game_state: GameState, weights:np.ndarray = None, functions =None) -> float:

    param =  utility_function(game_state)
    param_dimension = param.shape[0]

    weighted_values = weights * param if weights is not None  else param
    if functions is not None:
        for i in range(param_dimension):
            act_fun =  ACTIVATION_FUNCTION_NAME_MAP[functions[i]]
            weighted_values[i] = act_fun(weighted_values[i])
    return float(np.sum(weighted_values))


GameStateEvaluatorUtility = Callable[[GameState,], float]

def utility_boost (game_state, utility, player_id):
    if game_state.end_game_state is not None:
        if game_state.end_game_state.winner == player_id:
            utility *= 3
        elif game_state.end_game_state.winner != player_id:
            utility *= 0.5

    return utility

def WeightedUtilityFunction_MMHVR(game_state: GameState, weights:np.ndarray = None, functions =None, player_id = None, *args, **kwargs) -> float:
    utility = CalculateWeightedUtility(MMHVR_values, game_state, weights, functions)

    if player_id is not None:
        utility = utility_boost(game_state, utility, player_id)

    return utility

def CheckForGoalState(game_state: GameState, player_id) -> bool:
    return game_state.end_game_state is not None and game_state.end_game_state.winner == player_id

def obtain_move_semantic_id(move:BasicMove) -> tuple[int,int|tuple]:
    if hasattr(move, 'cardUniqueId'):
        return move.command.value, move.cardUniqueId
    elif hasattr(move, 'patronId'):
        return move.command.value, int(move.patronId.value)
    elif hasattr(move, 'cardsUniqueIds'):
        return move.command.value, tuple(move.cardsUniqueIds)
    elif hasattr(move, 'effects'):
        return move.command.value, tuple(move.effects)
    else:
        return move.command.value, -1

def calculate_ucb(total_utility, number_of_playouts, parents_number_of_playouts) -> float:
    if number_of_playouts == 0:
        raise ValueError("Node must have at least one playout before it can be calculated")

    exploitation_term = total_utility / number_of_playouts
    c = np.sqrt(2)
    exploration_term = np.log(parents_number_of_playouts) / number_of_playouts

    return exploitation_term + c * np.sqrt(exploration_term)

def MakePriorChoice(game_state:GameState, possible_moves: list[BasicMove], heuristic) -> MakeChoiceMoveUniqueEffect | None:
    choice = [mv for mv in possible_moves if isinstance(mv, MakeChoiceMoveUniqueEffect)]
    if len(choice) != len(possible_moves):
        return None

    best_move_value = float('-inf')
    best_move = None

    for move in choice:
        new_game_state,_ = game_state.apply_move(move)
        move_value = heuristic(new_game_state)
        if move_value > best_move_value:
            best_move_value = move_value
            best_move = move
    return best_move

def IsPriorMoves(move: BasicMove) -> bool:
    return isinstance(move, SimpleCardMove) and move.command != MoveEnum.BUY_CARD

class Node:
    def __init__(self):
        self.children: dict[tuple, NotRootNode] = {}
        # for ucb calculations
        self.number_of_playouts = 0
        self.total_utility = 0

    def expand(self, move: BasicMove, semantic_id: tuple):
        child = NotRootNode(self, move)
        if semantic_id in self.children.keys():
            raise ValueError(f"Child with semantic_id {semantic_id} already exists.")
        self.children[semantic_id] = child

    def search_unexpanded_child(self, possible_moves: list[BasicMove]) -> NotRootNode | None:
        for move in possible_moves:
            semantic_id = obtain_move_semantic_id(move)
            if semantic_id not in self.children:
                self.expand(move, semantic_id)
                return self.children[semantic_id]
        return None

class NotRootNode(Node):
    def __init__(self, parent, parent_move):
        super().__init__()
        self.parent: Node = parent
        self.parent_move: BasicMove = parent_move

    def calculate_ucb(self):
        return calculate_ucb(self.total_utility, self.number_of_playouts, self.parent.number_of_playouts)

    def back_propagation(self, utility):
        current_node = self
        while current_node is not None:
            current_node.number_of_playouts += 1
            current_node.total_utility += utility

            if hasattr(current_node, 'parent'):
                current_node = current_node.parent
            else:
                current_node = None

    def update_parent_move(self, moves: list[BasicMove]):
        parent_move_semantic_id = obtain_move_semantic_id(self.parent_move)
        for move in moves:
            semantic_id = obtain_move_semantic_id(move)
            if semantic_id == parent_move_semantic_id:
                self.parent_move = move
                return

        raise ValueError("parent move not found")

class RootNode(Node):
    def __init__(self, game_state, possible_moves):
        super().__init__()
        self.game_state: GameState = game_state
        self.possible_moves: list[BasicMove] = possible_moves

class MCTS:
    def __init__(self, game_state, possible_moves, player_id, evaluation_function):
        self.root = RootNode(game_state, possible_moves)
        self.player_id = player_id
        self.evaluation_function = evaluation_function
        random.seed(16)

    def evaluation(self, game_state: GameState) -> float:
        return self.evaluation_function(game_state)

    def playout_and_back_prop(self, node, game_state):
        terminal_game_state = MCTS.playout(node.parent_move, game_state, self.player_id)
        utility = self.evaluation(terminal_game_state)
        node.back_propagation(utility)

    def iteration(self):
        actual_node = self.root
        actual_game_state = self.root.game_state
        actual_possible_moves = self.root.possible_moves

        new_child = None
        while new_child is None:
            if actual_game_state.current_player.player_id != self.player_id:
                raise ValueError("Searching too deep")

            new_child: NotRootNode | None = actual_node.search_unexpanded_child(actual_possible_moves)

            if new_child is None:
                actual_node = MCTS.selection(actual_node.children, actual_possible_moves)

                actual_node.update_parent_move(actual_possible_moves)

                if actual_node.parent_move.command == MoveEnum.END_TURN:
                    self.playout_and_back_prop(actual_node, actual_game_state)
                    break

                try:
                    actual_game_state, actual_possible_moves = actual_game_state.apply_move(actual_node.parent_move)
                except Exception as e:
                    raise ValueError (f"problems with apply_move, Move: {actual_node.parent_move} ")

            elif new_child is not None:
                self.playout_and_back_prop(new_child, actual_game_state)

    @staticmethod
    def playout(move: BasicMove, game_state: GameState, player_id) -> GameState:
        while not (CheckForGoalState(game_state, player_id) or move.command == MoveEnum.END_TURN):
            try:
                game_state, possible_moves = game_state.apply_move(move)
            except Exception as e:
                raise ValueError ("problems with apply_move")
            move = random.choice(possible_moves)
        return game_state

    @staticmethod
    def selection (nodes: dict[tuple, NotRootNode], possible_moves: list[BasicMove]) -> NotRootNode:
        actual_children: list[NotRootNode] = []
        for possible_move in possible_moves:
            semantic_id = obtain_move_semantic_id(possible_move)
            if semantic_id in nodes.keys():
                actual_child = nodes[semantic_id]
                actual_children.append(actual_child)

        if len(actual_children) == 0:
            raise ValueError("no actual child found")

        best_node = actual_children[0]
        best_ucb = float('-inf')

        for node in actual_children:
            ucb = node.calculate_ucb()
            if ucb > best_ucb:
                best_node = node
                best_ucb = ucb

        return best_node

    def move_choice(self, max_iterations, given_time) -> BasicMove:
        # print(f"    [MCTS] -> start move choice with {len(self.root.possible_moves)} possible moves and {max_iterations} iterations with {given_time} ms time limit")
        start_time = time.perf_counter()
        for i in range(max_iterations):
            elapsed_time_ms = (time.perf_counter() - start_time) * 1000
            if elapsed_time_ms - given_time> 150:
                # print(f"    [MCTS] -> early stopping time {elapsed_time_ms} ms")
                break
            self.iteration()

        best_move = random.choice(self.root.possible_moves)
        best_utility = float('-inf')

        if len(self.root.children.values()) != len(self.root.possible_moves):
            raise ValueError("Incoherence between possible moves and children")

        nodes = self.root.children.values()
        for node in nodes:
            if node.total_utility > best_utility:
                best_utility = node.total_utility
                best_move = node.parent_move

        return best_move

class AIFBotMCTS(BaseAI):

    ## ========================SET UP========================
    def __init__(self, bot_name, seed=None, weights=None, functions=None):
        super().__init__(bot_name)
        self.evaluation_function = WeightedUtilityFunction_MMHVR
        self.player_id: PlayerEnum = PlayerEnum.NO_PLAYER_SELECTED
        self.start_of_game: bool = True
        self.best_moves:list[BasicMove] = []
        self.seed = seed if seed is not None else random.randint(0, 2**64)
        self.Weights = weights
        self.Functions =functions

    def select_patron(self, available_patrons):
        pick = random.choice(available_patrons)
        return pick

    ## ========================Functionality========================
    def UtilityFunction(self, game_state: GameState) -> float:
        return self.evaluation_function(game_state, self.Weights, self.Functions, self.player_id)

    @safe_play(fallback="last")
    def play(self, game_state: GameState, possible_moves:list[BasicMove], remaining_time: int) -> BasicMove:
        #Set Up
        if self.start_of_game:
            self.player_id = game_state.current_player.player_id
            self.start_of_game = False

        if len(possible_moves) == 1 and possible_moves[0].command == MoveEnum.END_TURN:
            return possible_moves[0]

        for move in possible_moves:
            if IsPriorMoves(move):
                # Return the first prior move encountered
                return move

        best_choice = MakePriorChoice(game_state, possible_moves, self.UtilityFunction)
        if best_choice is not None:
            # print(f"    [Prior choice] -> selected move {best_choice.command}")
            return best_choice

        #Move Evaluation
        monte_carlo_tree_search = MCTS(game_state, possible_moves, self.player_id, self.UtilityFunction)
        best_move = monte_carlo_tree_search.move_choice(500, 1000)

        # End of Search
        if best_move is None:
            best_move = next(move for move in possible_moves if move.command == MoveEnum.END_TURN)

        return best_move

    def game_end(self, end_game_state: EndGameState, final_state: GameState):
        pass

