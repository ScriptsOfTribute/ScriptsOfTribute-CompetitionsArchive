import random
from datetime import timedelta
import time
import math
from collections import defaultdict

from scripts_of_tribute.base_ai import BaseAI
from scripts_of_tribute.board import PlayerEnum, GameState, BasicMove,  CurrentPlayer,  SeededGameState
from scripts_of_tribute.move import    BasicMove
from scripts_of_tribute.enums import MoveEnum, PatronId, BoardState


# tournament addition
from ..extensions import safe_play

class TierEnum:
    S = 50/50
    A = 25/50
    B = 10/50
    C = 5/50
    D = 1/50
    UNKNOWN = 0

CARD_TIER_DICT : dict[str,float] = {
    "Currency Exchange": TierEnum.S,
    "Luxury Exports": TierEnum.S,
    "Oathman": TierEnum.A,
    "Ebony Mine": TierEnum.B,
    "Hlaalu Councilor": TierEnum.B,
    "Hlaalu Kinsman": TierEnum.B,
    "House Embassy": TierEnum.B,
    "House Marketplace": TierEnum.B,
    "Hireling": TierEnum.C,
    "Hostile Takeover": TierEnum.C,
    "Kwama Egg Mine": TierEnum.C,
    "Customs Seizure": TierEnum.D,
    "Goods Shipment": TierEnum.D,
    "Midnight Raid": TierEnum.S,
    "Blood Sacrifice": TierEnum.S,
    "Bloody Offering": TierEnum.S,
    "Bonfire": TierEnum.C,
    "Briarheart Ritual": TierEnum.C,
    "Clan-Witch": TierEnum.C,
    "Elder Witch": TierEnum.B,
    "Hagraven": TierEnum.B,
    "Hagraven Matron": TierEnum.A,
    "Imperial Plunder": TierEnum.A,
    "Imperial Spoils": TierEnum.B,
    "Karth Man-Hunter": TierEnum.C,
    "War Song": TierEnum.D,
    "Blackfeather Knave": TierEnum.S,
    "Plunder": TierEnum.S,
    "Toll of Flesh": TierEnum.S,
    "Toll of Silver": TierEnum.S,
    "Murder of Crows": TierEnum.A,
    "Pilfer": TierEnum.A,
    "Squawking Oratory": TierEnum.A,
    "Law of Sovereign Roost": TierEnum.B,
    "Pool of Shadow": TierEnum.B,
    "Scratch": TierEnum.B,
    "Blackfeather Brigand": TierEnum.C,
    "Blackfeather Knight": TierEnum.C,
    "Peck": TierEnum.D,
    "Conquest": TierEnum.S,
    "Grand Oratory": TierEnum.S,
    "Hira's End": TierEnum.S,
    "Hel Shira Herald": TierEnum.A,
    "March on Hattu": TierEnum.A,
    "Shehai Summoning": TierEnum.A,
    "Warrior Wave": TierEnum.A,
    "Ansei Assault": TierEnum.B,
    "Ansei's Victory": TierEnum.B,
    "Battle Meditation": TierEnum.B,
    "No Shira Poet": TierEnum.C,
    "Way of the Sword": TierEnum.D,
    "Prophesy": TierEnum.S,
    "Scrying Globe": TierEnum.S,
    "The Dreaming Cave": TierEnum.S,
    "Augur's Counsel": TierEnum.B,
    "Psijic Relicmaster": TierEnum.A,
    "Sage Counsel": TierEnum.A,
    "Prescience": TierEnum.B,
    "Psijic Apprentice": TierEnum.B,
    "Ceporah's Insight": TierEnum.C,
    "Psijic's Insight": TierEnum.C,
    "Time Mastery": TierEnum.D,
    "Mainland Inquiries": TierEnum.D,
    "Rally": TierEnum.S,
    "Siege Weapon Volley": TierEnum.S,
    "The Armory": TierEnum.S,
    "Banneret": TierEnum.A,
    "Knight Commander": TierEnum.A,
    "Reinforcements": TierEnum.A,
    "Archers' Volley": TierEnum.B,
    "Legion's Arrival": TierEnum.B,
    "Shield Bearer": TierEnum.B,
    "Bangkorai Sentries": TierEnum.C,
    "Knights of Saint Pelin": TierEnum.C,
    "The Portcullis": TierEnum.D,
    "Fortify": TierEnum.D,
    "Bag of Tricks": TierEnum.B,
    "Bewilderment": TierEnum.D,
    "Grand Larceny": TierEnum.A,
    "Jarring Lullaby": TierEnum.S,
    "Jeering Shadow": TierEnum.B,
    "Moonlit Illusion": TierEnum.A,
    "Pounce and Profit": TierEnum.S,
    "Prowling Shadow": TierEnum.B,
    "Ring's Guile": TierEnum.B,
    "Shadow's Slumber": TierEnum.A,
    "Slight of Hand": TierEnum.B,
    "Stubborn Shadow": TierEnum.B,
    "Swipe": TierEnum.D,
    "Twilight Revelry": TierEnum.S,
    "Ghostscale Sea Serpent": TierEnum.B,
    "King Orgnum's Command": TierEnum.C,
    "Maormer Boarding Party": TierEnum.B,
    "Maormer Cutter": TierEnum.B,
    "Pyandonean War Fleet": TierEnum.B,
    "Sea Elf Raid": TierEnum.C,
    "Sea Raider's Glory": TierEnum.C,
    "Sea Serpent Colossus": TierEnum.B,
    "Serpentguard Rider": TierEnum.A,
    "Serpentprow Schooner": TierEnum.B,
    "Snakeskin Freebooter": TierEnum.S,
    "Storm Shark Wavecaller": TierEnum.B,
    "Summerset Sacking": TierEnum.B,
    "Ambush": TierEnum.B,
    "Barterer": TierEnum.C,
    "Black Sacrament": TierEnum.B,
    "Blackmail": TierEnum.B,
    "Gold": TierEnum.UNKNOWN,
    "Harvest Season": TierEnum.C,
    "Imprisonment": TierEnum.C,
    "Ragpicker": TierEnum.C,
    "Tithe": TierEnum.C,
    "Writ of Coin": TierEnum.D,
    "Unknown": TierEnum.UNKNOWN,
    "Alessian Rebel": TierEnum.C,
    "Ayleid Defector": TierEnum.B,
    "Ayleid Quartermaster": TierEnum.B,
    "Chainbreaker Captain": TierEnum.A,
    "Chainbreaker Sergeant": TierEnum.B,
    "Morihaus, Sacred Bull": TierEnum.S,
    "Morihaus, the Archer": TierEnum.A,
    "Pelinal Whitestrake": TierEnum.S,
    "Priestess of the Eight": TierEnum.B,
    "Saint's Wrath": TierEnum.B,
    "Soldier of the Empire": TierEnum.C,
    "Whitestrake Ascendant": TierEnum.S,
}


class Node:
    """Represents a node in the Monte Carlo Tree Search."""

    def __init__(self, father_game_state: SeededGameState, node_move: BasicMove, father_orig: 'Node', possible_moves: list[BasicMove] = None):
        # Heuristic values MCTS
        self.patron_favour = 50
        self.patron_neutral = 10
        self.patron_unfavour = -50
        self.power_value = 40 # was 40
        self.prestige_value = 50 # was 50
        self.agent_on_board_value = 30
        self.hp_value = 3
        self.opponent_agents_penalty_value = 40
        self.potential_combo_value = 3
        self.card_value = 11 # was 10
        self.penalty_for_high_tier_in_tavern = 2
        self.heuristic_max = 40000
        self.heuristic_min = -10000
        self.c = math.sqrt(2)  # UCB exploration parameter

        self.wins = 0.0
        self.visits = 0
        self.move = node_move
        self.father = father_orig
        self.childs: list['Node'] = []
        
        # Apply the move to get the new game state for this node
        if node_move is not None and node_move.command != MoveEnum.END_TURN:
            new_game_state, new_moves = father_game_state.apply_move(node_move)
            self.node_game_state = new_game_state
            self.possible_moves = new_moves
        else:
            self.node_game_state = father_game_state
            self.possible_moves = possible_moves if possible_moves is not None else []


    def create_childs(self):
        """Creates child nodes for all possible moves from the current state."""
        for child_move in self.possible_moves:
            self.childs.append(Node(self.node_game_state, child_move, self))

    def ucb_score(self) -> float:
        """Calculates the UCB1 score for this node."""
        if self.visits < 1:
            return float('inf')
        
        # The C# implementation uses a custom wins value.
        # The UCB formula is wins/visits + c * sqrt(log(parent_visits)/visits)
        parent_visits = self.father.visits if self.father else self.visits
        
        # The original C# code adds wins and the exploration term. 
        # A more standard approach is to use the average win rate (wins/visits).
        # Replicating the original logic:
        return self.wins + self.c * math.sqrt(math.log(parent_visits) / self.visits)

    def select_best_child(self) -> 'Node':
        """
        Selects the best child node.
        The original C# code selects based on the highest raw 'wins' value, which is unusual.
        A standard MCTS would select the child with the highest average win rate (wins/visits).
        This implementation replicates the original logic.
        """
        if not self.childs:
            return self
            
        best_child = self.childs[0]
        best_score = -float('inf')

        for child in self.childs:
            # Original logic uses raw wins as the score for final selection
            score = child.wins 
            if score > best_score:
                best_score = score
                best_child = child

        #print(f"Move from best child is: {best_child.move.command} with id {best_child.move.move_id} and score {best_score}")
        return best_child

    def _not_end_turn_possible_moves(self, moves: list[BasicMove]) -> list[BasicMove]:
        return [m for m in moves if m.command != MoveEnum.END_TURN]

    def _draw_next_move(self, moves: list[BasicMove], game_state: SeededGameState, rng: random.Random) -> BasicMove:
        """Selects a random move for the simulation phase."""
        not_end_turn_moves = self._not_end_turn_possible_moves(moves)
        if not_end_turn_moves:
            # Small chance to end turn even if other moves are available
            if game_state.board_state == BoardState.NORMAL and rng.randint(0, 1000) == 42:
                return BasicMove(move_id=0, command=MoveEnum.END_TURN)
            return rng.choice(not_end_turn_moves)
        return BasicMove(move_id=0, command=MoveEnum.END_TURN)

    def simulate(self, rng: random.Random) -> float:
        """
        Simulates a random playout from the current node's state and returns a heuristic score.
        """
        if self.move and self.move.command == MoveEnum.END_TURN:
            return self.heuristic(self.node_game_state)

        game_state = self.node_game_state
        moves = self.possible_moves
        
        next_move = self._draw_next_move(moves, game_state, rng)

        while next_move.command != MoveEnum.END_TURN:
            game_state, moves = game_state.apply_move(next_move)
            next_move = self._draw_next_move(moves, game_state, rng)
            
        return self.heuristic(game_state)

    def heuristic(self, game_state: SeededGameState) -> float:
        """
        Evaluates the given game state and returns a score.
        This is a direct translation of the C# heuristic logic.
        """
        final_value = 0
        enemy_patron_favour = 0
        
        current_player_id = game_state.current_player.player_id
        
        for patron_id, player_id in game_state.patron_states.patrons.items():
            if patron_id == PatronId.TREASURY:
                continue
            if player_id == current_player_id:
                final_value += self.patron_favour
            elif player_id == PlayerEnum.NO_PLAYER_SELECTED:
                final_value += self.patron_neutral
            else:
                final_value += self.patron_unfavour
                enemy_patron_favour += 1
        
        if enemy_patron_favour >= 2:
            final_value -= 100

        final_value += game_state.current_player.power * self.power_value
        final_value += game_state.current_player.prestige * self.prestige_value

        if game_state.current_player.prestige < 30:

            for agent in game_state.current_player.agents:
                tier = CARD_TIER_DICT.get(agent.representing_card.name, TierEnum.UNKNOWN)
                final_value += self.agent_on_board_value * tier + agent.currentHP * self.hp_value

            for agent in game_state.enemy_player.agents:
                tier = CARD_TIER_DICT.get(agent.representing_card.name, TierEnum.UNKNOWN)
                final_value -= self.agent_on_board_value * tier + agent.currentHP * self.hp_value + self.opponent_agents_penalty_value

            if type(game_state.enemy_player)!= CurrentPlayer:
                all_cards = game_state.current_player.hand + game_state.current_player.played + game_state.current_player.cooldown_pile + game_state.current_player.draw_pile
                potential_combo_number = defaultdict(int)

                all_cards_enemy = game_state.enemy_player.hand_and_draw + game_state.enemy_player.hand_and_draw + game_state.enemy_player.played + game_state.enemy_player.cooldown_pile
                potential_combo_number_enemy = defaultdict(int)

            elif type(game_state.enemy_player) == CurrentPlayer:
                all_cards = game_state.current_player.hand + game_state.current_player.played + game_state.current_player.cooldown_pile + game_state.current_player.draw_pile
                potential_combo_number = defaultdict(int)

                all_cards_enemy = game_state.enemy_player.hand + game_state.enemy_player.hand + game_state.enemy_player.played + game_state.enemy_player.cooldown_pile
                potential_combo_number_enemy = defaultdict(int)

            for card in all_cards:
                tier =CARD_TIER_DICT.get(card.name, TierEnum.UNKNOWN)
                final_value += tier * self.card_value
                if card.deck != PatronId.TREASURY:
                    potential_combo_number[card.deck] += 1
            
            for card in all_cards_enemy:
                if card.deck != PatronId.TREASURY:
                    potential_combo_number_enemy[card.deck] += 1

            for count in potential_combo_number.values():
                final_value += int(math.pow(count, self.potential_combo_value))

            for card in game_state.tavern_available_cards:
                tier = CARD_TIER_DICT.get(card.name, TierEnum.UNKNOWN)
                final_value -= self.penalty_for_high_tier_in_tavern * tier

        return self._normalize_heuristic(final_value)

    def _normalize_heuristic(self, value: int) -> float:
        """Normalizes the heuristic score to a value between 0 and 1."""
        normalized_value = (value - self.heuristic_min) / (self.heuristic_max - self.heuristic_min)
        return max(0.0, normalized_value)





class CouncilOfTwo(BaseAI):

    def __init__(self, bot_name):
        super().__init__(bot_name)
        self.my_id: PlayerEnum = PlayerEnum.NO_PLAYER_SELECTED
        self.start_of_game: bool = True
        self.start_of_turn :bool = True
        self.turn_counter = 0


        self.rng = random.Random()


    def pregame_prepare(self):
        super().pregame_prepare()
        self.start_of_game = True
        self.start_of_turn = True
        self.turn_counter = 0


        ## MCTS Stuff ##
        self.act_root: Node | None = None
        self.act_node: Node | None = None
        self.start_of_game = True
        self.start_of_turn = True
        self.used_time_in_turn = timedelta(0)
        # Time allocated for computation per move.
        self.time_for_move_computation = timedelta(seconds=0.3) # was 0.3
        # Total time allowed per turn.
        self.turn_timeout = timedelta(seconds=9.9) #was 29.9

    def _check_if_possible_moves_are_the_same(self, moves1: list[BasicMove], moves2: list[BasicMove]) -> bool:
        """Checks if two lists of moves are identical."""
        # check if both lists are equal in terms of commands
        if len(moves1) != len(moves2):
            return False
        # Compare the commands of the moves in both lists
        moves1 : list[MoveEnum] = [move.command for move in moves1 if move.command != MoveEnum.END_TURN]
        moves2 : list[MoveEnum] = [move.command for move in moves2 if move.command != MoveEnum.END_TURN]
        if len(moves1) != len(moves2):
            return False
        return set(moves1) == set(moves2)

    def _tree_policy(self, v: Node) -> Node:
        """
        Traverses the tree from node v to find the best node to expand.
        Uses UCB scores for selection.
        """
        if v.childs:
            best_child = None
            max_value = -float('inf')
            for child_node in v.childs:
                ucb = child_node.ucb_score()
                if ucb > max_value:
                    max_value = ucb
                    best_child = child_node
            return self._tree_policy(best_child)

        if v.move and v.move.command == MoveEnum.END_TURN:
            return v
        
        v.create_childs()
        return v

    def _back_up(self, v: Node , delta: float):
        """
        Backpropagates the simulation result up the tree.
        The original C# code uses Math.Max, which is not standard for MCTS.
        A standard approach would sum the results. This implementation replicates the original.
        """
        if v:
            v.visits += 1
            v.wins = max(delta, v.wins)
            self._back_up(v.father, delta)

    def select_patron(self, available_patrons):
        limited_patron_list = [0,1,2,4,6,9]
        selected_patrons = [p for p in available_patrons if p.value in limited_patron_list]

        """
        For patron 0 the average reward is 60.27 with sample size 53775
        For patron 1 the average reward is 65.49 with sample size 211416
        For patron 2 the average reward is 87.24 with sample size 115207
        For patron 6 the average reward is 70.22 with sample size 339649
        For patron 9 the average reward is 124.31 with sample size 100
        For combo (0, 1) the average reward is 79.00 with sample size 20117
        For combo (0, 6) the average reward is 49.08 with sample size 33654
        For combo (1, 2) the average reward is 234.83 with sample size 203
        For combo (1, 6) the average reward is 63.86 with sample size 190992
        For combo (1, 9) the average reward is 124.31 with sample size 100
        For combo (2, 6) the average reward is 86.97 with sample size 114996


        For patron 0 the average reward is 152.54 with sample size 5900
        For patron 1 the average reward is 111.60 with sample size 61707
        For patron 2 the average reward is 113.14 with sample size 57835
        For patron 6 the average reward is 110.93 with sample size 116364
        For combo (0, 1) the average reward is 153.00 with sample size 4336
        For combo (0, 6) the average reward is 151.27 with sample size 1564
        For combo (1, 2) the average reward is 234.83 with sample size 203
        For combo (1, 6) the average reward is 108.03 with sample size 57168
        For combo (2, 6) the average reward is 112.71 with sample size 57632

        For patron 0 the average reward is 168.31 with sample size 2828
        For patron 1 the average reward is 161.39 with sample size 9183
        For patron 2 the average reward is 174.99 with sample size 2780
        For patron 6 the average reward is 157.42 with sample size 11717
        For combo (0, 1) the average reward is 189.15 with sample size 1334
        For combo (0, 6) the average reward is 149.70 with sample size 1494
        For combo (1, 2) the average reward is 234.83 with sample size 203
        For combo (1, 6) the average reward is 154.60 with sample size 7646
        For combo (2, 6) the average reward is 170.27 with sample size 2577
        """
        
        if PatronId.DUKE_OF_CROWS.value in [p.value for p in available_patrons]:
            return PatronId.DUKE_OF_CROWS
        
        if PatronId.RAJHIN.value in [p.value for p in available_patrons]:
            return PatronId.RAJHIN

        if PatronId.ANSEI.value in [p.value for p in available_patrons]:
            return PatronId.ANSEI
        
        if PatronId.PELIN.value in [p.value for p in available_patrons]:
            return PatronId.PELIN
        



        rng = random.Random(42)
        pick = rng.choice(selected_patrons)

        
        return pick
    
    @safe_play(fallback="last")
    def play(self, game_state:GameState, possible_moves: list[BasicMove], remaining_time):

        if self.start_of_game:
            self.my_id = game_state.current_player.player_id
            self.start_of_game = False

            
        if self.start_of_turn:
            self.turn_counter += 1
            self.start_of_turn = False

            ## MCTS STuff ##
            self.act_root = Node(game_state, None, None, possible_moves)
            self.act_root.create_childs()
            self.start_of_turn = False
            self.used_time_in_turn = timedelta(0)
        else:
                ## More MCTS Stuff ## 
                # If the game state has changed in an unexpected way, create a new root.
                #if not self._check_if_possible_moves_are_the_same(self.act_root.possible_moves, possible_moves):
                self.act_root = Node(game_state, None, None, possible_moves)
                self.act_root.create_childs()
        

        ### There are cases that are no real 'decisions' so waist no time and store no state for training ###
        if len(possible_moves) == 1 and possible_moves[0].command == MoveEnum.END_TURN:
                self.start_of_turn = True

        """
        try:
            move_option_maxbot = self.what_would_maxbot_do(game_state, possible_moves) 
        except Exception as e:
            print(f"Error is in algo2 { e}")
        try:
            move_option_MCTS = self.what_would_MCTS_do(game_state, possible_moves)
        except Exception as e:
            print(f"Error is in algo3 { e}")
        """

        try:

            # Naïvely we choose maxbot if we are first player
            if self.my_id == PlayerEnum.PLAYER1:
                chosen_index = self.breadth_search(game_state, possible_moves)
                #move_option_maxbot = self.what_would_maxbot_do(game_state, possible_moves) 
                #chosen_index = move_option_maxbot
            else:
                chosen_index = self.breadth_search(game_state, possible_moves)
                #move_option_maxbot = self.what_would_maxbot_do(game_state, possible_moves) 
                #chosen_index = move_option_maxbot
                #move_option_MCTS = self.what_would_MCTS_do(game_state, possible_moves)
                #chosen_index = move_option_MCTS

            pick = possible_moves[chosen_index]
        except Exception as e:
                print(f"Things are still wrong further up the script and therefor: { e}")
                pick = possible_moves[0]

        if pick.command == MoveEnum.END_TURN:
            self.used_time_in_turn = timedelta(0)
            self.start_of_turn = True

        return pick
    
    def game_end(self, end_game_state, final_state):
        # Save the visited states to a file with a date and time stamp and a random number

        winner = PlayerEnum.PLAYER1 if end_game_state.winner == "PLAYER1" else PlayerEnum.PLAYER2
        print(f"The winner is {end_game_state.winner} because of {end_game_state.reason}. Scores: {final_state.current_player.prestige} vs {final_state.enemy_player.prestige}")


    def breadth_search(self,game_state:GameState, possible_moves):

         ### There are cases that are no real 'decisions' so waist no time and store no state for training ###
        if len(possible_moves) == 1 and possible_moves[0].command == MoveEnum.END_TURN:
                return 0
        
        # If hand had gold or 'writ' cards, just play them. Saves on compute time
        if MoveEnum.PLAY_CARD in [m.command for m in possible_moves]:
            for i,m in enumerate(possible_moves):
                card_index = 0
                if m.command == MoveEnum.PLAY_CARD:
                    # This relies on the moves being in the same order as the cards in hand
                    card_name = game_state.current_player.hand[card_index].name
                    if card_name in ['Gold', 'Writ of Coin']:
                        return i
                    card_index +=1

            



        # Tracking for training
        #self.visited_states.append(self.convert_state_to_embedding(game_state, raw=True))

        best_move = None
        best_move_val = -99
        chosen_move_index = -999

        some_node = Node(game_state, None, None, possible_moves)

        for move_index,first_move in enumerate(possible_moves):
            if first_move.command == MoveEnum.END_TURN:
                continue

            new_game_state, new_moves = game_state.apply_move(first_move)

            if new_game_state.end_game_state is not None:  # check if game is over, if we win we are fine with this move
                if new_game_state.end_game_state.winner == self.my_id:
                    return move_index

            if len(new_moves) == 1 and new_moves[0].command == MoveEnum.END_TURN:  # if there are no moves possible then lets just check value of this game state
                curr_val = some_node.heuristic(new_game_state)
                if curr_val > best_move_val:
                    best_move = first_move
                    best_move_val = curr_val
                    chosen_move_index = move_index

            for second_move in new_moves:
                if second_move.command == MoveEnum.END_TURN:
                    continue

                second_game_state, second_moves = new_game_state.apply_move(second_move)

                if second_game_state.end_game_state is not None:
                    if second_game_state.end_game_state.winner == self.my_id:
                        return move_index
                    
                curr_val = some_node.heuristic(second_game_state)
                if curr_val > best_move_val:
                    best_move = first_move
                    best_move_val = curr_val
                    chosen_move_index = move_index

                for third_move in second_moves:
                    if third_move.command == MoveEnum.END_TURN:
                        continue
                    third_game_state, _ = second_game_state.apply_move(third_move)

                    if third_game_state.end_game_state is not None:
                        if third_game_state.end_game_state.winner == self.my_id:
                            return first_move
                        
                    curr_val = some_node.heuristic(third_game_state)
                    if curr_val > best_move_val:
                        best_move = first_move
                        best_move_val = curr_val
                        chosen_move_index = move_index

        if best_move is not None:
            return chosen_move_index
        return 0

    def what_would_maxbot_do(self,game_state:GameState, possible_moves):

         ### There are cases that are no real 'decisions' so waist no time and store no state for training ###
        if len(possible_moves) == 1 and possible_moves[0].command == MoveEnum.END_TURN:
                return 0
        
        # If hand had gold or 'writ' cards, just play them. Saves on compute time
        if MoveEnum.PLAY_CARD in [m.command for m in possible_moves]:
            for i,m in enumerate(possible_moves):
                card_index = 0
                if m.command == MoveEnum.PLAY_CARD:
                    # This relies on the moves being in the same order as the cards in hand
                    card_name = game_state.current_player.hand[card_index].name
                    if card_name in ['Gold', 'Writ of Coin']:
                        return i
                    card_index +=1

            



        # Tracking for training
        #self.visited_states.append(self.convert_state_to_embedding(game_state, raw=True))

        best_move = None
        best_move_val = -99
        chosen_move_index = -999
        for move_index,first_move in enumerate(possible_moves):
            if first_move.command == MoveEnum.END_TURN:
                continue

            new_game_state, new_moves = game_state.apply_move(first_move)

            if new_game_state.end_game_state is not None:  # check if game is over, if we win we are fine with this move
                if new_game_state.end_game_state.winner == self.my_id:
                    return move_index

            if len(new_moves) == 1 and new_moves[0].command == MoveEnum.END_TURN:  # if there are no moves possible then lets just check value of this game state
                curr_val = new_game_state.current_player.prestige + new_game_state.current_player.power
                if curr_val > best_move_val:
                    best_move = first_move
                    best_move_val = curr_val
                    chosen_move_index = move_index

            for second_move in new_moves:
                if second_move.command == MoveEnum.END_TURN:
                    continue

                second_game_state, second_moves = new_game_state.apply_move(second_move)

                if second_game_state.end_game_state is not None:
                    if second_game_state.end_game_state.winner == self.my_id:
                        return move_index
                    
                curr_val = second_game_state.current_player.prestige + second_game_state.current_player.power
                if curr_val > best_move_val:
                    best_move = first_move
                    best_move_val = curr_val
                    chosen_move_index = move_index

                for third_move in second_moves:
                    if third_move.command == MoveEnum.END_TURN:
                        continue
                    third_game_state, _ = second_game_state.apply_move(third_move)

                    if third_game_state.end_game_state is not None:
                        if third_game_state.end_game_state.winner == self.my_id:
                            return first_move
                        
                    curr_val = third_game_state.current_player.prestige + third_game_state.current_player.power
                    if curr_val > best_move_val:
                        best_move = first_move
                        best_move_val = curr_val
                        chosen_move_index = move_index

        if best_move is not None:
            return chosen_move_index
        return 0

    def what_would_MCTS_do(self,game_state:GameState, possible_moves):

        if len(possible_moves) == 1 and possible_moves[0].command == MoveEnum.END_TURN:
            return 0
        
        if self.used_time_in_turn + self.time_for_move_computation >= self.turn_timeout:
                move = self.rng.choice(possible_moves)
        else:
            self.act_root.father = None
            start_time = time.monotonic()
            
            while time.monotonic() - start_time < self.time_for_move_computation.total_seconds():
                self.act_node = self._tree_policy(self.act_root)
                delta = self.act_node.simulate(self.rng)
                self._back_up(self.act_node, delta)

            self.used_time_in_turn += self.time_for_move_computation
            
            self.act_root = self.act_root.select_best_child()
            move = self.act_root.move

        chosen_move_index = [i for i, m in enumerate(possible_moves) if  m.move_id == move.move_id][0]
        return chosen_move_index
    
   
       