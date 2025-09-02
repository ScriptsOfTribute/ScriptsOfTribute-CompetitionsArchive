import random

from scripts_of_tribute.base_ai import BaseAI
from scripts_of_tribute.board import EndGameState, GameState, BasicMove
from scripts_of_tribute.enums import PlayerEnum, MoveEnum, PatronId

# tournament addition
from ..extensions import safe_play


class ZMyBot(BaseAI):

    def __init__(self, bot_name):
        super().__init__(bot_name)
        self.player_id: PlayerEnum = PlayerEnum.NO_PLAYER_SELECTED
        self.start_of_game: bool = True
    
    def select_patron(self, available_patrons):
        limited_patron_list = [0,1,2,4,6,9]
        selected_patrons = [p for p in available_patrons if p.value in limited_patron_list]
        

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
    def play(self, game_state: GameState, possible_moves, remaining_time) -> BasicMove:

        best_move = None
        best_move_val = -99
        if self.start_of_game:
            self.player_id = game_state.current_player.player_id
            self.start_of_game = False

        for first_move in possible_moves:
            if first_move.command == MoveEnum.END_TURN:
                continue
            new_game_state, new_moves = game_state.apply_move(first_move)

            if new_game_state.end_game_state is not None:  # check if game is over, if we win we are fine with this move
                if new_game_state.end_game_state.winner == self.player_id:
                    return first_move

            if len(new_moves) == 1 and new_moves[0].command == MoveEnum.END_TURN:  # if there are no moves possible then lets just check value of this game state
                curr_val = new_game_state.current_player.prestige + new_game_state.current_player.power
                if curr_val > best_move_val:
                    best_move = first_move
                    best_move_val = curr_val

            for second_move in new_moves:
                if second_move.command == MoveEnum.END_TURN:
                    continue
                second_game_state, second_moves = new_game_state.apply_move(second_move)
                if second_game_state.end_game_state is not None:
                    if second_game_state.end_game_state.winner == self.player_id:
                        return first_move
                curr_val = second_game_state.current_player.prestige + second_game_state.current_player.power
                if curr_val > best_move_val:
                    best_move = first_move
                    best_move_val = curr_val

            
                for third_move in second_moves:
                    if third_move.command == MoveEnum.END_TURN:
                        continue
                    third_game_state, third_moves = second_game_state.apply_move(third_move)
                    if third_game_state.end_game_state is not None:
                        if third_game_state.end_game_state.winner == self.player_id:
                            return first_move
                    curr_val = third_game_state.current_player.prestige + third_game_state.current_player.power
                    if curr_val > best_move_val:
                        best_move = first_move
                        best_move_val = curr_val


                


        if best_move is None:
            return next(move for move in possible_moves if move.command == MoveEnum.END_TURN)
        return best_move
    
    def game_end(self, end_game_state: EndGameState, final_state: GameState):
        pass

