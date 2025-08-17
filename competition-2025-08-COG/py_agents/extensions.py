from functools import wraps

def safe_play(fallback="last"):
    """
    Decorator for play() to catch exceptions and return a fallback move.
    fallback:
        "last"  -> returns last move in possible_moves
        "first" -> returns first move in possible_moves
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, game_state, possible_moves, remaining_time):
            try:
                return func(self, game_state, possible_moves, remaining_time)
            except Exception as e:
                print(f"[safe_play] Exception in {func.__name__}: {e}")
                if not possible_moves:
                    raise  # No moves to return
                if fallback == "last":
                    return possible_moves[-1]
                elif fallback == "first":
                    return possible_moves[0]
                else:
                    return possible_moves[-1]
        return wrapper
    return decorator