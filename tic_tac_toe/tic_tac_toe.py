from functools import lru_cache
from math import inf


# minimax

@lru_cache(maxsize=None)
def minimax(n: tuple, maximizer_turn: bool):
    if is_final_state(n):
        return utility(n)
    if maximizer_turn:
        v = -inf
        for s in successor(n, True):
            v = max(v, minimax(s, False))
        return v
    else:
        v = inf
        for s in successor(n, False):
            v = min(v, minimax(s, True))
        return v


# minimax_alpha_beta

@lru_cache(maxsize=None)
def minimax_alpha_beta(n: tuple, maximizer_turn: bool, alpha: int, beta: int):
    if is_final_state(n):
        return utility(n)
    if maximizer_turn:
        for s in successor(n, True):
            alpha = max(alpha, minimax_alpha_beta(s, False, alpha, beta))
            if alpha >= beta:
                return alpha
        return alpha
    else:
        for s in successor(n, False):
            beta = min(beta, minimax_alpha_beta(s, True, alpha, beta))
            if alpha >= beta:
                return beta
        return beta


# depth_limited_minimax

@lru_cache(maxsize=None)
def depth_limited_minimax(n: tuple, d: int, maximizer_turn: bool):
    if is_final_state(n) or d == 0:
        return heuristic(n)
    if maximizer_turn:
        v = -inf
        for s in successor(n, True):
            v = max(v, depth_limited_minimax(s, d - 1, False))
        return v
    else:
        v = inf
        for s in successor(n, False):
            v = min(v, depth_limited_minimax(s, d - 1, True))
        return v


# depth_limited_minimax_alpha_beta

@lru_cache(maxsize=None)
def depth_limited_minimax_alpha_beta(n: tuple, d: int, maximizer_turn: bool, alpha: int, beta: int):
    if is_final_state(n) or d == 0:
        return heuristic(n)
    if maximizer_turn:
        for s in successor(n, True):
            alpha = max(alpha, depth_limited_minimax_alpha_beta(s, d - 1, False, alpha, beta))
            if alpha >= beta:
                return alpha
        return alpha
    else:
        for s in successor(n, False):
            beta = min(beta, depth_limited_minimax_alpha_beta(s, d - 1, True, alpha, beta))
            if alpha >= beta:
                return beta
        return beta


def is_final_state(n: tuple):
    if not n.count('_'):
        return True
    if (
        (n[0] != '_' and n[0] == n[1] and n[1] == n[2]) or
        (n[3] != '_' and n[3] == n[4] and n[4] == n[5]) or
        (n[6] != '_' and n[6] == n[7] and n[7] == n[8]) or
        (n[0] != '_' and n[0] == n[3] and n[3] == n[6]) or
        (n[1] != '_' and n[1] == n[4] and n[4] == n[7]) or
        (n[2] != '_' and n[2] == n[5] and n[5] == n[8]) or
        (n[0] != '_' and n[0] == n[4] and n[4] == n[8]) or
        (n[2] != '_' and n[2] == n[4] and n[4] == n[6])
    ):
        return True
    return False


def heuristic(n: tuple):
    if (
        (n[0] == 'x' and n[1] == 'x' and n[2] == 'x') or
        (n[3] == 'x' and n[4] == 'x' and n[5] == 'x') or
        (n[6] == 'x' and n[7] == 'x' and n[8] == 'x') or
        (n[0] == 'x' and n[3] == 'x' and n[6] == 'x') or
        (n[1] == 'x' and n[4] == 'x' and n[7] == 'x') or
        (n[2] == 'x' and n[5] == 'x' and n[8] == 'x') or
        (n[0] == 'x' and n[4] == 'x' and n[8] == 'x') or
        (n[2] == 'x' and n[4] == 'x' and n[6] == 'x')
    ):
        return 1
    if (
        (n[0] == 'o' and n[1] == 'o' and n[2] == 'o') or
        (n[3] == 'o' and n[4] == 'o' and n[5] == 'o') or
        (n[6] == 'o' and n[7] == 'o' and n[8] == 'o') or
        (n[0] == 'o' and n[3] == 'o' and n[6] == 'o') or
        (n[1] == 'o' and n[4] == 'o' and n[7] == 'o') or
        (n[2] == 'o' and n[5] == 'o' and n[8] == 'o') or
        (n[0] == 'o' and n[4] == 'o' and n[8] == 'o') or
        (n[2] == 'o' and n[4] == 'o' and n[6] == 'o')
    ):
        return -1
    if not n.count('_'):
        return 0
    # add heuristics here:
    return 0


def utility(n: tuple):
    if (
        (n[0] == 'x' and n[1] == 'x' and n[2] == 'x') or
        (n[3] == 'x' and n[4] == 'x' and n[5] == 'x') or
        (n[6] == 'x' and n[7] == 'x' and n[8] == 'x') or
        (n[0] == 'x' and n[3] == 'x' and n[6] == 'x') or
        (n[1] == 'x' and n[4] == 'x' and n[7] == 'x') or
        (n[2] == 'x' and n[5] == 'x' and n[8] == 'x') or
        (n[0] == 'x' and n[4] == 'x' and n[8] == 'x') or
        (n[2] == 'x' and n[4] == 'x' and n[6] == 'x')
    ):
        return 1
    if (
        (n[0] == 'o' and n[1] == 'o' and n[2] == 'o') or
        (n[3] == 'o' and n[4] == 'o' and n[5] == 'o') or
        (n[6] == 'o' and n[7] == 'o' and n[8] == 'o') or
        (n[0] == 'o' and n[3] == 'o' and n[6] == 'o') or
        (n[1] == 'o' and n[4] == 'o' and n[7] == 'o') or
        (n[2] == 'o' and n[5] == 'o' and n[8] == 'o') or
        (n[0] == 'o' and n[4] == 'o' and n[8] == 'o') or
        (n[2] == 'o' and n[4] == 'o' and n[6] == 'o')
    ):
        return -1
    if not n.count('_'):
        return 0


def successor(n: tuple, maximizer_turn: bool):
    if maximizer_turn:
        for tile_index, tile in enumerate(n):
            if tile == '_':
                s = n[:tile_index] + ('x',) + n[tile_index + 1:]
                yield s
    else:
        for tile_index, tile in enumerate(n):
            if tile == '_':
                s = n[:tile_index] + ('o',) + n[tile_index + 1:]
                yield s
