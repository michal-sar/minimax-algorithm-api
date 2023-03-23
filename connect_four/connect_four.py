from functools import lru_cache
from math import inf


# minimax

@lru_cache(maxsize=None)
def minimax(yellow_tokens: int, token_mask: int, maximizer_turn: bool):
    if maximizer_turn:
        red_tokens = ~yellow_tokens & token_mask
        is_final_state, u = utility(red_tokens, token_mask)
        if is_final_state:
            return -u, 1
        v = -inf
        evaluated_nodes = 0
        for move in possible_moves(token_mask):
            successor_yellow_tokens = yellow_tokens | move
            successor_mask = token_mask | move
            res_eval, res_nodes = minimax(successor_yellow_tokens, successor_mask, False)
            evaluated_nodes += res_nodes
            v = max(v, res_eval)
        return v, evaluated_nodes
    else:
        is_final_state, u = utility(yellow_tokens, token_mask)
        if is_final_state:
            return u, 1
        v = inf
        evaluated_nodes = 0
        for move in possible_moves(token_mask):
            successor_mask = token_mask | move
            res_eval, res_nodes = minimax(yellow_tokens, successor_mask, True)
            evaluated_nodes += res_nodes
            v = min(v, res_eval)
        return v, evaluated_nodes


# minimax_alpha_beta

@lru_cache(maxsize=None)
def minimax_alpha_beta(yellow_tokens: int, token_mask: int, maximizer_turn: bool, alpha: int, beta: int):
    if maximizer_turn:
        red_tokens = ~yellow_tokens & token_mask
        is_final_state, u = utility(red_tokens, token_mask)
        if is_final_state:
            return -u, 1
        evaluated_nodes = 0
        for move in possible_moves(token_mask):
            successor_yellow_tokens = yellow_tokens | move
            successor_token_mask = token_mask | move
            res_eval, res_nodes = minimax_alpha_beta(successor_yellow_tokens, successor_token_mask, False, alpha, beta)
            evaluated_nodes += res_nodes
            alpha = max(alpha, res_eval)
            if alpha >= beta:
                return alpha, evaluated_nodes
        return alpha, evaluated_nodes
    else:
        is_final_state, u = utility(yellow_tokens, token_mask)
        if is_final_state:
            return u, 1
        evaluated_nodes = 0
        for move in possible_moves(token_mask):
            successor_token_mask = token_mask | move
            res_eval, res_nodes = minimax_alpha_beta(yellow_tokens, successor_token_mask, True, alpha, beta)
            evaluated_nodes += res_nodes
            beta = min(beta, res_eval)
            if alpha >= beta:
                return beta, evaluated_nodes
        return beta, evaluated_nodes


# depth_limited_minimax

@lru_cache(maxsize=None)
def depth_limited_minimax(yellow_tokens: int, token_mask: int, d: int, maximizer_turn: bool):
    if maximizer_turn:
        red_tokens = ~yellow_tokens & token_mask
        is_final_state, h = heuristic(red_tokens, token_mask)
        if d == 0 or is_final_state:
            return -h, 1
        v = -inf
        evaluated_nodes = 0
        for move in possible_moves(token_mask):
            successor_yellow_tokens = yellow_tokens | move
            successor_mask = token_mask | move
            res_eval, res_nodes = depth_limited_minimax(successor_yellow_tokens, successor_mask, d - 1, False)
            evaluated_nodes += res_nodes
            v = max(v, res_eval)
        return v, evaluated_nodes
    else:
        is_final_state, h = heuristic(yellow_tokens, token_mask)
        if d == 0 or is_final_state:
            return h, 1
        v = inf
        evaluated_nodes = 0
        for move in possible_moves(token_mask):
            successor_mask = token_mask | move
            res_eval, res_nodes = depth_limited_minimax(yellow_tokens, successor_mask, d - 1, True)
            evaluated_nodes += res_nodes
            v = min(v, res_eval)
        return v, evaluated_nodes


# depth_limited_minimax_alpha_beta

@lru_cache(maxsize=None)
def depth_limited_minimax_alpha_beta(yellow_tokens: int, token_mask: int, d: int,
                                     maximizer_turn: bool, alpha: int, beta: int):
    if maximizer_turn:
        red_tokens = ~yellow_tokens & token_mask
        is_final_state, h = heuristic(red_tokens, token_mask)
        if d == 0 or is_final_state:
            return -h, 1
        evaluated_nodes = 0
        for move in possible_moves(token_mask):
            successor_yellow_tokens = yellow_tokens | move
            successor_token_mask = token_mask | move
            res_eval, res_nodes = depth_limited_minimax_alpha_beta(successor_yellow_tokens, successor_token_mask, d - 1,
                                                                   False, alpha, beta)
            evaluated_nodes += res_nodes
            alpha = max(alpha, res_eval)
            if alpha >= beta:
                return alpha, evaluated_nodes
        return alpha, evaluated_nodes
    else:
        is_final_state, h = heuristic(yellow_tokens, token_mask)
        if d == 0 or is_final_state:
            return h, 1
        evaluated_nodes = 0
        for move in possible_moves(token_mask):
            successor_token_mask = token_mask | move
            res_eval, res_nodes = depth_limited_minimax_alpha_beta(yellow_tokens, successor_token_mask, d - 1,
                                                                   True, alpha, beta)
            evaluated_nodes += res_nodes
            beta = min(beta, res_eval)
            if alpha >= beta:
                return beta, evaluated_nodes
        return beta, evaluated_nodes


def heuristic(tokens: int, token_mask: int):
    pattern_mask = tokens & (tokens >> 6)
    if pattern_mask & (pattern_mask >> 12):
        return True, 1
    pattern_mask = tokens & (tokens >> 7)
    if pattern_mask & (pattern_mask >> 14):
        return True, 1
    pattern_mask = tokens & (tokens >> 8)
    if pattern_mask & (pattern_mask >> 16):
        return True, 1
    pattern_mask = tokens & (tokens >> 1)
    if pattern_mask & (pattern_mask >> 2):
        return True, 1
    pattern_mask = 1
    for index in range(7):
        pattern_mask &= token_mask >> index * 7 + 5
    if pattern_mask == 1:
        return True, 0
    # add heuristics here:
    return False, 0


def utility(tokens: int, token_mask: int):
    pattern_mask = tokens & (tokens >> 6)
    if pattern_mask & (pattern_mask >> 12):
        return True, 1
    pattern_mask = tokens & (tokens >> 7)
    if pattern_mask & (pattern_mask >> 14):
        return True, 1
    pattern_mask = tokens & (tokens >> 8)
    if pattern_mask & (pattern_mask >> 16):
        return True, 1
    pattern_mask = tokens & (tokens >> 1)
    if pattern_mask & (pattern_mask >> 2):
        return True, 1
    pattern_mask = 1
    for index in range(7):
        pattern_mask &= token_mask >> index * 7 + 5
    if pattern_mask == 1:
        return True, 0
    return False, None


def possible_moves(token_mask: int):
    for index in range(7):
        if (token_mask >> index * 7 + 5) & 1 == 0:
            yield ((((1 << 6) - 1) << index * 7) & token_mask) + (1 << index * 7)
