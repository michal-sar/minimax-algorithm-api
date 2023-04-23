from functools import lru_cache
from math import inf


# minimax

@lru_cache(maxsize=None)
def minimax(yellow_tokens: int,
            token_mask: int,
            maximizer_turn: bool):
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
            res_eval, res_nodes = minimax(
                successor_yellow_tokens, successor_mask, False)
            evaluated_nodes += res_nodes
            v = max(v, res_eval)
        return v, evaluated_nodes + 1
    else:
        is_final_state, u = utility(yellow_tokens, token_mask)
        if is_final_state:
            return u, 1
        v = inf
        evaluated_nodes = 0
        for move in possible_moves(token_mask):
            successor_mask = token_mask | move
            res_eval, res_nodes = minimax(
                yellow_tokens, successor_mask, True)
            evaluated_nodes += res_nodes
            v = min(v, res_eval)
        return v, evaluated_nodes + 1


# minimax_alpha_beta

@lru_cache(maxsize=None)
def minimax_alpha_beta(yellow_tokens: int,
                       token_mask: int,
                       maximizer_turn: bool,
                       alpha: int, beta: int):
    if maximizer_turn:
        red_tokens = ~yellow_tokens & token_mask
        is_final_state, u = utility(red_tokens, token_mask)
        if is_final_state:
            return -u, 1
        evaluated_nodes = 0
        for move in possible_moves(token_mask):
            successor_yellow_tokens = yellow_tokens | move
            successor_token_mask = token_mask | move
            res_eval, res_nodes = minimax_alpha_beta(
                successor_yellow_tokens, successor_token_mask,
                False, alpha, beta)
            evaluated_nodes += res_nodes
            alpha = max(alpha, res_eval)
            if alpha >= beta:
                return alpha, evaluated_nodes + 1
        return alpha, evaluated_nodes + 1
    else:
        is_final_state, u = utility(yellow_tokens, token_mask)
        if is_final_state:
            return u, 1
        evaluated_nodes = 0
        for move in possible_moves(token_mask):
            successor_token_mask = token_mask | move
            res_eval, res_nodes = minimax_alpha_beta(
                yellow_tokens, successor_token_mask,
                True, alpha, beta)
            evaluated_nodes += res_nodes
            beta = min(beta, res_eval)
            if alpha >= beta:
                return beta, evaluated_nodes + 1
        return beta, evaluated_nodes + 1


# depth_limited_minimax

@lru_cache(maxsize=None)
def depth_limited_minimax(yellow_tokens: int,
                          token_mask: int, d: int,
                          maximizer_turn: bool):
    if maximizer_turn:
        red_tokens = ~yellow_tokens & token_mask
        is_final_state, h = heuristic(red_tokens, token_mask, d)
        if d == 0 or is_final_state:
            return -h, 1
        v = -inf
        evaluated_nodes = 0
        for move in possible_moves(token_mask):
            successor_yellow_tokens = yellow_tokens | move
            successor_mask = token_mask | move
            res_eval, res_nodes = depth_limited_minimax(
                successor_yellow_tokens, successor_mask,
                d - 1, False)
            evaluated_nodes += res_nodes
            v = max(v, res_eval)
        return v, evaluated_nodes + 1
    else:
        is_final_state, h = heuristic(yellow_tokens, token_mask, d)
        if d == 0 or is_final_state:
            return h, 1
        v = inf
        evaluated_nodes = 0
        for move in possible_moves(token_mask):
            successor_mask = token_mask | move
            res_eval, res_nodes = depth_limited_minimax(
                yellow_tokens, successor_mask,
                d - 1, True)
            evaluated_nodes += res_nodes
            v = min(v, res_eval)
        return v, evaluated_nodes + 1


# depth_limited_minimax_alpha_beta

@lru_cache(maxsize=None)
def depth_limited_minimax_alpha_beta(yellow_tokens: int,
                                     token_mask: int, d: int,
                                     maximizer_turn: bool,
                                     alpha: int, beta: int):
    if maximizer_turn:
        red_tokens = ~yellow_tokens & token_mask
        is_final_state, h = heuristic(red_tokens, token_mask, d)
        if d == 0 or is_final_state:
            return -h, 1
        evaluated_nodes = 0
        for move in possible_moves(token_mask):
            successor_yellow_tokens = yellow_tokens | move
            successor_token_mask = token_mask | move
            res_eval, res_nodes = depth_limited_minimax_alpha_beta(
                successor_yellow_tokens, successor_token_mask,
                d - 1, False, alpha, beta)
            evaluated_nodes += res_nodes
            alpha = max(alpha, res_eval)
            if alpha >= beta:
                return alpha, evaluated_nodes + 1
        return alpha, evaluated_nodes + 1
    else:
        is_final_state, h = heuristic(yellow_tokens, token_mask, d)
        if d == 0 or is_final_state:
            return h, 1
        evaluated_nodes = 0
        for move in possible_moves(token_mask):
            successor_token_mask = token_mask | move
            res_eval, res_nodes = depth_limited_minimax_alpha_beta(
                yellow_tokens, successor_token_mask,
                d - 1, True, alpha, beta)
            evaluated_nodes += res_nodes
            beta = min(beta, res_eval)
            if alpha >= beta:
                return beta, evaluated_nodes + 1
        return beta, evaluated_nodes + 1


def heuristic(tokens: int, token_mask: int, d: int):
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
    if d != 0:
        return False, 0
    h = 0
    not_tokens = ~tokens
    opponent_tokens = ~tokens & token_mask
    not_opponent_tokens = ~opponent_tokens
    buffer = not_opponent_tokens & (not_opponent_tokens >> 1)
    buffer_vertical = buffer & (buffer >> 2)
    buffer = not_opponent_tokens & (not_opponent_tokens >> 7)
    buffer_horizontal = buffer & (buffer >> 14)
    buffer = not_opponent_tokens & (not_opponent_tokens >> 8)
    buffer_diagonal1 = buffer & (buffer >> 16)
    buffer = not_opponent_tokens & (not_opponent_tokens >> 6)
    buffer_diagonal2 = buffer & (buffer >> 12)
    buffer = not_tokens & (not_tokens >> 1)
    opponent_buffer_vertical = buffer & (buffer >> 2)
    buffer = not_tokens & (not_tokens >> 7)
    opponent_buffer_horizontal = buffer & (buffer >> 14)
    buffer = not_tokens & (not_tokens >> 8)
    opponent_buffer_diagonal1 = buffer & (buffer >> 16)
    buffer = not_tokens & (not_tokens >> 6)
    opponent_buffer_diagonal2 = buffer & (buffer >> 12)
    pattern_mask = (tokens | (tokens >> 1)) & 137412980756383
    pattern_mask &= (pattern_mask >> 2)
    pattern_mask &= buffer_vertical
    h += bin(pattern_mask).count('1')
    pattern_mask = tokens & (tokens >> 1)
    pattern_mask |= (pattern_mask >> 2)
    pattern_mask &= 31028737590151
    pattern_mask &= buffer_vertical
    h += bin(pattern_mask).count('1')
    pattern_mask = (tokens | (tokens >> 7)) & 2181708111807
    pattern_mask &= (pattern_mask >> 14)
    pattern_mask &= buffer_horizontal
    h += bin(pattern_mask).count('1')
    pattern_mask = tokens & (tokens >> 7)
    pattern_mask |= (pattern_mask >> 14)
    pattern_mask &= 133160895
    pattern_mask &= buffer_horizontal
    h += bin(pattern_mask).count('1')
    pattern_mask = (tokens | (tokens >> 8)) & 1073538912159
    pattern_mask &= (pattern_mask >> 16)
    pattern_mask &= buffer_diagonal1
    h += bin(pattern_mask).count('1')
    pattern_mask = tokens & (tokens >> 8)
    pattern_mask |= (pattern_mask >> 16)
    pattern_mask &= 14795655
    pattern_mask &= buffer_diagonal1
    h += bin(pattern_mask).count('1')
    pattern_mask = (tokens | (tokens >> 6)) & 2147077824318
    pattern_mask &= (pattern_mask >> 12)
    pattern_mask &= buffer_diagonal2
    h += bin(pattern_mask).count('1')
    pattern_mask = tokens & (tokens >> 6)
    pattern_mask |= (pattern_mask >> 12)
    pattern_mask &= 118365240
    pattern_mask &= buffer_diagonal2
    h += bin(pattern_mask).count('1')
    pattern_mask = (opponent_tokens | (opponent_tokens >> 1)) \
        & 137412980756383
    pattern_mask &= (pattern_mask >> 2)
    pattern_mask &= opponent_buffer_vertical
    h -= bin(pattern_mask).count('1')
    pattern_mask = opponent_tokens & (opponent_tokens >> 1)
    pattern_mask |= (pattern_mask >> 2)
    pattern_mask &= 31028737590151
    pattern_mask &= opponent_buffer_vertical
    h -= bin(pattern_mask).count('1')
    pattern_mask = (opponent_tokens | (opponent_tokens >> 7)) \
        & 2181708111807
    pattern_mask &= (pattern_mask >> 14)
    pattern_mask &= opponent_buffer_horizontal
    h -= bin(pattern_mask).count('1')
    pattern_mask = opponent_tokens & (opponent_tokens >> 7)
    pattern_mask |= (pattern_mask >> 14)
    pattern_mask &= 133160895
    pattern_mask &= opponent_buffer_horizontal
    h -= bin(pattern_mask).count('1')
    pattern_mask = (opponent_tokens | (opponent_tokens >> 8)) \
        & 1073538912159
    pattern_mask &= (pattern_mask >> 16)
    pattern_mask &= opponent_buffer_diagonal1
    h -= bin(pattern_mask).count('1')
    pattern_mask = opponent_tokens & (opponent_tokens >> 8)
    pattern_mask |= (pattern_mask >> 16)
    pattern_mask &= 14795655
    pattern_mask &= opponent_buffer_diagonal1
    h -= bin(pattern_mask).count('1')
    pattern_mask = (opponent_tokens | (opponent_tokens >> 6)) \
        & 2147077824318
    pattern_mask &= (pattern_mask >> 12)
    pattern_mask &= opponent_buffer_diagonal2
    h -= bin(pattern_mask).count('1')
    pattern_mask = opponent_tokens & (opponent_tokens >> 6)
    pattern_mask |= (pattern_mask >> 12)
    pattern_mask &= 118365240
    pattern_mask &= opponent_buffer_diagonal2
    h -= bin(pattern_mask).count('1')
    return False, h * 0.02


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
            yield ((((1 << 6) - 1) << index * 7) & token_mask) \
                + (1 << index * 7)
