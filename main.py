from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tic_tac_toe import tic_tac_toe
from connect_four import connect_four
from math import inf

from timeit import default_timer
from memory_profiler import memory_usage
from gc import collect

app = FastAPI()

origins = [
  "http://localhost:8080",
  "https://minimax-algorithm.netlify.app/",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # origins,
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)


def validate_tic_tac_toe_board(board: str):
    board = tuple(board)
    if len(board) != 9:
        raise HTTPException(status_code=400, detail="len(board) != 9")
    for tile in board:
        if tile not in ['x', 'o', '_']:
            raise HTTPException(status_code=400, detail="tile not in ['x', 'o', '_']")
    x_count = board.count('x')
    o_count = board.count('o')
    if x_count != o_count and x_count != o_count + 1:
        raise HTTPException(status_code=400, detail="x_count != o_count and x_count != o_count + 1")
    return board


def validate_connect_four_board(board: str):
    columns = board.split(",")
    if len(columns) != 7:
        raise HTTPException(status_code=400, detail="len(columns) != 7")
    y_count = 0
    r_count = 0
    for column in columns:
        if len(column) > 6:
            raise HTTPException(status_code=400, detail="len(column) > 6")
        for token in column:
            if token == 'y':
                y_count += 1
            elif token == 'r':
                r_count += 1
            else:
                raise HTTPException(status_code=400, detail="token not in ['y', 'r']")
    if y_count != r_count and y_count != r_count + 1:
        raise HTTPException(status_code=400, detail="y_count != r_count and y_count != r_count + 1")
    return board


def interpret_connect_four_board(board: str):
    columns = board.split(",")
    yellow_tokens = 0
    token_mask = 0
    for column_index, column in enumerate(columns):
        for token_index, token in enumerate(column):
            if token == 'y':
                yellow_tokens |= 1 << (column_index * 7 + token_index)
                token_mask |= yellow_tokens
            elif token == 'r':
                token_mask |= 1 << (column_index * 7 + token_index)
    return yellow_tokens, token_mask


def validate_depth_limit(depth_limit: bool = False, depth_limit_value: int = None):
    if not depth_limit:
        return None
    if depth_limit and depth_limit_value is None:
        raise HTTPException(status_code=400, detail="depth_limit_value can't be 'None' when depth_limit is 'True'")
    if depth_limit and not (depth_limit_value >= 1 and depth_limit_value <= 25):
        raise HTTPException(status_code=400, detail="depth_limit_value can't be smaller than 1 or greater than 25")
    return depth_limit_value


@app.get("/status")
async def return_status():
    return {"status": "online"}


@app.get("/tic_tac_toe/{board}")
async def evaluate_tic_tac_toe(
    board: str = Depends(validate_tic_tac_toe_board),
    alpha_beta_pruning: bool = False,
    depth_limit: bool = False,
    depth_limit_value: int = Depends(validate_depth_limit),
):
    start_time = default_timer()

    x_count = board.count('x')
    o_count = board.count('o')

    if not alpha_beta_pruning and not depth_limit:
        if x_count == o_count:
            evaluations = [tic_tac_toe.minimax(s, False)
                           for s in tic_tac_toe.successor(board, True)]
        else:
            evaluations = [tic_tac_toe.minimax(s, True)
                           for s in tic_tac_toe.successor(board, False)]

        mem_usage = memory_usage((tic_tac_toe.minimax, (board, True)))
        max_mem_usage = max(mem_usage)
        print(f"Memory usage: {max_mem_usage:.2f} MB")

        cache_info = tic_tac_toe.minimax.cache_info()
        tic_tac_toe.minimax.cache_clear()
    if alpha_beta_pruning and not depth_limit:
        if x_count == o_count:
            evaluations = [tic_tac_toe.minimax_alpha_beta(s, False, -inf, inf)
                           for s in tic_tac_toe.successor(board, True)]
        else:
            evaluations = [tic_tac_toe.minimax_alpha_beta(s, True, -inf, inf)
                           for s in tic_tac_toe.successor(board, False)]

        mem_usage = memory_usage((tic_tac_toe.minimax_alpha_beta, (board, True, -inf, inf)))
        max_mem_usage = max(mem_usage)
        print(f"Memory usage: {max_mem_usage:.2f} MB")

        cache_info = tic_tac_toe.minimax_alpha_beta.cache_info()
        tic_tac_toe.minimax_alpha_beta.cache_clear()
    if not alpha_beta_pruning and depth_limit:
        if x_count == o_count:
            evaluations = [tic_tac_toe.depth_limited_minimax(s, depth_limit_value - 1, False)
                           for s in tic_tac_toe.successor(board, True)]
        else:
            evaluations = [tic_tac_toe.depth_limited_minimax(s, depth_limit_value - 1, True)
                           for s in tic_tac_toe.successor(board, False)]

        # mem_usage = memory_usage((tic_tac_toe.depth_limited_minimax, (board, depth_limit_value,
        #                                                               True)))
        # max_mem_usage = max(mem_usage)
        # print(f"Memory usage: {max_mem_usage:.2f} MB")

        cache_info = tic_tac_toe.depth_limited_minimax.cache_info()
        tic_tac_toe.depth_limited_minimax.cache_clear()
    if alpha_beta_pruning and depth_limit:
        if x_count == o_count:
            evaluations = [tic_tac_toe.depth_limited_minimax_alpha_beta(s, depth_limit_value - 1, False, -inf, inf)
                           for s in tic_tac_toe.successor(board, True)]
        else:
            evaluations = [tic_tac_toe.depth_limited_minimax_alpha_beta(s, depth_limit_value - 1, True, -inf, inf)
                           for s in tic_tac_toe.successor(board, False)]

        # mem_usage = memory_usage((tic_tac_toe.depth_limited_minimax_alpha_beta, (board, depth_limit_value,
        #                                                                          True, -inf, inf)))
        # max_mem_usage = max(mem_usage)
        # print(f"Memory usage: {max_mem_usage:.2f} MB")

        cache_info = tic_tac_toe.depth_limited_minimax_alpha_beta.cache_info()
        tic_tac_toe.depth_limited_minimax_alpha_beta.cache_clear()

    print(f"\nCache: On\nExecution time: {default_timer() - start_time:.7f}")
    print(f"Evaluations: {evaluations}")

    print(f"Cache size: {cache_info.currsize}")
    print(f"Hits: {cache_info.hits}")
    print(f"Misses: {cache_info.misses}\n")

    collect()
    return {"evaluations": evaluations}


@app.get("/connect_four/{board}")
async def evaluate_connect_four(
    board: str = Depends(validate_connect_four_board),
    alpha_beta_pruning: bool = False,
    depth_limit: bool = False,
    depth_limit_value: int = Depends(validate_depth_limit),
):
    yellow_tokens, token_mask = interpret_connect_four_board(board)

    start_time = default_timer()

    y_count = bin(yellow_tokens).count("1")
    r_count = bin(token_mask).count("1") - y_count

    if not alpha_beta_pruning and not depth_limit:
        if y_count == r_count:
            evaluations = [connect_four.minimax(yellow_tokens | move, token_mask | move, False)
                           for move in connect_four.possible_moves(token_mask)]
        else:
            evaluations = [connect_four.minimax(yellow_tokens, token_mask | move, True)
                           for move in connect_four.possible_moves(token_mask)]

        # mem_usage = 0
        # for move in connect_four.possible_moves(token_mask):
        #     mem_usage = max(
        #         mem_usage,
        #         max(memory_usage((connect_four.minimax,
        #                           (yellow_tokens | move, token_mask | move, False))))
        #     )
        # print(f"Memory usage: {mem_usage:.2f} MB")

        cache_info = connect_four.minimax.cache_info()
        connect_four.minimax.cache_clear()
    if alpha_beta_pruning and not depth_limit:
        if y_count == r_count:
            evaluations = [connect_four.minimax_alpha_beta(yellow_tokens | move, token_mask | move, False, -inf, inf)
                           for move in connect_four.possible_moves(token_mask)]
        else:
            evaluations = [connect_four.minimax_alpha_beta(yellow_tokens, token_mask | move, True, -inf, inf)
                           for move in connect_four.possible_moves(token_mask)]

        # mem_usage = 0
        # for move in connect_four.possible_moves(token_mask):
        #     mem_usage = max(
        #         mem_usage,
        #         max(memory_usage((connect_four.minimax_alpha_beta,
        #                           (yellow_tokens | move, token_mask | move, False, -inf, inf))))
        #     )
        # print(f"Memory usage: {mem_usage:.2f} MB")

        cache_info = connect_four.minimax_alpha_beta.cache_info()
        connect_four.minimax_alpha_beta.cache_clear()
    if not alpha_beta_pruning and depth_limit:
        if y_count == r_count:
            evaluations = [connect_four.depth_limited_minimax(yellow_tokens | move, token_mask | move,
                                                              depth_limit_value - 1, False)
                           for move in connect_four.possible_moves(token_mask)]
        else:
            evaluations = [connect_four.depth_limited_minimax(yellow_tokens, token_mask | move,
                                                              depth_limit_value - 1, True)
                           for move in connect_four.possible_moves(token_mask)]

        # mem_usage = 0
        # for move in connect_four.possible_moves(token_mask):
        #     mem_usage = max(
        #         mem_usage,
        #         max(memory_usage((connect_four.minimax,
        #                           (yellow_tokens | move, token_mask | move, depth_limit_value - 1,
        #                            False))))
        #     )
        # print(f"Memory usage: {mem_usage:.2f} MB")

        cache_info = connect_four.depth_limited_minimax.cache_info()
        connect_four.depth_limited_minimax.cache_clear()
    if alpha_beta_pruning and depth_limit:
        if y_count == r_count:
            evaluations = [connect_four.depth_limited_minimax_alpha_beta(yellow_tokens | move, token_mask | move,
                                                                         depth_limit_value - 1, False, -inf, inf)
                           for move in connect_four.possible_moves(token_mask)]
        else:
            evaluations = [connect_four.depth_limited_minimax_alpha_beta(yellow_tokens, token_mask | move,
                                                                         depth_limit_value - 1, True, -inf, inf)
                           for move in connect_four.possible_moves(token_mask)]

        # mem_usage = 0
        # for move in connect_four.possible_moves(token_mask):
        #     mem_usage = max(
        #         mem_usage,
        #         max(memory_usage((connect_four.depth_limited_minimax_alpha_beta,
        #                           (yellow_tokens | move, token_mask | move, depth_limit_value - 1,
        #                            False, -inf, inf))))
        #     )
        # print(f"Memory usage: {mem_usage:.2f} MB")

        cache_info = connect_four.depth_limited_minimax_alpha_beta.cache_info()
        connect_four.depth_limited_minimax_alpha_beta.cache_clear()

    print(f"\nCache: On\nExecution time: {default_timer() - start_time:.7f}")
    print(f"Evaluations: {evaluations}")

    print(f"Cache size: {cache_info.currsize}")
    print(f"Hits: {cache_info.hits}")
    print(f"Misses: {cache_info.misses}\n")

    collect()
    return {"evaluations": evaluations}
