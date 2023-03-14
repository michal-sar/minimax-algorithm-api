from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from tic_tac_toe import tic_tac_toe
from connect_four import connect_four
from multiprocessing.pool import Pool
from asyncio import get_event_loop, CancelledError
from json import loads
from math import inf

# from timeit import default_timer
# from gc import collect

app = FastAPI()

origins = [
  "http://localhost:8080",
  "https://minimax-algorithm.netlify.app/",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origins],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

websocket_connections = 0


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


async def apply_async_task(websocket, func, *args):
    pool = Pool(1)

    loop = get_event_loop()
    future = loop.create_future()

    def future_set_result(result):
        future.set_result(result)

    pool.apply_async(func, args, callback=future_set_result)

    try:
        # print("Waiting...")
        result = await future
        # print(f"Finished: {result}")
        await websocket.send_json({'evaluations': result[0], 'id': result[1]})
        # pool.terminate() <- ???

    except CancelledError:
        pool.terminate()
        # print("Cancelled!")
        raise


@app.websocket("/ws")
async def ws_endpoint(
    websocket: WebSocket,
):
    current_task = None
    global websocket_connections

    loop = get_event_loop()

    await websocket.accept()
    websocket_connections += 1
    # print(f"Number of connections: {websocket_connections}")
    try:
        while True:
            message = await websocket.receive_text()
            data = loads(message)

            if data['type'] == 'tic_tac_toe':
                if current_task is not None:
                    current_task.cancel()
                current_task = loop.create_task(apply_async_task(websocket, ws_evaluate_tic_tac_toe, data))

            elif data['type'] == 'connect_four':
                if current_task is not None:
                    current_task.cancel()
                current_task = loop.create_task(apply_async_task(websocket, ws_evaluate_connect_four, data))

            elif data['type'] == 'cancel_task':
                if current_task is not None:
                    current_task.cancel()
                    current_task = None

    except WebSocketDisconnect:
        if current_task is not None:
            current_task.cancel()
        websocket_connections -= 1
        # print(f"Number of connections: {websocket_connections}")


def ws_evaluate_tic_tac_toe(data):
    # start_time = default_timer()

    board = validate_tic_tac_toe_board(data['board'])
    alpha_beta_pruning: bool = data['alpha_beta_pruning']
    depth_limit: bool = data['depth_limit']
    depth_limit_value = validate_depth_limit(data['depth_limit'], data['depth_limit_value'])

    x_count = board.count('x')
    o_count = board.count('o')

    if not alpha_beta_pruning and not depth_limit:
        if x_count == o_count:
            evaluations = [tic_tac_toe.minimax(s, False)
                           for s in tic_tac_toe.successor(board, True)]
        else:
            evaluations = [tic_tac_toe.minimax(s, True)
                           for s in tic_tac_toe.successor(board, False)]

        # cache_info = tic_tac_toe.minimax.cache_info()
        tic_tac_toe.minimax.cache_clear()

    if alpha_beta_pruning and not depth_limit:
        if x_count == o_count:
            evaluations = [tic_tac_toe.minimax_alpha_beta(s, False, -inf, inf)
                           for s in tic_tac_toe.successor(board, True)]
        else:
            evaluations = [tic_tac_toe.minimax_alpha_beta(s, True, -inf, inf)
                           for s in tic_tac_toe.successor(board, False)]

        # cache_info = tic_tac_toe.minimax_alpha_beta.cache_info()
        tic_tac_toe.minimax_alpha_beta.cache_clear()

    if not alpha_beta_pruning and depth_limit:
        if x_count == o_count:
            evaluations = [tic_tac_toe.depth_limited_minimax(s, depth_limit_value - 1, False)
                           for s in tic_tac_toe.successor(board, True)]
        else:
            evaluations = [tic_tac_toe.depth_limited_minimax(s, depth_limit_value - 1, True)
                           for s in tic_tac_toe.successor(board, False)]

        # cache_info = tic_tac_toe.depth_limited_minimax.cache_info()
        tic_tac_toe.depth_limited_minimax.cache_clear()

    if alpha_beta_pruning and depth_limit:
        if x_count == o_count:
            evaluations = [tic_tac_toe.depth_limited_minimax_alpha_beta(s, depth_limit_value - 1, False, -inf, inf)
                           for s in tic_tac_toe.successor(board, True)]
        else:
            evaluations = [tic_tac_toe.depth_limited_minimax_alpha_beta(s, depth_limit_value - 1, True, -inf, inf)
                           for s in tic_tac_toe.successor(board, False)]

        # cache_info = tic_tac_toe.depth_limited_minimax_alpha_beta.cache_info()
        tic_tac_toe.depth_limited_minimax_alpha_beta.cache_clear()

    # print(f"\nCache: On\nExecution time: {default_timer() - start_time:.7f}")
    # print(f"Evaluations: {evaluations}")

    # print(f"Cache size: {cache_info.currsize}")
    # print(f"Hits: {cache_info.hits}")
    # print(f"Misses: {cache_info.misses}\n")

    # collect()
    return evaluations, data["id"]  # Background Tasks? <- cache_clear(), collect()


def ws_evaluate_connect_four(data):
    # start_time = default_timer()

    board = validate_connect_four_board(data['board'])
    alpha_beta_pruning: bool = data['alpha_beta_pruning']
    depth_limit: bool = data['depth_limit']
    depth_limit_value = validate_depth_limit(data['depth_limit'], data['depth_limit_value'])

    yellow_tokens, token_mask = interpret_connect_four_board(board)

    y_count = bin(yellow_tokens).count("1")
    r_count = bin(token_mask).count("1") - y_count

    if not alpha_beta_pruning and not depth_limit:
        if y_count == r_count:
            evaluations = [connect_four.minimax(yellow_tokens | move, token_mask | move, False)
                           for move in connect_four.possible_moves(token_mask)]
        else:
            evaluations = [connect_four.minimax(yellow_tokens, token_mask | move, True)
                           for move in connect_four.possible_moves(token_mask)]

        # cache_info = connect_four.minimax.cache_info()
        connect_four.minimax.cache_clear()

    if alpha_beta_pruning and not depth_limit:
        if y_count == r_count:
            evaluations = [connect_four.minimax_alpha_beta(yellow_tokens | move, token_mask | move, False, -inf, inf)
                           for move in connect_four.possible_moves(token_mask)]
        else:
            evaluations = [connect_four.minimax_alpha_beta(yellow_tokens, token_mask | move, True, -inf, inf)
                           for move in connect_four.possible_moves(token_mask)]

        # cache_info = connect_four.minimax_alpha_beta.cache_info()
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

        # cache_info = connect_four.depth_limited_minimax.cache_info()
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

        # cache_info = connect_four.depth_limited_minimax_alpha_beta.cache_info()
        connect_four.depth_limited_minimax_alpha_beta.cache_clear()

    # print(f"\nCache: On\nExecution time: {default_timer() - start_time:.7f}")
    # print(f"Evaluations: {evaluations}")

    # print(f"Cache size: {cache_info.currsize}")
    # print(f"Hits: {cache_info.hits}")
    # print(f"Misses: {cache_info.misses}\n")

    # collect()
    return evaluations, data["id"]  # Background Tasks? <- cache_clear(), collect()
