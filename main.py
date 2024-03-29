from fastapi import FastAPI, HTTPException, Depends, WebSocket, \
    WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from asyncio import Event, get_event_loop, wait_for, CancelledError
from multiprocessing.pool import Pool
from json import loads
from math import inf
from timeit import default_timer
from config import settings
from tic_tac_toe import tic_tac_toe
from connect_four import connect_four


app = FastAPI()

origins = [
  "http://localhost:8080",
  "https://minimax-algorithm.netlify.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
)

curr_ws_connections = 0
curr_workers = 0
curr_workers_change = Event()


def validate_tic_tac_toe_board(board: str):
    board = tuple(board)
    if len(board) != 9:
        raise HTTPException(
            status_code=400,
            detail="len(board) != 9")
    for tile in board:
        if tile not in ['x', 'o', '_']:
            raise HTTPException(
                status_code=400,
                detail="tile not in ['x', 'o', '_']")
    x_count = board.count('x')
    o_count = board.count('o')
    if x_count != o_count and x_count != o_count + 1:
        raise HTTPException(
            status_code=400,
            detail="x_count != o_count and x_count != o_count + 1")
    return board


def validate_connect_four_board(board: str):
    columns = board.split(",")
    if len(columns) != 7:
        raise HTTPException(
            status_code=400,
            detail="len(columns) != 7")
    y_count = 0
    r_count = 0
    for column in columns:
        if len(column) > 6:
            raise HTTPException(
                status_code=400,
                detail="len(column) > 6")
        for token in column:
            if token == 'y':
                y_count += 1
            elif token == 'r':
                r_count += 1
            else:
                raise HTTPException(
                    status_code=400,
                    detail="token not in ['y', 'r']")
    if y_count != r_count and y_count != r_count + 1:
        raise HTTPException(
            status_code=400,
            detail="y_count != r_count and y_count != r_count + 1")
    return board


def encode_connect_four_board(board: str):
    columns = board.split(",")
    yellow_tokens = 0
    token_mask = 0
    for column_index, column in enumerate(columns):
        for token_index, token in enumerate(column):
            if token == 'y':
                yellow_tokens |= 1 << (column_index * 7
                                       + token_index)
                token_mask |= yellow_tokens
            elif token == 'r':
                token_mask |= 1 << (column_index * 7
                                    + token_index)
    return yellow_tokens, token_mask


def validate_depth_limit(depth_limit: bool = False,
                         depth_limit_value: int = None):
    if not depth_limit:
        return None
    if depth_limit and depth_limit_value is None:
        raise HTTPException(
            status_code=400,
            detail="depth_limit_value can't be \
                'None' when depth_limit is 'True'")
    if depth_limit and not (depth_limit_value >= 1
                            and depth_limit_value <= 25):
        raise HTTPException(
            status_code=400,
            detail="depth_limit_value can't be \
                smaller than 1 or greater than 25")
    return depth_limit_value


async def apply_async_task(ws, func, *args):
    global curr_workers

    while not curr_workers < settings.worker_limit:
        await ws.send_json({"status": "waiting"})
        await curr_workers_change.wait()

    curr_workers_change.clear()
    curr_workers += 1

    pool = Pool(1)
    loop = get_event_loop()
    future = loop.create_future()

    def future_set_result(result):
        future.set_result(result)
    pool.apply_async(func, args, callback=future_set_result)
    await ws.send_json({"status": "running"})

    try:
        result = await wait_for(
            future, timeout=settings.task_timeout)

        pool.close()
        curr_workers -= 1
        curr_workers_change.set()
        await ws.send_json({"status": "complete",
                            "evaluations": result[0],
                            "evaluated_nodes": result[1]})

        print(f"Finished: {result}")

    except CancelledError:
        pool.terminate()
        curr_workers -= 1
        curr_workers_change.set()

        print("Cancelled!")
        raise

    except TimeoutError:
        pool.terminate()
        curr_workers -= 1
        curr_workers_change.set()
        await ws.send_json({"status": "timeout"})

        print("Timeout!")
        raise


@app.get("/heuristic_function_tic_tac_toe/{board}")
async def heuristic_function_tic_tac_toe(
    board: str = Depends(validate_tic_tac_toe_board),
):
    h = tic_tac_toe.heuristic(board)
    return {"estimation": h}


@app.get("/heuristic_function_connect_four/{board}")
async def heuristic_function_connect_four(
    board: str = Depends(validate_connect_four_board),
):
    yellow_tokens, token_mask = encode_connect_four_board(board)
    _, h = connect_four.heuristic(yellow_tokens, token_mask, 0)
    return {"estimation": h}


@app.websocket("/ws")
async def ws_endpoint(
    ws: WebSocket,
):
    curr_task = None
    global curr_ws_connections
    loop = get_event_loop()

    if not curr_ws_connections < settings.ws_connection_limit:
        await ws.accept()
        await ws.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await ws.accept()
    curr_ws_connections += 1
    print(f"Number of connections: {curr_ws_connections}")

    try:
        while True:
            message = await ws.receive_text()
            data = loads(message)

            if data["type"] == "tic_tac_toe":
                if curr_task is not None:
                    curr_task.cancel()
                curr_task = loop.create_task(
                    apply_async_task(
                        ws, evaluate_tic_tac_toe, data))

            elif data["type"] == "connect_four":
                if curr_task is not None:
                    curr_task.cancel()
                curr_task = loop.create_task(
                    apply_async_task(
                        ws, evaluate_connect_four, data))

            elif data["type"] == "cancel_task":
                if curr_task is not None:
                    curr_task.cancel()
                    curr_task = None

    except WebSocketDisconnect:
        if curr_task is not None:
            curr_task.cancel()
        curr_ws_connections -= 1
        print(f"Number of connections: {curr_ws_connections}")


def evaluate_tic_tac_toe(data):
    start_time = default_timer()

    board = validate_tic_tac_toe_board(data["board"])
    alpha_beta_pruning: bool = data["alpha_beta_pruning"]
    depth_limit: bool = data["depth_limit"]
    depth_limit_value = validate_depth_limit(
        data["depth_limit"], data["depth_limit_value"])

    x_count = board.count('x')
    o_count = board.count('o')

    evaluations = []
    evaluated_nodes = 0

    if not alpha_beta_pruning and not depth_limit:
        if x_count == o_count:
            for s in tic_tac_toe.successor(board, True):
                res_eval, res_nodes = tic_tac_toe.minimax(s, False)
                evaluations.append(float("{:.2f}".format(res_eval)))
                evaluated_nodes += res_nodes
        else:
            for s in tic_tac_toe.successor(board, False):
                res_eval, res_nodes = tic_tac_toe.minimax(s, True)
                evaluations.append(float("{:.2f}".format(res_eval)))
                evaluated_nodes += res_nodes

    if alpha_beta_pruning and not depth_limit:
        if x_count == o_count:
            for s in tic_tac_toe.successor(board, True):
                res_eval, res_nodes = tic_tac_toe.minimax_alpha_beta(
                    s, False, -inf, inf)
                evaluations.append(float("{:.2f}".format(res_eval)))
                evaluated_nodes += res_nodes
        else:
            for s in tic_tac_toe.successor(board, False):
                res_eval, res_nodes = tic_tac_toe.minimax_alpha_beta(
                    s, True, -inf, inf)
                evaluations.append(float("{:.2f}".format(res_eval)))
                evaluated_nodes += res_nodes

    if not alpha_beta_pruning and depth_limit:
        if x_count == o_count:
            for s in tic_tac_toe.successor(board, True):
                res_eval, res_nodes = \
                    tic_tac_toe.depth_limited_minimax(
                        s, depth_limit_value - 1, False)
                evaluations.append(float("{:.2f}".format(res_eval)))
                evaluated_nodes += res_nodes
        else:
            for s in tic_tac_toe.successor(board, False):
                res_eval, res_nodes = \
                    tic_tac_toe.depth_limited_minimax(
                        s, depth_limit_value - 1, True)
                evaluations.append(float("{:.2f}".format(res_eval)))
                evaluated_nodes += res_nodes

    if alpha_beta_pruning and depth_limit:
        if x_count == o_count:
            for s in tic_tac_toe.successor(board, True):
                res_eval, res_nodes = \
                    tic_tac_toe.depth_limited_minimax_alpha_beta(
                        s, depth_limit_value - 1, False, -inf, inf)
                evaluations.append(float("{:.2f}".format(res_eval)))
                evaluated_nodes += res_nodes
        else:
            for s in tic_tac_toe.successor(board, False):
                res_eval, res_nodes = \
                    tic_tac_toe.depth_limited_minimax_alpha_beta(
                        s, depth_limit_value - 1, True, -inf, inf)
                evaluations.append(float("{:.2f}".format(res_eval)))
                evaluated_nodes += res_nodes

    print(f"\nExecution time: {default_timer() - start_time:.7f}")

    return evaluations, evaluated_nodes


def evaluate_connect_four(data):
    start_time = default_timer()

    board = validate_connect_four_board(data["board"])
    alpha_beta_pruning: bool = data["alpha_beta_pruning"]
    depth_limit: bool = data["depth_limit"]
    depth_limit_value = validate_depth_limit(
        data["depth_limit"], data["depth_limit_value"])

    yellow_tokens, token_mask = encode_connect_four_board(board)

    y_count = bin(yellow_tokens).count("1")
    r_count = bin(token_mask).count("1") - y_count

    evaluations = []
    evaluated_nodes = 0

    if not alpha_beta_pruning and not depth_limit:
        if y_count == r_count:
            for move in connect_four.possible_moves(token_mask):
                res_eval, res_nodes = connect_four.minimax(
                    yellow_tokens | move, token_mask | move, False)
                evaluations.append(float("{:.2f}".format(res_eval)))
                evaluated_nodes += res_nodes
        else:
            for move in connect_four.possible_moves(token_mask):
                res_eval, res_nodes = connect_four.minimax(
                    yellow_tokens, token_mask | move, True)
                evaluations.append(float("{:.2f}".format(res_eval)))
                evaluated_nodes += res_nodes

    if alpha_beta_pruning and not depth_limit:
        if y_count == r_count:
            for move in connect_four.possible_moves(token_mask):
                res_eval, res_nodes = \
                    connect_four.minimax_alpha_beta(
                        yellow_tokens | move, token_mask | move,
                        False, -inf, inf)
                evaluations.append(float("{:.2f}".format(res_eval)))
                evaluated_nodes += res_nodes
        else:
            for move in connect_four.possible_moves(token_mask):
                res_eval, res_nodes = \
                    connect_four.minimax_alpha_beta(
                        yellow_tokens, token_mask | move,
                        True, -inf, inf)
                evaluations.append(float("{:.2f}".format(res_eval)))
                evaluated_nodes += res_nodes

    if not alpha_beta_pruning and depth_limit:
        if y_count == r_count:
            for move in connect_four.possible_moves(token_mask):
                res_eval, res_nodes = \
                    connect_four.depth_limited_minimax(
                        yellow_tokens | move, token_mask | move,
                        depth_limit_value - 1, False)
                evaluations.append(float("{:.2f}".format(res_eval)))
                evaluated_nodes += res_nodes
        else:
            for move in connect_four.possible_moves(token_mask):
                res_eval, res_nodes = \
                    connect_four.depth_limited_minimax(
                        yellow_tokens, token_mask | move,
                        depth_limit_value - 1, True)
                evaluations.append(float("{:.2f}".format(res_eval)))
                evaluated_nodes += res_nodes

    if alpha_beta_pruning and depth_limit:
        if y_count == r_count:
            for move in connect_four.possible_moves(token_mask):
                res_eval, res_nodes = \
                    connect_four.depth_limited_minimax_alpha_beta(
                        yellow_tokens | move, token_mask | move,
                        depth_limit_value - 1, False, -inf, inf)
                evaluations.append(float("{:.2f}".format(res_eval)))
                evaluated_nodes += res_nodes
        else:
            for move in connect_four.possible_moves(token_mask):
                res_eval, res_nodes = \
                    connect_four.depth_limited_minimax_alpha_beta(
                        yellow_tokens, token_mask | move,
                        depth_limit_value - 1, True, -inf, inf)
                evaluations.append(float("{:.2f}".format(res_eval)))
                evaluated_nodes += res_nodes

    print(f"\nExecution time: {default_timer() - start_time:.7f}")

    return evaluations, evaluated_nodes
