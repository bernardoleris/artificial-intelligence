import heapq
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import deque

class PuzzleState:
    def __init__(self, board, empty_pos, moves=0):
        self.board = board
        self.empty_pos = empty_pos
        self.moves = moves

    def __str__(self):
        return '\n'.join([' '.join(map(str, row)) for row in self.board])

    def is_goal(self):
        goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
        return self.board == goal_state

    def __lt__(self, other):
        return (self.board, self.moves) < (other.board, other.moves)

def get_possible_moves(state):
    moves = []
    x, y = state.empty_pos

    if x > 0:  # move empty tile up
        moves.append((x - 1, y))
    if x < 2:  # move empty tile down
        moves.append((x + 1, y))
    if y > 0:  # move empty tile left
        moves.append((x, y - 1))
    if y < 2:  # move empty tile right
        moves.append((x, y + 1))

    return moves

def apply_move(state, move):
    new_board = [row[:] for row in state.board]
    x, y = state.empty_pos
    new_x, new_y = move
    new_board[x][y], new_board[new_x][new_y] = new_board[new_x][new_y], new_board[x][y]
    return PuzzleState(new_board, (new_x, new_y), state.moves + 1)

def h1(state):
    """Number of misplaced tiles."""
    goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
    misplaced = 0
    for i in range(3):
        for j in range(3):
            if state.board[i][j] != 0 and state.board[i][j] != goal_state[i][j]:
                misplaced += 1
    return misplaced

def h2(state):
    """Manhattan distance."""
    goal_positions = {
        1: (0, 0), 2: (0, 1), 3: (0, 2),
        4: (1, 0), 5: (1, 1), 6: (1, 2),
        7: (2, 0), 8: (2, 1), 0: (2, 2)
    }
    distance = 0
    for i in range(3):
        for j in range(3):
            if state.board[i][j] != 0:
                goal_x, goal_y = goal_positions[state.board[i][j]]
                distance += abs(i - goal_x) + abs(j - goal_y)
    return distance

def a_star(initial_state, heuristic):
    open_list = []
    heapq.heappush(open_list, (heuristic(initial_state), 0, initial_state))
    closed_list = set()
    parents = {str(initial_state.board): None}

    while open_list:
        _, cost, current_state = heapq.heappop(open_list)
        
        if current_state.is_goal():
            return reconstruct_path(parents, current_state)

        closed_list.add(str(current_state.board))
        
        for move in get_possible_moves(current_state):
            neighbor = apply_move(current_state, move)
            if str(neighbor.board) in closed_list:
                continue

            new_cost = cost + 1
            heapq.heappush(open_list, (new_cost + heuristic(neighbor), new_cost, neighbor))
            parents[str(neighbor.board)] = current_state

    return None

def bfs(initial_state):
    open_list = deque([initial_state])
    closed_list = set()
    parents = {str(initial_state.board): None}

    while open_list:
        current_state = open_list.popleft()
        
        if current_state.is_goal():
            return reconstruct_path(parents, current_state)

        closed_list.add(str(current_state.board))
        
        for move in get_possible_moves(current_state):
            neighbor = apply_move(current_state, move)
            if str(neighbor.board) in closed_list:
                continue

            open_list.append(neighbor)
            parents[str(neighbor.board)] = current_state

    return None

def reconstruct_path(parents, state):
    path = []
    while state:
        path.append(state)
        state = parents[str(state.board)]
    return path[::-1]

def count_inversions(board):
    flat_board = [tile for row in board for tile in row if tile != 0]
    inversions = 0
    for i in range(len(flat_board)):
        for j in range(i + 1, len(flat_board)):
            if flat_board[i] > flat_board[j]:
                inversions += 1
    return inversions

def is_solvable(board):
    inversions = count_inversions(board)
    return inversions % 2 == 0

def generate_random_board():
    board = list(range(9))
    random.shuffle(board)
    board = [board[i:i + 3] for i in range(0, len(board), 3)]
    empty_pos = [(ix, iy) for ix, row in enumerate(board) for iy, i in enumerate(row) if i == 0][0]
    return board, empty_pos

def plot_board(state, title=""):
    board = state.board
    fig, ax = plt.subplots()
    ax.matshow(np.array(board) != 0, cmap='gray')

    for i in range(3):
        for j in range(3):
            c = board[i][j]
            if c != 0:
                ax.text(j, i, str(c), va='center', ha='center')

    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.show()

# Generate a random solvable board
board, empty_pos = generate_random_board()
while not is_solvable(board):
    board, empty_pos = generate_random_board()

initial_state = PuzzleState(board, empty_pos)

print("Initial State:")
print(initial_state)
plot_board(initial_state, "Initial State")

if is_solvable(board):
    path = bfs(initial_state)  # Use BFS instead of A*

    if path:
        # Remove the initial state from the path
        path = path[1:]
        for step in path:
            print()
            print(step)
            plot_board(step, f"Move {step.moves}")
    else:
        print("No solution found.")
else:
    print("The initial configuration is not solvable.")
