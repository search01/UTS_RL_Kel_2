import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

class Board:
    def __init__(self, size=5):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.current_player = 1

    def is_valid_move(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size and self.board[x][y] == 0

    def place_stone(self, x, y):
        if self.is_valid_move(x, y):
            self.board[x][y] = self.current_player
            self.current_player = 3 - self.current_player
            return True
        return False

    def check_winner(self):
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] != 0 and self.check_direction(i, j):
                    return self.board[i][j]
        return None

    def check_direction(self, x, y):
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dx, dy in directions:
            count = 1
            for step in range(1, 5):
                nx, ny = x + step * dx, y + step * dy
                if 0 <= nx < self.size and 0 <= ny < self.size and self.board[nx][ny] == self.board[x][y]:
                    count += 1
                else:
                    break
            if count >= 5:
                return True
        return False

    def get_available_moves(self):
        return [(x, y) for x in range(self.size) for y in range(self.size) if self.board[x][y] == 0]

    def display(self, ax):
        ax.clear()
        ax.imshow(self.board, cmap='gray', extent=[0, self.size, 0, self.size], alpha=0.2)
        ax.set_xticks(np.arange(0.5, self.size, 1))
        ax.set_yticks(np.arange(0.5, self.size, 1))
        ax.set_xticklabels(range(self.size))
        ax.set_yticklabels(range(self.size))
        ax.grid(True, linestyle='-', linewidth=0.5, color='black')

        # Display 'X' and 'O' on the board
        for x in range(self.size):
            for y in range(self.size):
                if self.board[x][y] == 1:
                    ax.text(y + 0.5, self.size - x - 0.5, 'X', ha='center', va='center', fontsize=24, color='blue')
                elif self.board[x][y] == 2:
                    ax.text(y + 0.5, self.size - x - 0.5, 'O', ha='center', va='center', fontsize=24, color='red')

        ax.invert_yaxis()
        plt.draw()

class PolicyValueNet:
    def __init__(self, board_size):
        self.board_size = board_size
        self.model = self.build_model()

    def build_model(self):
        inputs = layers.Input(shape=(self.board_size, self.board_size, 1))
        x = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')(inputs)
        x = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
        x = layers.Flatten()(x)
        policy = layers.Dense(self.board_size * self.board_size, activation='softmax')(x)
        value = layers.Dense(64, activation='relu')(x)
        value = layers.Dense(1, activation='tanh')(value)
        
        model = models.Model(inputs=inputs, outputs=[policy, value])
        model.compile(optimizer='adam', loss=['categorical_crossentropy', 'mean_squared_error'])
        return model

    def predict(self, board):
        board = np.array(board).reshape(1, self.board_size, self.board_size, 1)
        policy, value = self.model.predict(board)
        return policy[0], value[0]

class MCTSNode:
    def __init__(self, board, parent=None):
        self.board = board
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0

    def is_fully_expanded(self):
        return len(self.children) == len(self.board.get_available_moves())

    def best_child(self, exploration_weight=1.4):
        best_score = -float('inf')
        best_nodes = []
        for child in self.children.values():
            score = (child.value / child.visits +
                     exploration_weight * np.sqrt(np.log(self.visits) / child.visits))
            if score > best_score:
                best_score = score
                best_nodes = [child]
            elif score == best_score:
                best_nodes.append(child)
        return np.random.choice(best_nodes)

    def expand(self):
        available_moves = self.board.get_available_moves()
        for move in available_moves:
            if move not in self.children:
                new_board = Board(self.board.size)
                new_board.board = np.copy(self.board.board)
                new_board.current_player = self.board.current_player
                new_board.place_stone(*move)
                self.children[move] = MCTSNode(new_board, self)
                return self.children[move]

    def backpropagate(self, result):
        self.visits += 1
        self.value += result
        if self.parent:
            self.parent.backpropagate(result)

def mcts(policy_net, root, simulations=100):
    for _ in range(simulations):
        node = root
        while node.is_fully_expanded():
            node = node.best_child()
        new_node = node.expand()
        policy, _ = policy_net.predict(new_node.board.board)
        result = simulate_random_game(new_node.board)
        new_node.backpropagate(result)

def simulate_random_game(board):
    current_board = Board(board.size)
    current_board.board = np.copy(board.board)
    current_board.current_player = board.current_player

    while current_board.check_winner() is None:
        available_moves = current_board.get_available_moves()
        if not available_moves:
            return 0
        move = np.random.choice(len(available_moves))
        x, y = available_moves[move]
        current_board.place_stone(x, y)
    winner = current_board.check_winner()
    return 1 if winner == 1 else -1

def play_game():
    board = Board(size=5)
    policy_net = PolicyValueNet(board_size=5)

    fig, ax = plt.subplots()
    board.display(ax)

    def onclick(event):
        if event.xdata is None or event.ydata is None:
            return  # Ignore clicks outside of the axes

        # Adjust y to flip it for correct mapping with board's array indexing
        x, y = int(event.ydata), int(event.xdata)
        x = board.size - x - 1  # Flip y coordinate for correct indexing

        if board.place_stone(x, y):
            board.display(ax)
            winner = board.check_winner()
            if winner:
                print(f"Player {winner} wins!")
                plt.close()
                return

            # MCTS simulation for the computer's move
            root = MCTSNode(board)
            mcts(policy_net, root, simulations=100)
            best_move = max(root.children, key=lambda move: root.children[move].visits)
            board.place_stone(*best_move)
            board.display(ax)
            winner = board.check_winner()
            if winner:
                print(f"Player {winner} wins!")
                plt.close()

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

if __name__ == "__main__":
    play_game()
