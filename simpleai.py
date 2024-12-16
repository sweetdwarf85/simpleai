import random
import numpy as np

class EnemyAI:
    def __init__(self, action_space):
        self.action_space = action_space
        self.q_table = {}
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.learning_rate = 0.1
        self.discount_factor = 0.9

    def get_state_key(self, state):
        return tuple(state)

    def choose_action(self, state):
        state_key = self.get_state_key(state)
        if random.random() < self.epsilon:
            return random.choice(self.action_space)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(len(self.action_space))
        return np.argmax(self.q_table[state_key])

    def learn(self, state, action, reward, next_state):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(len(self.action_space))
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(len(self.action_space))
        max_future_q = np.max(self.q_table[next_state_key])
        current_q = self.q_table[state_key][action]
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_future_q - current_q)
        self.q_table[state_key][action] = new_q

    def update_epsilon(self):
        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)

def play_game():
    action_space = [-1, 0, 1]  # -1 left +1 right
    ai = EnemyAI(action_space)

    player_pos = 5
    enemy_pos = 0
    game_length = 20  

    for episode in range(50):  # 50 laps
        print(f"Episode {episode + 1}")
        for step in range(20):  # 20 move
            print("-" * game_length)
            print(" " * player_pos + "P")  
            print(" " * enemy_pos + "E")  
            print("-" * game_length)

            state = [enemy_pos, player_pos]
            action = ai.choose_action(state)

            # ai movement
            enemy_pos = max(0, min(game_length - 1, enemy_pos + action))

            # player movement
            move = input("Move (a: left, d: right, s: stay): ").lower()
            if move == "a":
                player_pos = max(0, player_pos - 1)
            elif move == "d":
                player_pos = min(game_length - 1, player_pos + 1)

            # Reward: Positive when the enemy approaches the player, negative when the enemy moves away
            distance = abs(enemy_pos - player_pos)
            reward = -distance  # closer wins reward
            next_state = [enemy_pos, player_pos]

            ai.learn(state, action, reward, next_state)

        ai.update_epsilon()

play_game()
