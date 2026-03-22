from src.environment.pong_env import PingPong
from src.model.network import create_model
import numpy as np
import pygame

def play():
    game = PingPong()
    model = create_model()
    model.load_weights("models/dqn_model4.keras")

    state = game.reset()
    running = True

    while running:
        game.handle_events()
        game.move_user()

        q = model(state, training=False)[0].numpy()
        action_idx = np.argmax(q)
        action = np.eye(3)[action_idx]

        next_state, _, done = game.step(action)
        state = next_state

        game.render()

        if done:
            state = game.reset_rally()

        if game.isFinished():
            running = False

    pygame.quit()

if __name__ == "__main__":
    play()