from src.environment.pong_env import PingPong
from src.model.network import create_model
from src.agent.dqn_agent import DQNAgent
import numpy as np

def train():
    game = PingPong()

    model = create_model()
    target_model = create_model()
    target_model.set_weights(model.get_weights())

    agent = DQNAgent(model, target_model)

    TOTAL_STEPS = 2_500_000
    BATCH_SIZE = 32
    TRAIN_INTERVAL = 8
    TARGET_UPDATE = 10_000
    REPLAY_MIN = 100_000

    state = game.reset()

    while agent.total_steps < TOTAL_STEPS:
        action_idx = agent.select_action(state)
        action = np.eye(3)[action_idx]

        next_state, reward, done = game.step(action)

        agent.store(state, action_idx, reward, next_state, done)

        state = next_state
        agent.total_steps += 1

        if len(agent.replay_buffer) > REPLAY_MIN and agent.total_steps % TRAIN_INTERVAL == 0:
            agent.train(BATCH_SIZE)

        if agent.total_steps % TARGET_UPDATE == 0:
            target_model.set_weights(model.get_weights())

        agent.update_epsilon()

        if done:
            state = game.reset()