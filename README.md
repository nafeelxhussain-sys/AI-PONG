# 🏓 Pong AI using Deep Q-Learning (DQN)

A reinforcement learning-based Pong game where an AI agent learns to play using Deep Q-Networks.
The agent improves over time by interacting with a custom-built environment and maximizing rewards.

---

## 🚀 Features

* 🎮 Custom Pong environment built using Pygame
* 🤖 AI agent trained with Deep Q-Learning (DQN)
* 📈 Training metrics (win rate, rewards, episode length)
* 🧠 Experience replay + target network
* ⚡ Epsilon-greedy exploration strategy
* 👤 Human vs AI / AI vs AI gameplay

---

## 🧠 How it works

The agent observes the game state and decides actions to maximize cumulative reward.

### State Space (6 values)

* Ball position (x, y)
* Ball velocity (vx, vy)
* Right paddle position (agent)
* Left paddle position (opponent)

### Actions (3)

* Move Up
* Stay
* Move Down

### Reward System

* +1 → Opponent misses (agent scores)
* -1 → Agent misses
* +0.25 → Successful paddle hit
* Extra penalty based on miss distance

---

## 🏗️ Project Structure

```id="n5m7c2"
pong-ai/
│
├── src/
│   ├── environment/      # Pong game logic
│   ├── agent/            # DQN agent
│   ├── model/            # Neural network
│   ├── training/         # Training loop
│
├── scripts/
│   ├── train.py          # Train the agent
│   └── play.py           # Run the game
│
├── models/               # Saved models
├── results/              # Plots and logs
│
├── run_play.bat          # One-click run (Windows)
├── run_train.bat
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/nafeelxhussain-sys/AI-PONG/
cd pong-ai
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### 🎮 Play the game

```bash
python -m scripts.play
```

---

### 🎮 Spectate the game against AI

```bash
python -m scripts.spectate
```

---

### 🧠 Train the AI

```bash
python -m scripts.train
```

---

## 📈 Training Details

* Algorithm: Deep Q-Network (DQN)
* Neural Network: 2 hidden layers (256 units each)
* Optimizer: Adam
* Learning Rate: 1e-4
* Discount Factor (γ): 0.98
* Replay Buffer: 500,000
* Batch Size: 32

---

## 📊 Results

The agent learns to:

* Track the ball
* Position the paddle correctly
* Improve rally length over time
* Average winrate is 81%

Training metrics include:

* Win rate
* Average reward
* Episode length

## 🎮 Gameplay

Right side agent
[Watch Gameplay](results/gameplay.mp4)

---

## 🛠️ Future Improvements

* Double DQN
* Prioritized Experience Replay
* Better reward shaping
* Model checkpoint saving/loading
* Improved UI and visuals

---

## ⚠️ Notes

* Recommended Python version: **3.10 – 3.11**
* TensorFlow may have compatibility issues with Python 3.12


