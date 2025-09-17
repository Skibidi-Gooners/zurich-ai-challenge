# 🦖 NEAT Dino Game AI  

An AI agent that learns to play the classic Chrome Dino game using [NEAT](https://neat-python.readthedocs.io/en/latest/) (NeuroEvolution of Augmenting Topologies). The AI evolves neural networks over generations to improve performance and survive longer in the game.  

---

## 🚀 Features  
- Uses **NEAT-Python** to evolve neural networks.  
- Learns obstacle avoidance (cacti, birds) dynamically.  
- Includes pre-trained model (`best_dino.pickle`).  
- Fully customizable via `config-feedforward.txt`.  

---

## 📂 Project Structure  
```
├── trex_game.py              # Main game & AI training loop
├── config-feedforward.txt    # NEAT configuration file
├── best_dino.pickle          # Pre-trained AI model
└── README.md                 # Project documentation
```

---

## ⚙️ Installation  

1. Clone this repository:  
   ```bash
   git clone https://github.com/your-username/dino-neat-ai.git
   cd dino-neat-ai
   ```

2. Install dependencies:  
   ```bash
   pip install pygame neat-python
   ```

---

## ▶️ Usage  

### Train a new AI  
```bash
python trex_game.py
```

### Run the pre-trained AI  
Modify the script to load `best_dino.pickle`, or replace training with evaluation.  

---

## 🧠 NEAT Configuration  
The AI is configured via `config-feedforward.txt`.  
Example parameters:  
- `pop_size = 50` → Number of genomes per generation  
- `fitness_threshold = 1000` → Stops training once fitness reached  
- `activation_default = relu` → Default node activation function  

You can tweak these settings to control how the AI evolves.  
