### **Technical Overview & Core Concepts**

[cite_start]The core of PerfectDou is a novel training framework called **Perfect-Training-Imperfect-Execution (PTIE)**, which is a specialized version of the Centralized-Training-Decentralized-Execution (CTDE) paradigm tailored for imperfect information games[cite: 8, 28].

* **PTIE Framework**:
    * **Training Phase**: During training, the system has access to **perfect information**, meaning it knows the cards of all players. [cite_start]This global view is used to train the *value network (critic)*, allowing it to accurately assess the game state[cite: 8].
    * [cite_start]**Execution (Inference) Phase**: When playing, the trained *policy network (actor)* operates using only **imperfect information**—the information a human player would have (their own hand and publicly played cards)[cite: 8, 29].

[cite_start]This "perfect information distillation" process allows the policy network to learn sophisticated strategies and coordination, guided by the critic's perfect knowledge, without needing that knowledge during actual gameplay[cite: 7, 93].

---

### **Network Architecture**

PerfectDou employs an actor-critic model with distinct policy and value networks. [cite_start]The learning algorithm is **Proximal Policy Optimization (PPO)** with **Generalized Advantage Estimation (GAE)**[cite: 10, 31, 158].

#### **Policy Network (Actor)**

The policy network's job is to select an action based on incomplete information.

* **Input Features (Imperfect Information)**:
    * [cite_start]A flattened matrix of size **$23 \times 12 \times 15$**[cite: 156]. This encodes:
        * [cite_start]Current player's hand[cite: 152].
        * [cite_start]Unplayed cards[cite: 152].
        * [cite_start]Cards played by each player[cite: 152].
        * [cite_start]The three leftover cards for the Landlord[cite: 152].
        * [cite_start]The last 15 moves[cite: 152].
    * [cite_start]A game state array of size **$6 \times 1$**[cite: 156]. This includes:
        * [cite_start]Number of cards remaining for each player[cite: 152].
        * [cite_start]Number of bombs played[cite: 152].
        * [cite_start]A flag indicating who has control of the game[cite: 152].
    * [cite_start]**Available Actions**: Each legal move is also encoded as a feature vector[cite: 161].

* **Architecture**:
    1.  [cite_start]**Feature Encoding**: An **LSTM** processes the imperfect game state features to capture historical context[cite: 160].
    2.  [cite_start]**Target Attention**: The game state representation is concatenated with each available action's feature vector separately[cite: 162, 166].
    3.  [cite_start]**Output**: The combined vectors are fed through a multi-layer perceptron (MLP) and a **softmax** function to produce a probability distribution over all legal actions[cite: 163, 165].


---

#### **Value Network (Critic)**

The value network evaluates the game state using both perfect and imperfect information to provide a more accurate assessment.

* **Input Features (Perfect Information)**:
    * It includes all the imperfect features mentioned above.
    * **Additional Perfect Features**:
        * [cite_start]A flattened matrix of size **$25 \times 12 \times 15$** and a game state array of **$8 \times 1$**[cite: 156].
        * [cite_start]This adds the hand cards of the other two players[cite: 152].
        * [cite_start]It also includes the *minimum number of steps* for each opponent to play out their cards[cite: 152].

* **Architecture**:
    1.  [cite_start]**Feature Encoding**: Imperfect features are encoded using a shared network structure with the policy network[cite: 497]. [cite_start]Perfect features are encoded separately[cite: 498].
    2.  [cite_start]**Concatenation**: The encoded perfect and imperfect feature vectors are concatenated[cite: 499].
    3.  [cite_start]**Output**: The final vector is processed by an **MLP** to output a single scalar value representing the game state's evaluation[cite: 499, 159].


---

### **Training Data and Process**

* [cite_start]**Training Method**: The system is trained through **self-play** in a distributed environment[cite: 31]. [cite_start]Three separate models are maintained and trained for the Landlord and the two Peasant positions[cite: 213].
* **Distributed System**:
    * [cite_start]The architecture is inspired by **IMPALA**[cite: 211].
    * [cite_start]It uses a large number of **rollout workers** (1000 workers, each on a single CPU core) to collect gameplay experience[cite: 188, 226].
    * [cite_start]These workers send data to a pool of **GPU learners** (8 GPUs)[cite: 188, 518].
    * [cite_start]Gradients are computed on each GPU and then averaged synchronously across all GPUs for network updates[cite: 190].
    * [cite_start]Updated model parameters are sent back to the rollout workers every 24 GAE steps (8 steps per player)[cite: 191, 209].
* [cite_start]**Data Volume**: The final model was trained for **2.5e9 (2.5 billion) steps**[cite: 292]. [cite_start]An earlier version trained on 1e9 steps already showed superior performance to the previous state-of-the-art[cite: 294].
* **Hyperparameters**:
    * [cite_start]**Learning Rate**: 3e-4[cite: 533].
    * [cite_start]**Optimizer**: Adam[cite: 533].
    * [cite_start]**GAE $\lambda$**: 0.95[cite: 533].
    * [cite_start]**PPO Entropy Weight**: 0.1[cite: 533].
    * [cite_start]**Policy MLP Hidden Layers**: [256, 256, 256, 512][cite: 533].
    * [cite_start]**Value MLP Hidden Layers**: [256, 256, 256, 256][cite: 533].

---

### **Reward Design**

A key innovation is the **perfect reward function**, which leverages perfect information to provide a dense and accurate reward signal during training.

* [cite_start]**Oracle-Based Reward**: Instead of using only the sparse win/loss signal at the end of a game, the reward is calculated at each step based on an "oracle"[cite: 170, 171].
* [cite_start]**Oracle Calculation**: The oracle is a dynamic programming algorithm that calculates the **minimum number of moves required for a player to win** (play out all their cards) from a given state[cite: 171, 174, 657].
* [cite_start]**Reward Formula**: The reward at timestep $t$ is based on the change in advantage, where advantage is the difference in the minimum steps to win between the Landlord and the best-positioned Peasant[cite: 176, 177].

    Let $N_t^{\text{player}}$ be the minimum steps for a player to win at time $t$. The advantage at time $t$ is:
    $$Adv_t = N_t^{\text{Landlord}} - \min(N_t^{\text{Peasant1}}, N_t^{\text{Peasant2}})$$
    [cite_start][cite: 179]

    The reward $r_t$ is then:
    $$r_t = \begin{cases} -1.0 \times (Adv_t - Adv_{t-1}) \times \beta & \text{for Landlord} \\ 0.5 \times (Adv_t - Adv_{t-1}) \times \beta & \text{for Peasants} \end{cases}$$
    [cite_start][cite: 178]
    [cite_start]where $\beta$ is a scaling factor[cite: 182].

[cite_start]This design encourages cooperation between Peasants (since their reward depends on the minimum of their two distances to winning) and provides a more stable, informative gradient for learning[cite: 185].