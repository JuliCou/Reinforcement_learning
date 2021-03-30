import math
import numpy as np
import copy
import random
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim




class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=50, fc2_units=35, fc3_units=10):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, action_size)
    #
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class DQNModel():
    def __init__(self, state_size, action_size, dqn_type='DQN', gamma=0.99,
    	learning_rate=1e-3, target_tau=2e-3, seed=0):
        self.dqn_type = dqn_type
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.learn_rate = learning_rate
        self.tau = target_tau
        self.seed = random.seed(seed)
        """
        # DQN Agent Q-Network
        # For DQN training, two nerual network models are employed;
        # (a) A network that is updated every (step % update_rate == 0)
        # (b) A target network, with weights updated to equal the network at a slower (target_tau) rate.
        # The slower modulation of the target network weights operates to stablize learning.
        """
        self.network = QNetwork(state_size, action_size, seed)
        self.target_network = QNetwork(state_size, action_size, seed)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learn_rate)

	########################################################
    # ACT() method
    #
    def act(self, state, eps=0.0):
        self.network.eval()
        with torch.no_grad():
            action_values = self.network(state)
        self.network.train()
        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

	########################################################
    # LEARN() method
    # Update value parameters using given batch of experience tuples.
    def learn(self, experiences):
        """
        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get Q values from current observations (s, a) using model nextwork
        Qsa = self.network(states)[actions]
    
        if (self.dqn_type == 'DDQN'):
        # Double DQN
        # ************************
            Qsa_prime_actions = self.network(next_states).detach().max(0)[1]
            Qsa_prime_targets = self.target_network(next_states)[Qsa_prime_actions]

        else:
        # Regular (Vanilla) DQN
        # ************************
            # Get max Q values for (s',a') from target model
            Qsa_prime_target_values = self.target_network(next_states).detach()
            Qsa_prime_targets = Qsa_prime_target_values.max(1)[0][0]  

        # Compute Q targets for current states
        Qsa_targets = rewards + (self.gamma * Qsa_prime_targets * (1 - dones))

        # Compute loss (error)
        loss = F.mse_loss(Qsa, Qsa_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # ------------------- update target network ------------------- #
        self.soft_update(self.network, self.target_network, self.tau)


    ########################################################
    """
    Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target
    """
    def soft_update(self, local_model, target_model, tau):
        """
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


def generate_grille_jeu(taille_grille=4, nb_dragons=4, starting_position=(0, 0)):
    # Grille de jeu initialisée
    grille_jeu = np.zeros((taille_grille, taille_grille))
    # Positions dragons
    for i in range(nb_dragons):
        i = np.random.randint(taille_grille)
        j = np.random.randint(taille_grille)
        test = (i==starting_position[0]) and (j==starting_position[1]) # starting position
        while grille_jeu[i][j] != 0 or test:
            i = np.random.randint(taille_grille)
            j = np.random.randint(taille_grille)
            test = (i==starting_position[0]) and (j==starting_position[1]) # starting position
        grille_jeu[i][j] = -1 # dragons
    # Positions of the winning case
    i = np.random.randint(taille_grille)
    j = np.random.randint(taille_grille)
    test = (i==starting_position[0]) and (j==starting_position[1]) # starting position
    while grille_jeu[i][j] != 0 or test:
        i = np.random.randint(taille_grille)
        j = np.random.randint(taille_grille)
        test = (i==starting_position[0]) and (j==starting_position[1]) # starting position
    grille_jeu[i][j] = 1 # diamond
    #
    return grille_jeu


def application_action(grille_jeu, a, pos_i):
    length, width = grille_jeu.shape
    pos_t = copy.copy(pos_i)
    # Changement position
    # Up
    if a == 0:
        if pos_t[0] != 0:
            pos_t[0] -= 1
    # right
    if a == 1:
        if pos_t[1] != length-1:
            pos_t[1] += 1
    # down
    if a == 2:
        if pos_t[0] != width-1:
            pos_t[0] += 1
    # left
    if a == 3:
        if pos_t[1] != 0:
            pos_t[1] -= 1
    # Interaction environnement
    if grille_jeu[pos_t[0]][pos_t[1]] == -1:  # dragons
        reward = -10
        fin = True
    elif grille_jeu[pos_t[0]][pos_t[1]] == 1:
        reward = 10
        fin = True        
    else:
        fin = False
        reward = 0
        if pos_t == pos_i:
            reward = -5
        else:
            reward = -1
    return pos_t, reward, fin


# 
taille_grille = 5
starting_pos = [0, 0]
grille_jeu = generate_grille_jeu(taille_grille, 7, starting_pos)

print(grille_jeu)

model = DQNModel(1, 4, 'DDQN', 0.9, 1e-2, 2e-3)
epochs = 2000

for epoch in range(epochs):
    nb_couts = 0
    fin = False
    pos = copy.copy(starting_pos)
    while nb_couts < 100 and not(fin):
        nb_couts += 1
        # Application du modèle à l'état q_etat
        q_etat = int(taille_grille*pos[0] + pos[1])
        X = torch.tensor(q_etat, dtype=torch.float32).unsqueeze(dim=0)
        # Random choice of action - epsilon-greedy
        a_t = model.act(X, eps=1)
        # Application de l'action
        pos_future, reward, fin = application_action(grille_jeu, a_t, pos)
        q_etat_futur = int(taille_grille*pos_future[0] + pos_future[1])
        X_futur = torch.tensor(q_etat_futur, dtype=torch.float32).unsqueeze(dim=0)
        # Entrainement modèle
        model.learn((X, a_t, reward, X_futur, fin))
        # Update parameters
        pos = pos_future


# Initial picture
taille_ideale = 500
taille_grille = grille_jeu.shape[0]
longueur_case = int(taille_ideale/taille_grille)
taille_reelle = taille_grille * longueur_case

# Background
img = np.zeros([taille_reelle, taille_reelle, 3], dtype=np.uint8)
img.fill(200)

# Quadrillage
for i in range(taille_grille-1):
    img[longueur_case*(i+1), :, :] = np.array([0, 0, 0])
    img[:, longueur_case*(i+1), :] = np.array([0, 0, 0])

# Dragons
dragon_positions_length, dragon_positions_width = np.where(grille_jeu==-1)[0], np.where(grille_jeu==-1)[1]
for i_drag in range(len(dragon_positions_length)):
    d_l = dragon_positions_length[i_drag]
    d_w = dragon_positions_width[i_drag]
    img[(longueur_case*d_l+1):(longueur_case*(d_l+1)-1), (longueur_case*d_w+1):(longueur_case*(d_w+1)-1), :] = np.array([255, 0, 0])

# Diamond
winning_pos_y, winning_pos_x = np.where(grille_jeu==1)[0][0], np.where(grille_jeu==1)[1][0]
img[(longueur_case*winning_pos_y+1):(longueur_case*(winning_pos_y+1)-1), (longueur_case*winning_pos_x+1):(longueur_case*(winning_pos_x+1) -1), :] = np.array([255, 255, 255])


# Initial position and parameters
pos = [0, 0]
fin = False
nb_couts = 0

# Draw initial position
img_t = copy.copy(img)
img_t[(longueur_case*pos[0]+1):(longueur_case*(pos[0]+1)-1), (longueur_case*pos[1]+1):(longueur_case*(pos[1]+1)-1), :] = np.array([100, 255, 100])
images = [img_t]


# Test the values
while not(fin) and nb_couts < 100:
    # start_time = time.time()
    nb_couts += 1
    q_etat = int(taille_grille*pos[0] + pos[1])
    q_etat = int(taille_grille*pos[0] + pos[1])
    input_nn = torch.tensor(q_etat, dtype=torch.float32).unsqueeze(dim=0)
    a_t = model.act(input_nn)
    pos, reward, fin = application_action(grille_jeu, a_t, pos)
    # temps d'attente pour l'affichage
    img_t = copy.copy(img)
    img_t[(longueur_case*pos[0]+1):(longueur_case*(pos[0]+1)-1), (longueur_case*pos[1]+1):(longueur_case*(pos[1]+1)-1), :] = np.array([100, 255, 100])
    images.append(img_t)

imageio.mimsave("partie_grille5x5_deep.gif", images)
