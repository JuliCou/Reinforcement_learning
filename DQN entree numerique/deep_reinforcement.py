import math
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential, Softmax, Sigmoid, MSELoss
from torch.optim import SGD, Adam
import torch.nn.functional as F
import copy
import random
import imageio



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
        if pos_t == pos_i:
            reward = -5
        else:
            reward = 0.1
    return pos_t, reward, fin


def choix_action(pred, liste_actions, eps):
    p_action = random.random()
    if p_action < eps:
        return random.choice(liste_actions)
    else:
        return liste_actions[torch.argmax(pred).item()]


def soft_update(model, target_model, tau):
    """
    Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target
    """
    for target_param, param in zip(target_model.parameters(), model.parameters()):
        target_param.data.copy_(tau*param.data + (1.0-tau)*target_param.data)


torch.manual_seed(1)

# 
taille_grille = 4
starting_pos = [0, 0]
grille_jeu = generate_grille_jeu(taille_grille, 4, starting_pos)
actions = [0, 1, 2, 3]

# grille_jeu = np.zeros((4, 4))
# grille_jeu[0][1] = -1
# grille_jeu[1][1] = -1
# grille_jeu[3][2] = -1
# grille_jeu[1][3] = -1
# grille_jeu[0][2] = 1
print(grille_jeu)

# number of neurons in each layer
input_num_units = 1
hidden_num_units_layer_1 = 75
hidden_num_units_layer_2 = 50
hidden_num_units_layer_3 = 20
hidden_num_units_layer_4 = 10
output_num_units = len(actions)

# set remaining variables
epochs = 20000
learning_rate = 1e-3
starting_pos = [0, 0]
gamma = 1.5
tau = 0.2

model = Sequential(Linear(input_num_units, hidden_num_units_layer_1),
                   ReLU(),
                   Linear(hidden_num_units_layer_1, hidden_num_units_layer_2),
                   ReLU(),
                   Linear(hidden_num_units_layer_2, hidden_num_units_layer_3),
                   ReLU(),
                   Linear(hidden_num_units_layer_3, hidden_num_units_layer_4),
                   ReLU(),
                   Linear(hidden_num_units_layer_4, output_num_units))
# model = Sequential(Linear(input_num_units, hidden_num_units_layer_1),
#                    ReLU(),
#                    Linear(hidden_num_units_layer_1, hidden_num_units_layer_2),
#                    ReLU(),
#                    Linear(hidden_num_units_layer_2, hidden_num_units_layer_3),
#                    ReLU(),
#                    Linear(hidden_num_units_layer_3, output_num_units))

optimizer = SGD(model.parameters(), lr=learning_rate)
model_stable = copy.copy(model)

nb_success = 0
epoch = 1
taux_success = 0


while epoch < epochs and taux_success < 0.9:
    epoch += 1
    # for epoch in range(epochs):
    nb_couts = 0
    fin = False
    pos = copy.copy(starting_pos)
    while nb_couts < 100 and not(fin):
        nb_couts += 1
        # Application du modèle à l'état q_etat
        q_etat = int(taille_grille*pos[0] + pos[1])
        X = torch.tensor(q_etat, dtype=torch.float32).unsqueeze(dim=0)
        pred = model(X)
        # Random choice of action - epsilon-greedy
        if epoch < 5000:
            epsilon = 5000/(5000+epoch)
        else:
            epsilon = 0.5
        a_t = choix_action(pred, actions, eps=epsilon)
        # Application de l'action
        pos_future, reward, fin = application_action(grille_jeu, a_t, pos)
        q_etat_futur = int(taille_grille*pos_future[0] + pos_future[1])
        X_futur = torch.tensor(q_etat_futur, dtype=torch.float32).unsqueeze(dim=0)
        # Application du modèle à l'état q_etat_futur
        pred_action_future = torch.argmax(model(X_futur)).item()
        pred_future = model_stable(X_futur)[pred_action_future]
        # pred_future = model_stable(X_futur).max()
        # get loss
        y_target = pred[a_t]
        y_eval = reward + gamma*pred_future*(1-fin)
        # F.mse_loss() ou F.smooth_l1_loss()
        loss = F.smooth_l1_loss(y_target, y_eval)
        # perform backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Update parameters
        pos = pos_future
    if epoch % 5 == 0:
        soft_update(model, model_stable, 1e-3)
    else:
        # Update stable model
        soft_update(model, model_stable, tau)
    # Partie test :
    nb_couts = 0
    fin = False
    pos = copy.copy(starting_pos)
    while not(fin) and nb_couts < 100:
        # start_time = time.time()
        nb_couts += 1
        q_etat = int(taille_grille*pos[0] + pos[1])
        input_nn = torch.tensor(q_etat, dtype=torch.float32)
        pred = model(input_nn.unsqueeze(dim=0))
        a_t = torch.argmax(pred).item()
        pos, reward, fin = application_action(grille_jeu, a_t, pos)
    if reward == 10:
        nb_success += 1
    if epoch % 250  == 0 :
        taux_success = nb_success/250
        nb_success = 0
        print("Epoch ", str(epoch), " - réussite : ", str(taux_success))


print(nb_success)
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
    input_nn = torch.tensor(q_etat, dtype=torch.float32)
    pred = model_stable(input_nn.unsqueeze(dim=0))
    print(pred)
    a_t = torch.argmax(pred).item()
    pos, reward, fin = application_action(grille_jeu, a_t, pos)
    # temps d'attente pour l'affichage
    img_t = copy.copy(img)
    img_t[(longueur_case*pos[0]+1):(longueur_case*(pos[0]+1)-1), (longueur_case*pos[1]+1):(longueur_case*(pos[1]+1)-1), :] = np.array([100, 255, 100])
    images.append(img_t)

imageio.mimsave("partie_grille4x4_deep.gif", images)

