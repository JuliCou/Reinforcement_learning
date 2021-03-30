import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import copy
import random
import imageio
# Sauvegarde image numpy
from PIL import Image



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
        fin = False
    elif grille_jeu[pos_t[0]][pos_t[1]] == 1:
        reward = 50
        fin = True        
    else:
        fin = False
        if pos_t == pos_i:
            reward = -5
        else:
            reward = -0.1
    return pos_t, reward, fin


def choix_action(state, eps):
    p_action = random.random()
    if p_action < eps:
        return random.randrange(4)
    else:
        with torch.no_grad():
            return policy_net(state).max(1)[1].item()


def generate_img_grille(taille_grille, pos, taille_ideale=100):
    #
    # Initial picture sizing
    longueur_case = int(taille_ideale/taille_grille)
    taille_reelle = taille_grille * longueur_case
    #
    # Background
    img = np.zeros([taille_reelle, taille_reelle, 3], dtype=np.uint8)
    img.fill(200)
    #
    # Quadrillage
    for i in range(taille_grille-1):
        img[longueur_case*(i+1), :, :] = np.array([0, 0, 0])
        img[:, longueur_case*(i+1), :] = np.array([0, 0, 0])
    #
    # Position
    img[(longueur_case*pos[0]+1):(longueur_case*(pos[0]+1)-1), (longueur_case*pos[1]+1):(longueur_case*(pos[1]+1)-1), :] = np.array([100, 255, 100])
    #
    img = img.transpose((2, 0, 1))
    img = np.ascontiguousarray(img, dtype=np.float32) / 255
    img = torch.from_numpy(img)
    # Resize, and add a batch dimension (BCHW)
    return img.unsqueeze(0).to("cpu")


class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


def soft_update(model, target_model, tau):
    """
    Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target
    """
    for target_param, param in zip(target_model.parameters(), model.parameters()):
        target_param.data.copy_(tau*param.data + (1.0-tau)*target_param.data)


torch.manual_seed(1)

# 
taille_grille = 6
starting_pos = [0, 0]
grille_jeu = generate_grille_jeu(taille_grille, 10, starting_pos)
actions = [0, 1, 2, 3]
nb_test = 2

print(grille_jeu)


# set remaining variables
epochs = 20000
learning_rate = 5e-3
starting_pos = [0, 0]
gamma = 1.5
tau = 0.2
device = "cpu"


policy_net = DQN(100, 100, 4).to(device)
target_net = DQN(100, 100, 4).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.SGD(policy_net.parameters(), lr=learning_rate)


nb_success = 0
epoch = 1
taux_success = 0




while (epoch < epochs and taux_success < 0.9):
    epoch += 1
    # for epoch in range(epochs):
    nb_couts = 0
    fin = False
    pos = copy.copy(starting_pos)
    while nb_couts < 100 and not(fin):
        nb_couts += 1
        # Application du modèle à l'état q_etat
        state = generate_img_grille(taille_grille, pos)
        pred = policy_net(state)
        # Random choice of action - epsilon-greedy
        if epoch < 1000:
            epsilon = 1000/(1000+epoch)
        else:
            epsilon = 0.5
        a_t = choix_action(state, eps=epsilon)
        # Application de l'action
        pos_future, reward, fin = application_action(grille_jeu, a_t, pos)
        state_futur = generate_img_grille(taille_grille, pos_future)
        # Application du modèle à l'état q_etat_futur
        pred_action_future = torch.argmax(policy_net(state_futur)).item()
        pred_future = target_net(state_futur)[0][pred_action_future]
        # pred_future = target_net(state_futur).max()
        # get loss
        y_target = pred[0][a_t]
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
        soft_update(policy_net, target_net, 1e-3)
    else:
        # Update stable model
        soft_update(policy_net, target_net, tau)
    # Partie test :
    nb_couts = 0
    fin = False
    pos = copy.copy(starting_pos)
    while not(fin) and nb_couts < 100:
        # start_time = time.time()
        nb_couts += 1
        state = generate_img_grille(taille_grille, pos)
        pred = policy_net(state)
        a_t = torch.argmax(pred).item()
        pos, reward, fin = application_action(grille_jeu, a_t, pos)
    if reward == 50:
        nb_success += 1
    if epoch % 50  == 0 :
        taux_success = nb_success/50
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
    state = generate_img_grille(taille_grille, pos)
    pred = policy_net(state)
    print(pred)
    a_t = torch.argmax(pred).item()
    pos, reward, fin = application_action(grille_jeu, a_t, pos)
    # temps d'attente pour l'affichage
    img_t = copy.copy(img)
    img_t[(longueur_case*pos[0]+1):(longueur_case*(pos[0]+1)-1), (longueur_case*pos[1]+1):(longueur_case*(pos[1]+1)-1), :] = np.array([100, 255, 100])
    images.append(img_t)

nom_fichier = "grille_" + str(taille_grille) + "x" + str(taille_grille) + "_DQN_image" + str(nb_test) + ".gif"
imageio.mimsave(nom_fichier, images)

