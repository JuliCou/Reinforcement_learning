import numpy as np
import copy
# from tkinter import * 
import cv2
import random
import time
import imageio
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
        reward = -50
        fin = True
    elif grille_jeu[pos_t[0]][pos_t[1]] == 1:
        reward = 100
        fin = True        
    else:
        fin = False
        reward = 0
        if pos_t == pos_i:
            reward = -2
        else:
            reward = -0.1
    return pos_t, reward, fin


def choix_action(matrix_Q, q_etat, liste_actions, eps):
    p_action = random.random()
    if p_action < eps:
        return random.choice(liste_actions)
    else:
        a_t = np.where(matrix_Q[q_etat]==max(matrix_Q[q_etat]))[0]
        if a_t.shape[0] > 1:
            a_t = random.choice(a_t)
        else:
            a_t = a_t.item()
        return a_t


# 
taille_grille = 20
starting_pos = [0, 0]
grille_jeu = generate_grille_jeu(taille_grille, 100, starting_pos)

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

im = Image.fromarray(img)
im.save("grille_jeu.jpeg")

# Paramètres
matrix_Q = np.random.randn(taille_grille ** 2, 4)
lr = 0.01
gamma_futur = 1

actions = [0, 1, 2, 3]
epochs = 1000000


for partie in range(epochs):
    fin = False
    pos = copy.copy(starting_pos)
    nb_couts = 0
    while nb_couts < 500 and not(fin):
        nb_couts += 1
        # Random choice of action
        q_etat = int(taille_grille*pos[0] + pos[1])
        eps = epochs/(epochs+partie)
        a_t = choix_action(matrix_Q, q_etat, actions, eps)
        # Application de l'action et observation suite à l'action
        pos_future, reward, fin = application_action(grille_jeu, a_t, pos)
        # Update Q and new_pos
        q_etat_futur = int(taille_grille*pos_future[0] + pos_future[1])
        matrix_Q[q_etat][a_t] += lr * (reward + gamma_futur*max(matrix_Q[q_etat_futur]*(1-fin)) - matrix_Q[q_etat][a_t])
        pos = pos_future


# Initial position and parameters
pos = [0, 0]
fin = False
nb_couts = 0
tps_attente = 1

# Draw initial position
img_t = copy.copy(img)
img_t[(longueur_case*pos[0]+1):(longueur_case*(pos[0]+1)-1), (longueur_case*pos[1]+1):(longueur_case*(pos[1]+1)-1), :] = np.array([100, 255, 100])
images = [img_t]


# Test the values
while not(fin) and nb_couts < 500:
    # start_time = time.time()
    nb_couts += 1
    q_etat = int(taille_grille*pos[0] + pos[1])
    a_t = np.where(matrix_Q[q_etat]==max(matrix_Q[q_etat]))[0]
    if a_t.shape[0] > 1:
        a_t = random.choice(a_t)
    else:
        a_t = a_t.item()
    pos, reward, fin = application_action(grille_jeu, a_t, pos)
    # temps d'attente pour l'affichage
    img_t = copy.copy(img)
    img_t[(longueur_case*pos[0]+1):(longueur_case*(pos[0]+1)-1), (longueur_case*pos[1]+1):(longueur_case*(pos[1]+1)-1), :] = np.array([100, 255, 100])
    images.append(img_t)

imageio.mimsave("partie_grille20x20_aleatoire.gif", images)
