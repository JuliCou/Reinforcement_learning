# Reinforcement_learning
Reinforcement learning university project + DQN, numpy and PyTorch implementation

## Première partie : Q-learning
Partie avec un tableau (n, m) avec n le nombre d'états possibles et m le nombre d'actions possibles.   
Le code est reinforcement_learning.py avec une grille déjà définie.   
Pour une grille générée de façon aléatoire, on choisit le code reinforcement_learning_grille_aleatoire.py   
On change le paramètre taille_grille (ligne 74) et le nombre de dragons, 2ème paramètre de la fonction generate_grille_jeu, ligne 76.

## Deuxième partie : DQN (Deep) (entrée numérique)
Cette partie utilise la librairie PyTorch pour entrainer un réseau de neurones à une entrée, représentant l'état du jeu et 4 sorties, (pour les 4 actions possibles).

## Troisième partie : DQN avec l'image du jeu en entrée
![Représentation de l'image en entrée](DQN%20entree%20image/test.jpg "Représentation de l'image en entrée")