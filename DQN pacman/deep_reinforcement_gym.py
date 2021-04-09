import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import copy
import random
import gym


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def choix_action(state, eps):
    p_action = random.random()
    if p_action < eps:
        return random.randrange(n_actions)
    else:
        with torch.no_grad():
            return policy_net(state).max(1)[1].item()



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


def get_screen():
    screen = env.render(mode='rgb_array')[:175,:,:].transpose((2, 0, 1))
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    return screen.unsqueeze(0).to(device)


torch.manual_seed(1)

# Démarrage du jeu
env = gym.make('MsPacman-v0')
obs = env.reset()
screen = get_screen()
screen_h, screen_w = screen.shape[2:]
n_actions = env.action_space.n

# set remaining variables
epochs = 100
learning_rate = 1e-3
gamma = 0.1
tau = 0.2
device = "cpu"



policy_net = DQN(screen_h, screen_w, n_actions).to(device)
target_net = DQN(screen_h, screen_w, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)


epoch = 0



best_model = copy.copy(policy_net)
score_suivi = []

while epoch < epochs:
    epoch += 1
    # Initialize the environment and state
    env.reset()
    last_screen = get_screen()
    current_screen = get_screen()
    state = current_screen - last_screen
    nb_couts = 0
    done = False
    nb_vie = 3
    while not(done) and nb_couts < 1000:
        nb_couts += 1
        # Random choice of action - epsilon-greedy
        epsilon = epochs/(epochs+epoch)
        a_t = choix_action(state, eps=epsilon)
        # Application de l'action
        _, reward, done, info = env.step(a_t)
        reward = torch.tensor([reward], device=device)
        fin = False
        if info["ale.lives"] != nb_vie:
            nb_vie -= 1
            fin = True
            reward = -20
        else:
            if done:
                fin = True
            else:
                if reward == 1:
                    reward = reward * 3
        # Observe new state
        last_screen = current_screen
        current_screen = get_screen()
        next_state = current_screen - last_screen
        # Predictions
        pred = policy_net(state)
        pred_future = target_net(next_state).max()
        state = next_state
        # get loss
        y_target = pred[0][a_t]
        y_eval = reward + gamma*pred_future*(1-fin)
        # F.mse_loss() ou F.smooth_l1_loss()
        loss = F.mse_loss(y_target, y_eval)
        # perform backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Update stable model
        soft_update(policy_net, target_net, tau)
    if epoch % 5 == 0:
        soft_update(policy_net, target_net, 1e-3)
    if epoch % 5 == 0:
        env.reset()
        current_screen = get_screen()
        done = False
        rec = 0
        while not(done):
            with torch.no_grad():
                a_t = policy_net(current_screen).max(1)[1].item()
            # Application de l'action
            _, reward, done, _ = env.step(a_t)
            rec += reward
            # Observe new state
            current_screen = get_screen()
        score_suivi.append(rec)
        print(epoch, rec)
        if rec == max(score_suivi):
            best_model = copy.copy(policy_net)


torch.save(policy_net.state_dict(), "model.pt")
torch.save(best_model.state_dict(), "best_model.pt")
model = DQN(screen_h, screen_w, n_actions)
model.load_state_dict(torch.load("model.pt"))
model.eval()