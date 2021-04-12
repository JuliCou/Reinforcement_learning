import torch
import gym
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import time


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def get_screen():
    screen = env.render(mode='rgb_array')[:175,:,:].transpose((2, 0, 1))
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    return screen.unsqueeze(0).to('cpu')
    


# Démarrage du jeu
env = gym.make('MsPacman-v0')
obs = env.reset()
screen = get_screen()
screen_h, screen_w = screen.shape[2:]
n_actions = env.action_space.n

model = DQN(screen_h, screen_w, n_actions)
model.load_state_dict(torch.load("model.pt"))
model.eval()


env.reset()
current_screen = get_screen()
done = False
rec = 0
temps = time.time()
while not(done):
    while time.time() - temps < 0.2:
        pass
    temps = time.time()
    env.render()
    with torch.no_grad():
        a_t = model(current_screen).max(1)[1].item()
    # Application de l'action
    _, reward, done, info = env.step(a_t)
    rec += reward
    # Observe new state
    current_screen = get_screen()

