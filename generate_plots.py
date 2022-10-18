import matplotlib.pyplot as plt
import json
import numpy as np
from scipy.ndimage.filters import uniform_filter1d

model = 'Res-UNet'

file_path = 'runs_res/debug/'

max_step = 50400 #28000#
step_start = 11
step_diff = 10
num_epochs = 12 #20#

val_jaccards = []
val_losses = []
train_losses = []

for i in range(0, 3):
    log_file_name = f'{file_path}train_{i}.log'
    
    val_jaccard = []
    val_loss = []
    train_loss = []
    with open(log_file_name, 'r') as f:
        for line in f:
            if len(line.strip()) == 0:
                continue
            values = json.loads(line.strip())
            if values['step'] <= max_step:
                if 'loss' in values:
                    train_loss.append(values['loss'])
                else:
                    val_loss.append(values['valid_loss'])
                    val_jaccard.append(values['jaccard_loss'])
                    
    val_jaccards.append(val_jaccard)
    val_losses.append(val_loss)
    train_losses.append(train_loss)

x = np.arange(step_start, len(train_losses[0]) * step_diff + step_start, step_diff)

for i in range(0, 3):
    plt.plot(x, uniform_filter1d(train_losses[i], size=100), label = f"Fold {i}")
plt.legend()
plt.title(f'{model} Training Loss vs. Step')
plt.show()
x = np.arange(0, num_epochs)
for i in range(0, 3):
    plt.plot(x, val_losses[i], label = f"Fold {i}")
plt.legend()
plt.title(f'{model} Validation Loss vs. Epoch')
plt.show()
for i in range(0, 3):
    plt.plot(x, val_jaccards[i], label = f"Fold {i}")
plt.legend()
plt.title(f'{model} Validation Jaccard vs. Epoch')
plt.show()