import numpy as np
import matplotlib.pyplot as plt

x1 = np.array([])
y1 = np.array([])
with open('ml_rl_secondtry_collect_data\loss_train_epoch_4.json', 'r') as f:
    for line in f:
        # divided the line with ",".
        line = line.split(",")
        epoch = line[0].split(":")
        # step is in the line[1].
        step = line[1].split(":")
        # rouge is in the line[2].
        loss = line[2].split(":")
        loss = loss[1].split("}")
        # append into array.
        x1 = np.append(x1, (int(epoch[1])*7556+int(step[1]))/16)
        y1 = np.append(y1, float(loss[0]))
plt.plot(x1, y1)

plt.title('RL + Supervised Learning')
plt.xlabel('step')
plt.ylabel('loss')
plt.legend()

plt.savefig('picture\Loss_ml_rl.png')
plt.show()