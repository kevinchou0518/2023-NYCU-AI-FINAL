import numpy as np
import matplotlib.pyplot as plt

x1 = np.array([])
y1 = np.array([])
with open('only_ml_5_epoch_2300_steps_20230605-163411\eval_train_epoch_4.json', 'r') as f:
    for line in f:
        # divided the line with ",".
        line = line.split(",")
        epoch = line[0].split(":")
        # step is in the line[1].
        step = line[1].split(":")
        # rouge is in the line[5].
        rouge = line[5].split(":")
        rouge = rouge[1].split("}")
        # append into array.
        x1 = np.append(x1, (int(epoch[1])*7500+int(step[1]))/100)
        y1 = np.append(y1, float(rouge[0]))
plt.plot(x1, y1, label = "Supervised Learning")

x2 = np.array([])
y2 = np.array([])
with open('ml_rl_secondtry_collect_data\eval_train_epoch_4.json', 'r') as f:
    for line in f:
        # divided the line with ",".
        line = line.split(",")
        epoch = line[0].split(":")
        # step is in the line[1].
        step = line[1].split(":")
        # rouge is in the line[5].
        rouge = line[5].split(":")
        rouge = rouge[1].split("}")
        # append into array.
        x2 = np.append(x2, (int(epoch[1])*7500+int(step[1]))/100)
        y2 = np.append(y2, float(rouge[0]))
plt.plot(x2, y2, label = "RL + Supervised Learning")

plt.title('RougeLsum')
plt.xlabel('step')
plt.ylabel('rougeLsum')
plt.legend()

plt.savefig('picture\RougeLsum.png')
plt.show()