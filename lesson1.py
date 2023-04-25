import numpy as np


def activation(x):
    return 0 if x < 0.5 else 1


def choice(house, music, beauty):
    inp_signals = np.array([house, music, beauty])
    weight1 = np.array([
        [0.3, 0.3, 0],
        [0.4, -0.5, 1]
    ])
    weight2 = np.array([-1, 1])

    sum_hidden = np.dot(weight1, inp_signals)
    out_hidden = np.array([activation(x) for x in sum_hidden])
    sum_end = np.dot(weight2, out_hidden)

    return activation(sum_end)


if __name__ == '__main__':
    print(choice(0, 0.5, 0.5))
