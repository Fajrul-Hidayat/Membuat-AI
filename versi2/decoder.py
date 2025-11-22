import numpy as np

def greedy_decode(pred, idx_to_char):
    out = np.argmax(pred[0], axis=-1)
    text = ""
    prev = -1

    for x in out:
        if x != prev and x != 0:  # 0 = blank CTC
            text += idx_to_char.get(x, "")
        prev = x

    return text

import numpy as np

def greedy_decode(pred, idx_to_char):
    out = np.argmax(pred[0], axis=-1)
    text = ""
    prev = -1

    for x in out:
        if x != prev and x != 0:  # 0 = blank CTC
            text += idx_to_char.get(x, "")
        prev = x

    return text
