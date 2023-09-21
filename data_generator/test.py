import numpy as np
import os
from PIL import Image
from scipy.ndimage import zoom
from scipy.signal import lfilter


def elementaryCellularAutomata(rule, n, width=None, randfrac=None):
    # Validate inputs
    if not (0 <= rule <= 255):
        raise ValueError("Rule should be an integer between 0 and 255.")

    if not isinstance(n, int) or n <= 0:
        raise ValueError("N should be a positive integer.")

    if width is None:
        width = 2 * n - 1

    if isinstance(width, int):
        if width <= 0:
            raise ValueError("Width should be a positive integer.")
        patt = np.ones(width, dtype=int)
        patt[(width + 1) // 2 - 1] = 2
    else:
        if not all(map(lambda x: x in [0, 1], width)):
            raise ValueError("Width should be a list of 0s and 1s only.")
        patt = np.array(width) + 1
        width = len(patt)

    if randfrac is not None:
        if not (0 <= randfrac <= 1):
            raise ValueError("Randfrac should be a float between 0 and 1.")
        dorand = True
    else:
        dorand = False

    # Unpack rule
    rulearr = [(rule >> i) & 1 for i in range(8)]
    rulearr = np.array(rulearr) + 1

    # Initialize output pattern
    pattern = np.zeros((n, width), dtype=int)

    # Generate the pattern
    for i in range(n):
        pattern[i, :] = patt
        ind = 2 ** 2 * np.roll(patt, 1) + 2 ** 1 * patt + 2 ** 0 * np.roll(patt, -1)
        ind = ind % 8  # Ensure indices are within 0-7
        patt = rulearr[ind]

        # Optional randomization
        if dorand:
            flip = np.random.rand(width) < randfrac
            patt[flip] = 3 - patt[flip]

    # Convert 1 and 2 to 0 and 1
    pattern -= 1

    return pattern

ans = elementaryCellularAutomata(200, 1024, 1024, 0.5)
from matplotlib import pyplot as plt
plt.imshow(ans)
plt.show()