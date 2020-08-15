#!/anaconda/envs/general/bin/python

import numpy as np
import sys

def make_and_write_matrix(filename, n=10, bounds=(-10, 10), mass_bounds=(1, 10)):
    pos = np.random.uniform(size=(n, 3), high=bounds[0], low=bounds[1])
    vel = np.random.normal(size=(n, 3), scale=1)
    mass = np.ones(shape=(n, 1))
    array = np.hstack((pos, vel, mass))
    array = array.astype('float64')
    with open(filename, 'wb') as f:
        f.write(str(n).encode())
        f.write('\n'.encode())
        f.write(array.tobytes())
    return array

if __name__ == '__main__':
    array = make_and_write_matrix(sys.argv[1], n=int(sys.argv[2]))

