import numpy as np
import matplotlib.pyplot as plt
from mplEasyAnimate import animation
from tqdm import tqdm
import argparse

def load_C_output(filename):
    with open(filename, 'rb') as f:
        line_one = f.readline()
        remainder = f.read()
    line_one = line_one.decode("utf-8")
    line_one = [int(x.rstrip()) for x in line_one.split(',')]
    bodies = line_one[0]
    timesteps=line_one[1]
    state = np.frombuffer(remainder, dtype=np.float64)
    state = state.reshape(timesteps, bodies, 7)
    return state

def load_and_animate_from_disk(datafile, animationName, fps, skip):
    state = load_C_output(datafile)
    anim = animation(animationName, fps=fps)
    for i, timestep in tqdm(enumerate(state), total=len(state)):
        if i%skip == 0:
            fig = plt.figure()
            plt.plot(timestep[:, 0], timestep[:, 1], 'o')
            anim.add_frame(fig)
            plt.close(fig)
    anim.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to output file from integrator", type=str)
    parser.add_argument("-o", "--output", default='anim.mp4', help="Path to save the animation at", type=str)
    parser.add_argument("-f", "--fps", default=30, help="set the frams per second animation is written at", type=int)
    parser.add_argument("-s", "--skip", default=5, help="frames per n frames to skip", type=int)
    args = parser.parse_args()
    load_and_animate_from_disk(args.path, args.output, args.fps, args.skip)

