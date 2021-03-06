import sys, select, termios, tty
import numpy as np

get_key_settings = termios.tcgetattr(sys.stdin)


def getKey(dt=0.1):
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], dt)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, get_key_settings)
    return key


def wrap_to_pi(ang):
    return (ang + np.pi) % (2 * np.pi) - np.pi
