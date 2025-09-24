#!/usr/bin/env python3

import one as o
import matplotlib.pyplot as plt
import numpy as np


def i(x, y):
    """y′ = −xy"""
    return -x * y


def ii(x, y):
    """y′ = xy"""
    return x * y


def iii(x, y):
    """
    xdx + ydy = 0
    y' = -x/y
    """
    y_safe = np.where(np.abs(y) < 1e-10, 1e-10, y)
    return -x / y_safe


def iv(x, y):
    """
    ydx + xdy = 0
    ydx = -xdy
    y' = -y/x
    """
    x_safe = np.where(np.abs(x) < 1e-10, 1e-10, x)
    return -y / x_safe


def v(x, y):
    """
    y' = y²-y
    """
    return y * (y - 1)


# fig1 = o.plot_direction_field(
#     f=i,
#     xmin=-5,
#     xmax=5,
#     ymin=-5,
#     ymax=5,
#     xstep=0.3,
#     ystep=0.3,
#     field_type="F",
#     streamlines=True,
#     arrow_scale=0.2,
#     title="I",
# )
# plt.show()

# fig2 = o.plot_direction_field(
#     f=ii,
#     xmin=-5,
#     xmax=5,
#     ymin=-5,
#     ymax=5,
#     xstep=0.3,
#     ystep=0.3,
#     field_type="F",
#     streamlines=True,
#     arrow_scale=0.2,
#     title="II",
# )
# plt.show()

# fig3 = o.plot_direction_field(
#     f=iii,
#     xmin=-5,
#     xmax=5,
#     ymin=-5,
#     ymax=5,
#     xstep=0.3,
#     ystep=0.3,
#     field_type="F",
#     streamlines=True,
#     arrow_scale=0.2,
#     title="III",
# )
# plt.show()

# fig4 = o.plot_direction_field(
#     f=iv,
#     xmin=-5,
#     xmax=5,
#     ymin=-5,
#     ymax=5,
#     xstep=0.3,
#     ystep=0.3,
#     field_type="F",
#     streamlines=True,
#     arrow_scale=0.2,
#     title="IV",
# )
# plt.show()

fig5 = o.plot_direction_field(
    f=v,
    xmin=-5,
    xmax=5,
    ymin=-5,
    ymax=5,
    xstep=0.3,
    ystep=0.3,
    field_type="F",
    streamlines=True,
    arrow_scale=0.2,
    title="V",
)
plt.show()
