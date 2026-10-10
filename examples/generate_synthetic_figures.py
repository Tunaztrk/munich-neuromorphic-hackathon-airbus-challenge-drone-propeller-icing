"""Generate hand-defined illustrations without datasets or trained models.

The orange curves are intentionally constructed responses, NOT inference outputs
from the project's SNN or GRU. Their closeness does not measure model accuracy.
All inputs and values are generated here in arbitrary, dimensionless coordinates.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / 'assets'


def current_shape(t):
    return np.where(t < .62, .18 + .97*t,
                    .25 + .77*(t-.62))


def ice_shape(t):
    return np.where(t < .62, .12 + .83*(1-np.exp(-3.2*t)),
                    .16 + .74*(t-.62))


def draw(name, title, reference, response):
    t = np.linspace(0., 1., 600)
    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    fig.subplots_adjust(left=.10, right=.98, top=.80, bottom=.22)
    fig.suptitle(title, fontsize=15, fontweight='bold', y=.96)
    fig.text(.5, .865, 'Synthetic illustration — no experimental data or model inference',
             ha='center', fontsize=10, color='#555555')
    ax.plot(t, reference, color='#1f77b4', lw=1.8, label='Synthetic reference')
    ax.plot(t, response, color='#ff7f0e', lw=1.8,
            label='Illustrative response (hand-defined)')
    ax.set(xlabel='Scenario progress (arbitrary scale)',
           ylabel='Signal level (arbitrary scale)', xlim=(0, 1), ylim=(0, 1))
    ax.grid(alpha=.18)
    ax.legend(frameon=False, loc='upper left')
    fig.text(.5, .035,
             'Curve agreement is constructed for explanation; it is not a performance result.',
             ha='center', fontsize=9, color='#555555')
    fig.savefig(OUT/name, dpi=180)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    t = np.linspace(0., 1., 600)
    rng = np.random.default_rng(73)
    current = current_shape(t) + rng.normal(0, .004, len(t))
    # Artificial delayed response makes the change-following limitation visible.
    current_response = current_shape(np.maximum(0, t-.025)) + .006*np.sin(28*t)
    ice = ice_shape(t) + rng.normal(0, .003, len(t))
    ice_response = ice_shape(np.maximum(0, t-.018)) + .012*np.sin(17*t)
    draw('synthetic_current.png', 'Current trajectory: growth, abrupt drop and recovery',
         current, current_response)
    draw('synthetic_ice_area.png', 'Ice-area trajectory: accumulation and partial shedding',
         ice, ice_response)
    print('Generated two synthetic illustrations from equations only.')


if __name__ == '__main__':
    main()
