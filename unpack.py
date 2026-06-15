# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from heapq import heappush, heappop
from scipy.fft import fft2, ifft2
from collections import deque

from gen import *

def unwrap_quality_guided(phi_w):

    h, w = phi_w.shape

    unwrapped = np.zeros_like(phi_w)

    visited = np.zeros_like(phi_w, dtype=bool)

    gx = np.diff(phi_w, axis=1, append=phi_w[:, -1:])
    gy = np.diff(phi_w, axis=0, append=phi_w[-1:, :])

    quality = 1.0 / (np.sqrt(gx**2 + gy**2) + 1e-6)

    seed = np.unravel_index(
        np.argmax(quality),
        quality.shape
    )

    pq = []

    visited[seed] = True
    unwrapped[seed] = phi_w[seed]

    heappush(
        pq,
        (-quality[seed], seed)
    )

    neighbors = [
        (-1,0),
        (1,0),
        (0,-1),
        (0,1)
    ]

    while pq:

        _, (y,x) = heappop(pq)

        for dy,dx in neighbors:

            ny = y+dy
            nx = x+dx

            if ny<0 or ny>=h:
                continue

            if nx<0 or nx>=w:
                continue

            if visited[ny,nx]:
                continue

            delta = phi_w[ny,nx]-phi_w[y,x]

            delta = (delta+np.pi)%(2*np.pi)-np.pi

            unwrapped[ny,nx] = (
                unwrapped[y,x]
                + delta
            )

            visited[ny,nx] = True

            heappush(
                pq,
                (-quality[ny,nx], (ny,nx))
            )

    return unwrapped


def unwrap_least_squares(phi_w):

    dx = np.angle(
        np.exp(
            1j*np.diff(
                phi_w,
                axis=1,
                append=phi_w[:,-1:]
            )
        )
    )

    dy = np.angle(
        np.exp(
            1j*np.diff(
                phi_w,
                axis=0,
                append=phi_w[-1:,:]
            )
        )
    )

    rhs = (
        np.diff(
            dx,
            axis=1,
            prepend=dx[:,0:1]
        )
        +
        np.diff(
            dy,
            axis=0,
            prepend=dy[0:1,:]
        )
    )

    h,w = phi_w.shape

    yy,xx = np.meshgrid(
        np.arange(h),
        np.arange(w),
        indexing='ij'
    )

    denom = (
        2*np.cos(
            2*np.pi*xx/w
        )
        +
        2*np.cos(
            2*np.pi*yy/h
        )
        -4
    )

    denom[0,0] = 1

    U = fft2(rhs)

    U /= denom

    U[0,0] = 0

    result = np.real(
        ifft2(U)
    )

    return result

def unwrap_branch_cut(phi_w):

    gx = np.diff(
        phi_w,
        axis=1,
        append=phi_w[:,-1:]
    )

    gy = np.diff(
        phi_w,
        axis=0,
        append=phi_w[-1:,:]
    )

    quality = (
        np.abs(gx)
        +
        np.abs(gy)
    )

    threshold = np.percentile(
        quality,
        90
    )

    cut_mask = (
        quality > threshold
    )

    h,w = phi_w.shape

    unwrapped = np.zeros_like(phi_w)

    visited = np.zeros_like(
        phi_w,
        dtype=bool
    )

    seed = (h//2,w//2)

    q = deque([seed])

    visited[seed] = True
    unwrapped[seed] = phi_w[seed]

    while q:

        y,x = q.popleft()

        for dy,dx in [
            (-1,0),
            (1,0),
            (0,-1),
            (0,1)
        ]:

            ny = y+dy
            nx = x+dx

            if ny<0 or ny>=h:
                continue

            if nx<0 or nx>=w:
                continue

            if cut_mask[ny,nx]:
                continue

            if visited[ny,nx]:
                continue

            delta = (
                phi_w[ny,nx]
                - phi_w[y,x]
            )

            delta = (
                (delta+np.pi)
                %(2*np.pi)
                - np.pi
            )

            unwrapped[ny,nx] = (
                unwrapped[y,x]
                + delta
            )

            visited[ny,nx] = True

            q.append((ny,nx))

    return unwrapped

def unpack(phi_true, phi_wrapped):
    phi_qg = unwrap_quality_guided(phi_wrapped)
    phi_ls = unwrap_least_squares(phi_wrapped)
    phi_bc = unwrap_branch_cut(phi_wrapped)

    fig = plt.figure(figsize=(16,12))
    
    titles = [
        ("True",phi_true),
        ("Wrapped",phi_wrapped),
        ("Quality Guided",phi_qg),
        ("Least Squares",phi_ls),
        ("Branch Cut",phi_bc)
    ]
    
    for i,(name,data) in enumerate(
        titles,
        start=1
    ):
    
        ax = fig.add_subplot(
            2,
            3,
            i,
            projection='3d'
        )
    
        X,Y = np.meshgrid(
            np.arange(data.shape[1]),
            np.arange(data.shape[0])
        )
    
        ax.plot_surface(
            X,
            Y,
            data,
            linewidth=0
        )
    
        ax.set_title(name)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
        
    phase1 = gen_wave()
    phase2 = gen_wave_edge(phase1)
    I1,I2,I3,I4 = gen_interferogram(phase1)

    I1n,I2n,I3n,I4n = gen_wave_noise((I1,I2,I3,I4))
    phase3 =  extract_wrapped_phase(I1n,I2n,I3n,I4n)
    
    plot_profile_compare(phase1, phase2, phase3)
    
    unpack(phase1, phase2)
    unpack(phase1, phase3)
    unpack(phase1, np.angle(np.exp(1j * phase1)))
    