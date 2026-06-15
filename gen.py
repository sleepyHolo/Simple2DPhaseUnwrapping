# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


# ==================================================
# 1. 生成真实相位
# ==================================================

def gen_wave(
        size=256,
        phase_scale=8*np.pi
):

    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)

    X, Y = np.meshgrid(x, y)

    phase = phase_scale * (X**2 + Y**2)

    return phase


# ==================================================
# 2. 添加随机台阶
# ==================================================

def gen_wave_edge(phase):

    phase = phase.copy()

    ny, nx = phase.shape

    pos = np.random.randint(nx//4, 3*nx//4)

    step_height = np.random.randint(2, 6) * 2*np.pi

    phase[:, pos:] += step_height

    return phase


# ==================================================
# 3. 四步相移干涉图
# ==================================================

def gen_interferogram(
        phase,
        A=1.0,
        B=1.0
):

    I1 = A + B*np.cos(phase)
    I2 = A + B*np.cos(phase + np.pi/2)
    I3 = A + B*np.cos(phase + np.pi)
    I4 = A + B*np.cos(phase + 3*np.pi/2)

    return I1, I2, I3, I4


# ==================================================
# 4. 添加失相关区域
# ==================================================

def gen_wave_noise(
        I_list,
        radius_ratio=0.15,
        sigma=0.3
):

    I_out = [img.copy() for img in I_list]
    
    ny, nx = I_out[0].shape

    X, Y = np.meshgrid(
        np.arange(nx),
        np.arange(ny)
    )

    cx = np.random.randint(nx//4, 3*nx//4)
    cy = np.random.randint(ny//4, 3*ny//4)

    radius = int(min(nx, ny)*radius_ratio)

    mask = (
        (X-cx)**2 +
        (Y-cy)**2
        <= radius**2
    )

    for img in I_out:

        img += sigma*np.random.randn(*img.shape)

        img[mask] = (
            np.mean(img)
            + sigma*5*np.random.randn(
                np.count_nonzero(mask)
            )
        )

    return tuple(I_out)


# ==================================================
# 5. 提取包裹相位
# ==================================================

def extract_wrapped_phase(
        I1,
        I2,
        I3,
        I4
):

    phase_wrapped = np.arctan2(
        I4-I2,
        I1-I3
    )

    return phase_wrapped


def plot_phase_profile(
        phase_true,
        phase_wrapped,
        title=""
):

    row = phase_true.shape[0] // 2

    x = np.arange(
        phase_true.shape[1]
    )

    plt.figure(figsize=(10,5))

    plt.plot(
        x,
        phase_true[row],
        label="True Phase"
    )

    plt.plot(
        x,
        phase_wrapped[row],
        label="Wrapped Phase"
    )

    plt.xlabel("Pixel")
    plt.ylabel("Phase(rad)")
    plt.title(title)

    plt.legend()
    plt.grid(True)

    plt.show()


def plot_profile_compare(
        phase_true,
        wrapped_clean,
        wrapped_noise
):

    row = phase_true.shape[0] // 2

    x = np.arange(
        phase_true.shape[1]
    )

    plt.figure(figsize=(12,6))

    plt.plot(
        x,
        phase_true[row],
        label="True Phase"
    )

    plt.plot(
        x,
        wrapped_clean[row],
        label="Wrapped Clean"
    )

    plt.plot(
        x,
        wrapped_noise[row],
        label="Wrapped Noisy"
    )

    plt.legend()

    plt.xlabel("Pixel")
    plt.ylabel("Phase(rad)")

    plt.title(
        "Phase Profile Comparison"
    )

    plt.grid(True)

    plt.show()
    
if __name__ == '__main__':
    
    phase1 = gen_wave()
    phase2 = gen_wave_edge(phase1)
    I1,I2,I3,I4 = gen_interferogram(phase1)

    I1n,I2n,I3n,I4n = gen_wave_noise((I1,I2,I3,I4))
    phase3 =  extract_wrapped_phase(I1n,I2n,I3n,I4n)
    
    plot_profile_compare(phase1, phase2, phase3)