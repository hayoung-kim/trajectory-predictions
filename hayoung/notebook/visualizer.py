import numpy as np
import torch
import matplotlib.pyplot as plt
import io
import PIL

from torch.distributions.multivariate_normal import MultivariateNormal


COLORS = """#377eb8
#ff7f00
#4daf4a
#984ea3
#ffd54f""".split('\n')


def batch_center_crop(arr, target_h, target_w):
    if len(arr.shape) > 3:
        *_, h, w, _ = arr.shape
        center = (h // 2, w // 2)
        return arr[..., center[0] - target_h // 2:center[0] + target_h // 2, center[1] - target_w // 2:center[1] + target_w // 2, :]
    elif len(arr.shape) == 3:
        h, w, _ = arr.shape
        center = (h // 2, w // 2)
        return arr[center[0] - target_h // 2:center[0] + target_h // 2, center[1] - target_w // 2:center[1] + target_w // 2, :]
    else:
        raise NotImplementedError
        
        
def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = np.array(PIL.Image.open(buf))

    return image


def plot_single_batch(
    player_past, 
    other_pasts, 
    player_future, 
    other_futures, 
    fig=None, 
    ax=None, 
    overhead_features=None, 
    COLORS=COLORS):
    
    if fig is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    
    # lidar
    if overhead_features is not None:
        overhead_features = np.transpose(overhead_features, (1,2,0))
        bevs = batch_center_crop(overhead_features, 200, 200)

        below_slice = slice(0,2)
        above_slice = slice(2,100)

        bev0 = np.array(bevs[..., below_slice], np.float64).sum(axis=-1)
        grey_pixels = np.where(bev0 > 0.01)
        bev1 = np.array(bevs[..., above_slice], np.float64).sum(axis=-1)
        red_pixels = np.where(bev1 > 0.01)

        image = 255 * np.ones(bev0.shape + (3,), dtype=np.uint8)
        image[grey_pixels] = [153, 153, 153]
        image[red_pixels] = [228, 27, 28]

        ax.imshow(image)
        H, W = bev0.shape
        
    else:
        H, W = 200.0, 200.0
        
    feature_pixels_per_meter = 2.0
    local2gridx = lambda x: W //2 + x * feature_pixels_per_meter
    local2gridy = lambda y: H //2 + y * feature_pixels_per_meter
    
    # ego
    for past in player_past:
        x, y = past
        ax.plot(
            local2gridx(x), local2gridy(y), 'b^', alpha=0.5, color=COLORS[0], markeredgecolor='k', markersize=10
        )

    for exp in player_future:
        x,y = exp
        ax.plot(
            local2gridx(x), local2gridy(y), 'bs', alpha=0.5, color=COLORS[0], markeredgecolor='k', markersize=10
        )
    
    # others
    n_others = other_futures.shape[0]
    for other in range(n_others):
        other_experts = other_futures[other]
        x, y = np.split(other_experts, 2, axis=-1)
        x, y = x.reshape(-1), y.reshape(-1)

        ax.plot(
            local2gridx(x), local2gridy(y), 'bs', alpha=0.5, color=COLORS[other+1], markeredgecolor='k', markersize=10
        )

        _other_pasts = other_pasts[other]
        x, y = np.split(_other_pasts, 2, axis=-1)
        x, y = x.reshape(-1), y.reshape(-1)

        ax.plot(
            local2gridx(x), local2gridy(y), 'b^', alpha=0.5, color=COLORS[other+1], markeredgecolor='k', markersize=10
        )
    
    return fig, ax

def plot_predictions_single_batch(fig, ax, samples, COLORS=COLORS):
    n_samples, A, T, _ = samples.shape
    
    H = 200.0
    feature_pixels_per_meter = 2.0
    
    local2grid = lambda x: H //2 + x * feature_pixels_per_meter
    
    for n in range(n_samples):
        for a in range(A):
            xs, ys = [], []
            for t in range(T):
                x, y = samples[n, a, t, :]
                xs.append(x)
                ys.append(y)
            xs = np.array(xs)
            ys = np.array(ys)
            ax.plot(
                local2grid(xs), local2grid(ys), '-o', color=COLORS[a], alpha=0.5, markeredgecolor='k', markersize=7
            )

    return fig, ax
    