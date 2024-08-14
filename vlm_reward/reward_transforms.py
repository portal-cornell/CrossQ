from scipy.ndimage._filters import correlate1d, _gaussian_kernel1d

def half_gaussian_filter_1d(x, sigma, smooth_last_N=False, axis=-1, order=0, output=None,
                      mode="reflect", cval=0.0, truncate=4.0, *, radius=None):
    """
    filters x using the half normal distribution, defined by sigma.
    """
    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sd + 0.5)
    if radius is not None:
        lw = radius

    # Since we are calling correlate, not convolve, revert the kernel
    weights = _gaussian_kernel1d(sigma, order, lw)[::-1]
    weights[:(len(weights) - 1) // 2] = 0
    weights /= sum(weights)

    filtered = correlate1d(x, weights, axis, output, mode, cval, 0)

    if not smooth_last_N:
        filtered[..., -lw:] = x[..., -lw:]
        
    return filtered 


if __name__=="__main__":
    x = torch.load('debugging/testing_after/rewards.pt')

    y2 = half_gaussian_filter_1d(x[None], sigma=20)
    from vlm_reward.utils.utils import rewards_matrix_heatmap
    rewards_matrix_heatmap(y2, 'heatmap.png')