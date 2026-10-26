"""Exact image costs sharing each frozen local reference across trials."""


def trial_image_costs(trials, batches, reference, render, data_fft, amplitude,
                      precision, kx, ky, xp):
    """Evaluate global trials in the original target/batch summation order.

    References, support and noise weights are fixed throughout the comparison.
    Only the current batch's phase and prediction are allocated on device.
    """
    valid = [not outside.any() and not exceeded
             for _, _, outside, exceeded in trials]
    costs = [0. if usable else float('inf') for usable in valid]
    for target, observations in batches:
        volume = reference(target, observations)
        for index, (camera, displacement, _, _) in enumerate(trials):
            if not valid[index]:
                continue
            shifted = displacement[observations]
            phase = xp.exp(2j*xp.pi*(xp.asarray(shifted[:, 0, None, None])*kx
                                   + xp.asarray(shifted[:, 1, None, None])*ky))
            prediction = render(volume, camera[target])*amplitude[observations, None, None]
            difference = data_fft[observations]*phase-prediction
            costs[index] += float(xp.sum(abs(difference)**2*precision[observations]))
    return costs
