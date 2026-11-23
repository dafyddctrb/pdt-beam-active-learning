"""Cumulative true-model call counts, shared by the web app and the command-line analysis."""
import numpy as np


def cumulative_calls(method, times, calls, deterioration_times):
    """True-model calls spent up to and including each time in `times`.

    The stored counter `nr_likelihood_calls` is not cumulative for the physics-model
    methods: Batch PM stores the cost of that step's batch update alone (each step starts
    from scratch), and Seq PM resets the counter when a deterioration reading restarts the
    chain and the displacements are replayed. The surrogate methods store a cumulative
    count already. The cost spent at step t is therefore
      Batch PM: calls[t];
      Seq PM:   calls[t] at a deterioration time (the replay), else calls[t] - calls[t-1];
      SRG:      calls[t] - calls[t-1].

    method: "Seq PM", "Seq SRG", "Batch PM" or "Batch SRG"; times and calls: one value
    per time step, sorted by time.
    """
    times = np.asarray(times)
    c = np.asarray(calls, dtype=float)
    if method == "Batch PM":
        step = c.copy()
    else:
        step = np.diff(c, prepend=0.0)
        if method == "Seq PM":
            reset = np.isin(times, [int(x) for x in deterioration_times])
            step[reset] = c[reset]
    return np.cumsum(step)
