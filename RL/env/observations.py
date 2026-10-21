"""Assemble critic observations without exposing privileged state to actors."""
import numpy as np


def build_critic_observations(policy_obs, infos):
    """Append per-env privileged tails to the matching policy observations.

    Callers must pass reset info for environments that automatically reset,
    since terminal privileged state does not describe the returned reset obs.
    Legacy environments without privileged state return None.
    """
    if len(infos) != len(policy_obs):
        raise ValueError('Expected one observation info dict per environment')
    tails = [info.get('critic_extra_obs') for info in infos]
    if all(tail is None for tail in tails):
        return None
    if any(tail is None for tail in tails):
        raise ValueError('Missing critic_extra_obs for part of the vector environment')
    tails = [np.asarray(tail, dtype=np.float32) for tail in tails]
    if any(tail.ndim != 1 for tail in tails):
        raise ValueError('critic_extra_obs must be a one-dimensional vector')
    extra = np.stack(tails)
    return np.concatenate((np.asarray(policy_obs, dtype=np.float32), extra), axis=-1)
