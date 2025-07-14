from gym.envs.registration import register

from gym_AdaptiveManagement.envs.AdaptiveManagement_base import AdaptiveManagement


register(
    id="adaptive-v0",
    entry_point="gym_AdaptiveManagement.envs:AdaptiveManagement",
)
