# custom_env/__init__.py
from gymnasium.envs.registration import register

register(
    id='MyCustomEnv-v0',
    entry_point='custom_env.my_custom_env:MyCustomEnv',
)
