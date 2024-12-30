from gym.envs.registration import register

register(
    id='Swap-v0',
    entry_point='place_env.swapMDP:SwapPlacement',
)

register(
    id='Classic-v0',
    entry_point='place_env.classicMDP:ClassicPlacement'
)