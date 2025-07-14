from stable_baselines3.common.env_checker import check_env

transition_models = np.array([
                        1, 0, 0, 1,
                        0.9, 0.1, 0, 1,
                        1, 0, 0, 1,
                        1, 0, 0, 1
                    ]).reshape(2, 2, 2, 2)

reward_function = np.array([
                  0.736, 0.735,
                  0.736, 0.8540772
                    ]).reshape(2, 2)

env = AdaptiveManagement(transition_models,reward_function)
# If the environment don't follow the interface, an error will be thrown
check_env(env, warn=True)