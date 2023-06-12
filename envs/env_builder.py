import yaml

import envs.atari_env as atari_env
import envs.env_dm as env_dm

def build_env(env_file, device, visualize):
    env_config = load_env_file(env_file)

    env_name = env_config["env_name"]
    print("Building {} env".format(env_name))
    
    if (env_name.startswith("dm_")):
        env = env_dm.DMEnv(config=env_config, device=device, visualize=visualize)
    elif (env_name.startswith("atari_")):
        env = atari_env.AtariEnv(config=env_config, device=device, visualize=visualize)
    else:
        assert(False), "Unsupported env: {}".format(env_name)

    return env

def load_env_file(file):
    with open(file, "r") as stream:
        env_config = yaml.safe_load(stream)
    return env_config
