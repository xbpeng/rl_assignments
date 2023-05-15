import argparse
import numpy as np
import os
import sys
import yaml
import envs.env_builder as env_builder
import learning.agent_builder as agent_builder
import util.util as util

def set_np_formatting():
    np.set_printoptions(edgeitems=30, infstr='inf',
                        linewidth=4000, nanstr='nan', precision=2,
                        suppress=False, threshold=10000, formatter=None)
    return

def load_args(argv):
    parser = argparse.ArgumentParser(description="Train or test control policies.")
    
    parser.add_argument("--rand_seed", dest="rand_seed", type=int, default=None)
    parser.add_argument("--mode", dest="mode", type=str, default="train")
    parser.add_argument("--visualize", dest="visualize", action="store_true", default=False)
    parser.add_argument("--env_config", dest="env_config")
    parser.add_argument("--agent_config", dest="agent_config")
    parser.add_argument("--device", dest="device", type=str, default="cpu")
    parser.add_argument("--log_file", dest="log_file", type=str, default="output/log.txt")
    parser.add_argument("--out_model_file", dest="out_model_file", type=str, default="output/model.pt")
    parser.add_argument("--int_output_dir", dest="int_output_dir", type=str, default="")
    parser.add_argument("--model_file", dest="model_file", type=str, default="")
    parser.add_argument("--max_samples", dest="max_samples", type=np.int64, default=np.iinfo(np.int64).max)
    parser.add_argument("--test_episodes", dest="test_episodes", type=np.int64, default=np.iinfo(np.int64).max)
    
    args = parser.parse_args()

    if (args.rand_seed is not None):
        util.set_rand_seed(args.rand_seed)

    return args

def build_env(args, device, visualize):
    env_file = args.env_config
    env = env_builder.build_env(env_file, device, visualize)
    return env

def build_agent(args, env, device):
    agent_file = args.agent_config
    agent = agent_builder.build_agent(agent_file, env, device)
    return agent

def train(agent, max_samples, out_model_file, int_output_dir, log_file):
    agent.train_model(max_samples=max_samples, out_model_file=out_model_file, 
                      int_output_dir=int_output_dir, log_file=log_file)
    return

def test(agent, test_episodes):
    result = agent.test_model(num_episodes=test_episodes)
    print("Mean Return: {}".format(result["mean_return"]))
    print("Mean Episode Length: {}".format(result["mean_ep_len"]))
    print("Episodes: {}".format(result["episodes"]))
    return result

def create_output_dirs(out_model_file, int_output_dir):
    output_dir = os.path.dirname(out_model_file)
    if (output_dir != "" and (not os.path.exists(output_dir))):
        os.makedirs(output_dir, exist_ok=True)
        
    if (int_output_dir != "" and (not os.path.exists(int_output_dir))):
        os.makedirs(int_output_dir, exist_ok=True)
    return

def main(argv):
    set_np_formatting()

    args = load_args(argv)

    mode = args.mode
    device = args.device
    visualize = args.visualize
    log_file = args.log_file
    out_model_file = args.out_model_file
    int_output_dir = args.int_output_dir
    model_file = args.model_file

    create_output_dirs(out_model_file, int_output_dir)

    env = build_env(args, device, visualize)
    agent = build_agent(args, env, device)

    if (model_file != ""):
        agent.load(model_file)

    if (mode == "train"):
        max_samples = args.max_samples
        train(agent=agent, max_samples=max_samples, out_model_file=out_model_file, 
              int_output_dir=int_output_dir, log_file=log_file)
    elif (mode == "test"):
        test_episodes = args.test_episodes
        test(agent=agent, test_episodes=test_episodes)
    else:
        assert(False), "Unsupported mode: {}".format(mode)
    return

if __name__ == "__main__":
    main(sys.argv)
