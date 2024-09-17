import argparse
import numpy as np
import os
import sys
import time
import torch

import envs.env_builder as env_builder
import learning.agent_builder as agent_builder
import util.mp_util as mp_util
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
    parser.add_argument("--master_port", dest="master_port", type=int, default=None)
    parser.add_argument("--num_workers", dest="num_workers", type=int, default=1)
    
    args = parser.parse_args()

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
    if (mp_util.is_root_proc()):
        output_dir = os.path.dirname(out_model_file)
        if (output_dir != "" and (not os.path.exists(output_dir))):
            os.makedirs(output_dir, exist_ok=True)
        
        if (int_output_dir != "" and (not os.path.exists(int_output_dir))):
            os.makedirs(int_output_dir, exist_ok=True)
    return

def set_rand_seed(args):
    rand_seed = args.rand_seed

    if (rand_seed is None):
        rand_seed = np.uint64(time.time() * 256)
        
    rand_seed += np.uint64(41 * mp_util.get_proc_rank())
    print("Setting seed: {}".format(rand_seed))
    util.set_rand_seed(rand_seed)
    return

def run(rank, num_procs, master_port, args):
    mode = args.mode
    device = args.device
    visualize = args.visualize
    log_file = args.log_file
    out_model_file = args.out_model_file
    int_output_dir = args.int_output_dir
    model_file = args.model_file
    
    mp_util.init(rank, num_procs, device, master_port)

    set_rand_seed(args)
    set_np_formatting()

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


def main(argv):
    args = load_args(argv)
    master_port = args.master_port
    num_workers = args.num_workers
    assert(num_workers > 0)
    
    # if master port is not specified, then pick a random one
    if (master_port is None):
        master_port = np.random.randint(6000, 7000)

    torch.multiprocessing.set_start_method("spawn")

    processes = []
    for i in range(num_workers - 1):
        rank = i + 1
        proc = torch.multiprocessing.Process(target=run, args=[rank, num_workers, master_port, args])
        proc.start()
        processes.append(proc)

    run(0, num_workers, master_port, args)

    for proc in processes:
        proc.join()
       
    return

if __name__ == "__main__":
    main(sys.argv)
