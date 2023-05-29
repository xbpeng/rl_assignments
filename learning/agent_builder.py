import yaml

import learning.expert_agent as expert_agent

import a1.bc_agent as bc_agent
import a2.cem_agent as cem_agent
import a2.pg_agent as pg_agent

def build_agent(agent_file, env, device):
    agent_config = load_agent_file(agent_file)
    
    agent_name = agent_config["agent_name"]
    print("Building {} agent".format(agent_name))

    if (agent_name == bc_agent.BCAgent.NAME):
        agent = bc_agent.BCAgent(config=agent_config, env=env, device=device)
    elif (agent_name == expert_agent.ExpertAgent.NAME):
        agent = expert_agent.ExpertAgent(config=agent_config, env=env, device=device)
    elif (agent_name == cem_agent.CEMAgent.NAME):
        agent = cem_agent.CEMAgent(config=agent_config, env=env, device=device)
    elif (agent_name == pg_agent.PGAgent.NAME):
        agent = pg_agent.PGAgent(config=agent_config, env=env, device=device)
    else:
        assert(False), "Unsupported agent: {}".format(agent_name)

    return agent

def load_agent_file(file):
    with open(file, "r") as stream:
        agent_config = yaml.safe_load(stream)
    return agent_config
