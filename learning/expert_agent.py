import learning.base_agent as base_agent
import learning.expert_model as expert_model

class ExpertAgent(base_agent.BaseAgent):
    NAME = "Expert"

    def __init__(self, config, env, device):
        super().__init__(config, env, device)
        return

    def _build_model(self, config):
        model_config = config["model"]
        self._model = expert_model.ExpertModel(model_config, self._env)
        return
    
    def _get_exp_buffer_length(self):
        return 1

    def _decide_action(self, obs, info):
        norm_obs = self._obs_norm.normalize(obs)
        norm_action_dist = self._model.eval_actor(norm_obs)

        if (self._mode == base_agent.AgentMode.TRAIN):
            norm_a = norm_action_dist.sample()
        elif (self._mode == base_agent.AgentMode.TEST):
            norm_a = norm_action_dist.mode
        else:
            assert(False), "Unsupported agent mode: {}".format(self._mode)
            
        norm_a_logp = norm_action_dist.log_prob(norm_a)

        norm_a = norm_a.detach()
        norm_a_logp = norm_a_logp.detach()
        a = self._a_norm.unnormalize(norm_a)

        a_info = {
            "a_logp": norm_a_logp
        }

        return a, a_info