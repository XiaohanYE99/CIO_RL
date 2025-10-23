import torch
from gym.wrappers import RecordVideo
from rlkit.envs.pearl_envs import HumanoidDirEnv, HumanoidClimbEnv

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic,MakeDeterministicDiscrete,CategoricalPolicy,MixedTanhGaussianPolicy
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

from rlkit.torch.pytorch_util import set_gpu_mode
set_gpu_mode(True)
Load=False

def experiment(variant):
    expl_env = NormalizedBoxEnv(HumanoidClimbEnv())
    eval_env = NormalizedBoxEnv(HumanoidClimbEnv())
    active_contact_dim=eval_env.active_contact_dim
    deactive_contact_dim=eval_env.deactive_contact_dim
    contact_dim = eval_env.active_contact_dim*eval_env.deactive_contact_dim
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size+contact_dim

    M = variant['layer_size']
    #contact net
    qfc1 = ConcatMlp(
        input_size=obs_dim + active_contact_dim,
        output_size=1,
        hidden_sizes=[M, M],
    ).to(ptu.device)
    qfc2 = ConcatMlp(
        input_size=obs_dim + active_contact_dim,
        output_size=1,
        hidden_sizes=[M, M],
    ).to(ptu.device)
    target_qfc1 = ConcatMlp(
        input_size=obs_dim + active_contact_dim,
        output_size=1,
        hidden_sizes=[M, M],
    ).to(ptu.device)
    target_qfc2 = ConcatMlp(
        input_size=obs_dim + active_contact_dim,
        output_size=1,
        hidden_sizes=[M, M],
    ).to(ptu.device)
    policy_c = CategoricalPolicy(
        obs_dim=obs_dim,
        action_dim=[active_contact_dim,deactive_contact_dim],
        hidden_sizes=[M, M],
    ).to(ptu.device)
    #policy net
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
    ).to(ptu.device)

    if Load: 
        policy.load_state_dict(torch.load("policy.pth"))

    eval_policy = MakeDeterministic(policy)
    eval_policy_c = MakeDeterministicDiscrete(policy_c)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
        eval_policy_c,
        render=True,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
        policy_c,
        render=False,
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        policy_c=policy_c,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        qfc1=qfc1,
        qfc2=qfc2,
        target_qfc1=target_qfc1,
        target_qfc2=target_qfc2,
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()




if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algorithm="SAC",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(1E6),
        algorithm_kwargs=dict(
            num_epochs=30000,
            num_eval_steps_per_epoch=500,
            num_trains_per_train_loop=32,#32,
            num_expl_steps_per_train_loop=500,#500,
            min_num_steps_before_training=5000,#5000,
            max_path_length=500,
            batch_size=2048,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=5,
            use_automatic_entropy_tuning=True,
        ),
    )
    setup_logger('name-of-experiment', variant=variant)
    # ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant)
