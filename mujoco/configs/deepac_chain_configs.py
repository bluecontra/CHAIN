# re-construct from CleanRL
from dataclasses import dataclass

@dataclass
class Args:
    # exp_name: str = os.path.basename(__file__)[: -len(".py")]
    # """the name of this experiment"""
    invest: bool = False
    """if toggled, stats for policy churn investigation will be recorded"""
    invest_window_size: int = 10
    """the window size of step to investigate policy churn"""
    invest_interval: int = 1
    """the interval to keep old policies in invest window"""
    alg: str = "td3_pcr"
    """the name of the algorithm to run"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = False
    """if toggled, cuda will be enabled by default"""
    gpu_no: str = '-1'
    """designate the gpu with corresponding number to run the exp"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    # wandb_project_name: str = "cleanRL"
    # """the wandb's project name"""
    # wandb_entity: str = None
    # """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_starts: int = int(5e3)
    """timestep to start learning"""
    eval_freq_timesteps: int = 50000
    """timestep to eval learned policy"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    policy_noise: float = 0.2
    """the scale of policy noise"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""

    # Regularization specific arguments
    # NOTE - refer to Figure 13-15 in the paper for the inffluence of different choices of manual regularization coeffs
    reg_coef: float = 0.0005
    """coefficient of policy churn regularization term."""
    v_reg_coef: float = 0.1
    """coefficient of value churn regularization term."""

    reg_his_idx: int = 2
    """the index (in reverse order) of historical policy used for regularization"""