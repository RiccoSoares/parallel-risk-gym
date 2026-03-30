from parallel_risk.env.parallel_risk_env import ParallelRiskEnv


def env(**kwargs):
    """Create a ParallelRiskEnv instance

    This is the recommended PettingZoo entry point pattern.

    Args:
        **kwargs: Arguments to pass to ParallelRiskEnv constructor

    Returns:
        ParallelRiskEnv: A new environment instance
    """
    return ParallelRiskEnv(**kwargs)
