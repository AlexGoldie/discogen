from jaxmarl.wrappers.baselines import SMAXLogWrapper
from jaxmarl.environments.smax import map_name_to_scenario, HeuristicEnemySMAX

def make_env():
    env_kwargs = {
        "see_enemy_actions": True,
        "walls_cause_death": True,
        "attack_mode": "closest",
    }
    scenario = map_name_to_scenario("10m_vs_11m")
    env = HeuristicEnemySMAX(scenario=scenario, **env_kwargs)
    env = SMAXLogWrapper(env)
    return env
