from evals.base import Evaluator


class TOFUEvaluator(Evaluator):
    def __init__(self, eval_cfg, **kwargs):
        super().__init__("TOFU", eval_cfg, **kwargs)
