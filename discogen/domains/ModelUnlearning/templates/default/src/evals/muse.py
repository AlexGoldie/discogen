from evals.base import Evaluator


class MUSEEvaluator(Evaluator):
    def __init__(self, eval_cfg, **kwargs):
        super().__init__("MUSE", eval_cfg, **kwargs)
