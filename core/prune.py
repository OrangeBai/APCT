from pytorch_lightning.callbacks.pruning import *
import torch.nn.utils.prune as pytorch_prune


class APPruning(ModelPruning):
    def __init__(self, pruning_fn):
        super().__init__(pruning_fn)

    def _apply_global_pruning(self, amount: float) -> None:
        pytorch_prune.global_unstructured(
            self._parameters_to_prune, pruning_method=self.pruning_fn, **self._resolve_global_kwargs(amount)
        )
