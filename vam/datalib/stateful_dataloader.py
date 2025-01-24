from typing import Any, Dict, List, Union

import torch.distributed as dist
from torchdata.stateful_dataloader import StatefulDataLoader as _StatefulDataLoader

Args = List[Any]
Kwargs = Dict[str, Any]
StateDict = Dict[str, Any]


class StatefulDataLoader(_StatefulDataLoader):

    def __init__(self, *args: Args, is_finetuning: bool = False, **kwargs: Kwargs) -> None:
        super().__init__(*args, **kwargs)
        # is distributed
        self.is_distributed = dist.is_available() and dist.is_initialized()
        if self.is_distributed:
            self.world_size = dist.get_world_size()
        else:
            self.world_size = 1

        self.is_finetuning = is_finetuning

    def _gather_state_dict(self, state_dict: StateDict) -> List[StateDict]:
        """
        Gather the state_dict from all ranks to the master rank
        """
        # Create a list to store the state_dicts from all ranks
        object_gather_list = [None for _ in range(self.world_size)]

        # Gather the state_dict from all ranks
        dist.all_gather_object(object_gather_list, state_dict)

        # Return the state_dict from the master rank
        return object_gather_list

    def state_dict(self) -> Union[StateDict, List[StateDict]]:
        # Get the state dict on each rank()
        state_dict = super().state_dict()

        # If we are not in distributed mode, return the state_dict as is
        if not self.is_distributed:
            return state_dict

        # If we are in distributed mode, we need to gather the state_dict from all ranks
        # only on the master rank
        state_dict = self._gather_state_dict(state_dict)
        return state_dict

    def load_state_dict(self, state_dict: Union[StateDict, List[StateDict]]) -> None:
        if self.is_finetuning:
            # If we are finetuning, we don't need to load the state_dict
            return

        # If we are not in distributed mode, load the state_dict as is
        if not self.is_distributed:
            super().load_state_dict(state_dict)
            return

        # get the rank of the current process
        rank = dist.get_rank()

        # get the state_dict for the current loader
        # contrarily to state_dict(), load_state_dict() is called on all ranks
        state_dict = state_dict[rank]

        # Load the state_dict on all ranks
        super().load_state_dict(state_dict)
