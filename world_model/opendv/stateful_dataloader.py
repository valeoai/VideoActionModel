from typing import Any, Dict, List, Union

import torch.distributed as dist
from torchdata.stateful_dataloader import StatefulDataLoader as _StatefulDataLoader

Args = List[Any]
Kwargs = Dict[str, Any]
StateDict = Dict[str, Any]


class StatefulDataLoader(_StatefulDataLoader):

    def _gather_state_dict(self, state_dict: StateDict) -> List[StateDict]:
        """
        Gather the state_dict from all ranks to the master rank
        """
        # is distributed
        self.is_distributed = dist.is_available() and dist.is_initialized()
        if self.is_distributed:
            self.is_master = dist.get_rank() == 0
            self.world_size = dist.get_world_size()
        else:
            self.is_master = True
            self.world_size = 1

        # Create a list to store the state_dicts from all ranks
        if self.is_master():
            world_size = self.get_world_size()
            object_gather_list = [None for _ in range(world_size)]
        else:
            object_gather_list = None

        # Gather the state_dict from all ranks
        dist.gather_object(state_dict, object_gather_list)

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
