from typing import List, Tuple
import torch

class KVCache:
    def __init__(self, max_size:int) -> None:
        """
        """
        
        self.max_size = max_size
        
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    def has_item(self, layer_idx) -> bool:
        return len(self.key_cache) > layer_idx

    def num_items(self) -> int:
        if len(self.key_cache) == 0:
            return 0
        else:
            # The shape of the key_cache is [Batch_Size, Seq_Len, Num_Heads_KV, Head_Dim]
            return self.key_cache[0].shape[1]

    def get(self, layer_idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.key_cache) <= layer_idx:
            # If we never added anything to the KV-Cache of this layer, let's create it.
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # ... otherwise we concatenate the new keys with the existing ones.
            # each tensor has shape: [Batch_Size, Seq_Len, Num_Heads_KV, Head_Dim]
            # slicing with [-self.max_size:] to constrain growth of cache
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx][-self.max_size:], key_states], dim=1
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx][-self.max_size:], value_states], dim=1
            )

        # ... and then we return all the existing keys + the new ones.
        return self.key_cache[layer_idx], self.value_cache[layer_idx]