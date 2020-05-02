import numpy as np
from typing import Any, List, Tuple, Union, Iterator
import torch

from detectron2.layers import cat


class Interactions:
    """
    Stores human-object interaction annotation data. GT instances have a `gt_interactions`
    property containing the <person, object, action> annotation of each instances. This tensor
    has shape (N, M, K) where N(M) is the number of instances and K is the number of actions.
    If i-th instance is interacting with j-th instance with k-th action. The (i, j, k) entry
    in the tensor will be 1, otherwise there will be 0.
    """

    def __init__(self, interactions: Union[torch.Tensor, np.ndarray]):
        """
        Arguments:
            interactions: A Tensor, numpy array of the interaction annotations.
                The shape should be (N, M, K) where N(M) is the number of instances,
                and K is the number of actions.
        """
        device = interactions.device if isinstance(interactions, torch.Tensor) else torch.device("cpu")
        interactions = torch.as_tensor(interactions, dtype=torch.float32, device=device)
        if interactions.numel() == 0:
            interactions = interactions.reshape((0, 4)).to(dtype=torch.float32, device=device)
        assert interactions.dim() == 3, interactions.size()

        self.tensor = interactions

    def clone(self) -> "Interactions":
        """
        Clone the Interactions.

        Returns:
            Interaction
        """
        return Interactions(self.tensor.clone())

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Interactions":
        """
        Create a new `Interactions` by indexing on this `Interactions`.

        The following usage are allowed:

        1. An integer. It will return an object with only one instance.
        2. A slice. It will return an object with the selected instances.
        3. A vector mask of type BoolTensor, whose length is num_instances.
           It will return an object with the instances whose mask is nonzero.
        """
        if isinstance(item, int):
            return Interactions(self.tensor[item].view(1, -1))
        x = self.tensor[item]
        assert x.dim() == 3, "Indexing on Boxes with {} failed to return a matrix!".format(item)
        return Interactions(x)

    def __repr__(self) -> str:
        return "Interactions(" + str(self.tensor) + ")"

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    @staticmethod
    def cat(interactions_list: List["Interactions"]) -> "Interactions":
        """
        Concatenates a list of Interactions into a single Interaction

        Arguments:
            interactions_list (list[Interactions])

        Returns:
            Interactions: the concatenated Interactions
        """
        assert isinstance(interactions_list, (list, tuple))
        assert len(interactions_list) > 0
        assert all(isinstance(x, Interactions) for x in interactions_list)

        cat_interactions = type(interactions_list[0])(cat([x.tensor for x in interactions_list], dim=0))
        return cat_interactions

    def take(self, index1: int, index2: int) -> torch.Tensor:
        """
        Take a tensor given the specific instance indices.
        `Interactions` has a tensor of shape (N, N, K) where N is the number of
            instances and K is the number of actions. Given two indices of instances,
            it returns a tensor of action annotations (K, )

        Arguments:
            index1 (int): index along the first axis
            index2 (int): index along the second axis

        Returns:
            torch.Tensor: action annotations between two instances
        """
        return self.tensor[index1, index2]

    def __iter__(self) -> Iterator[torch.Tensor]:
        """
        Yield a box as a Tensor of shape (N,K) at a time.
        """
        yield from self.tensor

    