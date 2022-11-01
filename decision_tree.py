from __future__ import annotations
import numpy as np
from typing import Callable, Optional


class Node:
    __slots__ = ["parent", "node_label"]

    def __init__(self, node_label: str) -> None:
        self.node_label = node_label


class DecisionTreeLeaf(Node):
    __slots__ = ["frequency"]

    def __init__(
        self,
        classification: str,
    ):
        super().__init__(classification)
        self.frequency = 0

    def get_classification(self) -> str:
        return self.node_label


class DecisionTreeNode(Node):
    __slots__ = [
        "function",
        "left_node",
        "right_node",
    ]

    def __init__(
        self,
        function: Callable[[np.ndarray], bool],
        node_label: str,
    ) -> None:
        """
        Class representing a node of a decision tree

        :param function: Function to evaluate the result of the node
        :node_label: The representation of the decision this node makes
        """
        self.function = function
        self.node_label = node_label
        self.left_node: Optional[Node] = None
        self.right_node: Optional[Node] = None

    def set_left_node(
        self,
        left_node: Node,
    ) -> None:
        self.left_node = left_node
        left_node.parent = self

    def set_right_node(
        self,
        right_node: Node,
    ) -> None:
        self.right_node = right_node
        right_node.parent = self
