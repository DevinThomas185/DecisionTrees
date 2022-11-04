from __future__ import annotations
import numpy as np
from typing import Callable, Optional


class Node:
    __slots__ = [
        "node_label",
        "parent",
    ]

    def __init__(
        self,
        node_label: str,
    ) -> None:
        """
        Basic node class, retaining the node label and parent for this node

        :param node_label: The label for this node, either a class or function
        """
        self.node_label = node_label


class DecisionTreeLeaf(Node):
    __slots__ = [
        "frequency",
        "class_index",
    ]

    def __init__(
        self,
        node_label: str,
        class_index: int,
    ) -> None:
        """
        The class for a leaf node

        :param node_label: The label for this node, a class
        :param class_index: Index in unique_classes that this label is at,
                            needed for evaluation without translating to the
                            class string to see if a prediction is correct
        """
        super().__init__(node_label)
        self.class_index = class_index
        self.frequency = 0


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
        A regular node with two subtree children

        :param function: The function to evaluate which way to go when this
                         node is reached
        :param node_label: The label for this node, a function string
        """
        super().__init__(node_label)
        self.function = function
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
