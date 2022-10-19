import numpy as np
from typing import Callable, Optional

class DecisionTree():
    
    __slots__ = [
        "root",
    ]

    def __init__(self) -> None:
        self.root = None

    

class DecisionTreeNode():
    __slots__ = [
        "function",
        "node_label",
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
        self.left_node: Optional[DecisionTreeNode] = None
        self.right_node: Optional[DecisionTreeNode] = None

    def set_left_node(
        self,
        left_node: DecisionTreeNode,
    ) -> None:
        self.left_node = left_node

    def set_right_node(
        self,
        right_node: DecisionTreeNode,
    ) -> None:
        self.right_node = right_node
