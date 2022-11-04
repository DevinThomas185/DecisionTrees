from decision_tree import Node, DecisionTreeLeaf, DecisionTreeNode
from typing import List
import numpy as np
import evaluation_utils


def prune_tree(
    root: Node,
    current_node: Node,
    validation: np.ndarray,
) -> Node:
    """
    Prune a tree

    :param root: The root of the tree, for evaluation
    :param current_node: The current node that we are looking at to prune
    :param validation: The validation set used to determine if we should prune
    """

    # Using a depth first traversal, we can prune the tree from the bottom up.
    # This means we do not need to prune the tree more than once as we will
    # improve the tree as much as possible by going deep and coming up.

    # For example, with a full tree of depth 3, we may be able to prune the left
    # subtree into a leaf, then the right subtree into a leaf and finally the
    # root node into a leaf, becoming a tree of depth 1

    # No need to prune a leaf - covering the base case
    if isinstance(current_node, DecisionTreeLeaf):
        return

    # Prune the left tree and then the right tree
    assert isinstance(current_node, DecisionTreeNode)
    prune_tree(root, current_node.left_node, validation)
    prune_tree(root, current_node.right_node, validation)

    # See if we can prune this leaf
    # We can only prune if this node has two leaf children
    if isinstance(current_node.left_node, DecisionTreeLeaf) and isinstance(
        current_node.right_node, DecisionTreeLeaf
    ):
        old_classification_error = 1 - evaluation_utils.evaluate(validation, root)

        # We keep a track of whether this current node is the left or right node
        # for its parent. This way, we know whether or not to replace the
        # parent's left or right node with a leaf for the majority class of the
        # current node's two child leaves.
        is_left = current_node.parent.left_node == current_node

        if current_node.left_node.frequency > current_node.right_node.frequency:
            new_leaf_class = current_node.left_node.node_label
            new_leaf_class_index = current_node.left_node.class_index
        else:
            new_leaf_class = current_node.right_node.node_label
            new_leaf_class_index = current_node.right_node.class_index

        new_leaf = DecisionTreeLeaf(
            node_label=new_leaf_class,
            class_index=new_leaf_class_index,
        )
        new_leaf.frequency = (
            current_node.left_node.frequency + current_node.right_node.frequency
        )

        # Set the correct node of the parent to this new leaf, current node
        # retains reference to the parent node for putting it back later
        # if necessary
        if is_left:
            current_node.parent.set_left_node(new_leaf)
        else:
            current_node.parent.set_right_node(new_leaf)

        new_classification_error = 1 - evaluation_utils.evaluate(validation, root)

        # Undo the pruning since the tree now has worse classification error
        if new_classification_error > old_classification_error:
            if is_left:
                current_node.parent.set_left_node(current_node)
            else:
                current_node.parent.set_right_node(current_node)
