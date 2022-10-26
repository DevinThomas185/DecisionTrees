import numpy as np
import matplotlib.pyplot as plt

from decision_tree import DecisionTreeLeaf, DecisionTreeNode

# Adjustment from current x position - Left or Right
OFFSET = 100
Y_SPACE = 1


def traverse_tree(root, m_depth, depth=1, x=0, y=50):

    # Plots the nodes
    if isinstance(root, DecisionTreeNode):
        plt.text(x, y, str(getattr(root, 'node_label')), size='smaller', rotation=0,
                 ha="center",
                 va="center",
                 bbox=dict(boxstyle="round",
                           ec=(0., 0., 0.),
                           fc=(1., 1., 1.),
                           )
                 )

        # Sets the correct spacing between nodes

        height = m_depth - depth - 1

        ypos_child = y - Y_SPACE
        left_child = x - (np.power(2, height) * OFFSET)
        right_child = x + (np.power(2, height) * OFFSET)

        x_val = [left_child, x, right_child]
        y_val = [ypos_child, y, ypos_child]
        plt.plot(x_val, y_val)

    # Plots the leaf
    else:
        assert (isinstance(root, DecisionTreeLeaf))
        plt.text(x, y, str(getattr(root, 'classification')), size='smaller', rotation=0,
                 ha="center",
                 va="center",
                 bbox=dict(boxstyle="circle",
                           ec=(0., 0., 0.),
                           fc=(0.3, 1.0, 0.3),
                           )
                 )

    # If on a node, recursively call function and increase depth by 1
    if isinstance(root, DecisionTreeNode):
        traverse_tree(root.left_node, m_depth, depth +
                      1, left_child, ypos_child)
        traverse_tree(root.right_node, m_depth, depth +
                      1, right_child, ypos_child)
        return


def plot_tree(root, max_depth, file):
    plt.figure(figsize=(min(2**5, 2**max_depth), max_depth), dpi=80)
    plt.axis('off')
    traverse_tree(root, max_depth)
    plt.savefig(file)
    plt.close()
    return
