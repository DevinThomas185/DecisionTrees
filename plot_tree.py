import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from decision_tree import DecisionTreeNode


def traverse_tree(root, depth=0, x=0, y=50, y_width=5):

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
        ypos_child = y - y_width
        left_child = x - 2*1/np.power(2, depth)
        right_child = x + 2*1/np.power(2, depth)

        # Special spacing conditions to prevent clumping of nodes
        if (depth > 4) and (depth < 6):
            left_child = x - 2*1/np.power(2, depth-2)
            right_child = x + 2*1/np.power(2, depth-2)
        if (depth >= 6):
            left_child = x - 2*1/np.power(2, depth-4)
            right_child = x + 2*1/np.power(2, depth-4)

        x_val = [left_child, x, right_child]
        y_val = [ypos_child, y, ypos_child]
        plt.plot(x_val, y_val)

    # Plots the leaf
    else:
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
        traverse_tree(getattr(root, 'left_node'), depth +
                      1, left_child, ypos_child, y_width)
        traverse_tree(getattr(root, 'right_node'), depth +
                      1, right_child, ypos_child, y_width)
        return


def plot_tree(root, max_depth, file):
    plt.figure(figsize=(2**5, max_depth), dpi=160)
    plt.axis('off')
    traverse_tree(root)
    plt.savefig(file)
    plt.close()
    return
