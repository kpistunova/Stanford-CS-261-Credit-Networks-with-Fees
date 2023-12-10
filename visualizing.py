import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from matplotlib.colors import Normalize
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
from datetime import datetime
import os
current_date = datetime.now().strftime('%Y-%m-%d')
base_directory = 'data'
new_directory_path = os.path.join(base_directory, current_date)
# Check if the directory does not exist
if not os.path.exists(new_directory_path):
    # If it doesn't exist, create a new directory
    os.makedirs(new_directory_path)
def my_draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=None,
    label_pos=0.5,
    font_size=10,
    font_color="k",
    font_family="sans-serif",
    font_weight="normal",
    alpha=None,
    bbox=None,
    horizontalalignment="center",
    verticalalignment="center",
    ax=None,
    rotate=True,
    clip_on=True,
    rad=0
):
    """Draw edge labels.

    Parameters
    ----------
    G : graph
        A networkx graph

    pos : dictionary
        A dictionary with nodes as keys and positions as values.
        Positions should be sequences of length 2.

    edge_labels : dictionary (default={})
        Edge labels in a dictionary of labels keyed by edge two-tuple.
        Only labels for the keys in the dictionary are drawn.

    label_pos : float (default=0.5)
        Position of edge label along edge (0=head, 0.5=center, 1=tail)

    font_size : int (default=10)
        Font size for text labels

    font_color : string (default='k' black)
        Font color string

    font_weight : string (default='normal')
        Font weight

    font_family : string (default='sans-serif')
        Font family

    alpha : float or None (default=None)
        The text transparency

    bbox : Matplotlib bbox, optional
        Specify text box properties (e.g. shape, color etc.) for edge labels.
        Default is {boxstyle='round', ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0)}.

    horizontalalignment : string (default='center')
        Horizontal alignment {'center', 'right', 'left'}

    verticalalignment : string (default='center')
        Vertical alignment {'center', 'top', 'bottom', 'baseline', 'center_baseline'}

    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.

    rotate : bool (deafult=True)
        Rotate edge labels to lie parallel to edges

    clip_on : bool (default=True)
        Turn on clipping of edge labels at axis boundaries

    Returns
    -------
    dict
        `dict` of labels keyed by edge

    Examples
    --------

    Also see the NetworkX drawing examples at
    https://networkx.org/documentation/latest/auto_examples/index.html

    See Also
    --------
    draw
    draw_networkx
    draw_networkx_nodes
    draw_networkx_edges
    draw_networkx_labels
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        ax = plt.gca()
    if edge_labels is None:
        labels = {(u, v): d for u, v, d in G.edges(data=True)}
    else:
        labels = edge_labels
    text_items = {}
    for (n1, n2), label in labels.items():
        (x1, y1) = pos[n1]
        (x2, y2) = pos[n2]
        (x, y) = (
            x1 * label_pos + x2 * (1.0 - label_pos),
            y1 * label_pos + y2 * (1.0 - label_pos),
        )
        pos_1 = ax.transData.transform(np.array(pos[n1]))
        pos_2 = ax.transData.transform(np.array(pos[n2]))
        linear_mid = 0.5*pos_1 + 0.5*pos_2
        d_pos = pos_2 - pos_1
        rotation_matrix = np.array([(0,1), (-1,0)])
        ctrl_1 = linear_mid + rad*rotation_matrix@d_pos
        ctrl_mid_1 = 0.5*pos_1 + 0.5*ctrl_1
        ctrl_mid_2 = 0.5*pos_2 + 0.5*ctrl_1
        bezier_mid = 0.5*ctrl_mid_1 + 0.5*ctrl_mid_2
        (x, y) = ax.transData.inverted().transform(bezier_mid)

        if rotate:
            # in degrees
            angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360
            # make label orientation "right-side-up"
            if angle > 90:
                angle -= 180
            if angle < -90:
                angle += 180
            # transform data coordinate angle to screen coordinate angle
            xy = np.array((x, y))
            trans_angle = ax.transData.transform_angles(
                np.array((angle,)), xy.reshape((1, 2))
            )[0]
        else:
            trans_angle = 0.0
        # use default box of white with white border
        if bbox is None:
            bbox = dict(boxstyle="round", ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0))
        if not isinstance(label, str):
            label = str(label)  # this makes "1" and 1 labeled the same

        t = ax.text(
            x,
            y,
            label,
            size=font_size,
            color=font_color,
            family=font_family,
            weight=font_weight,
            alpha=alpha,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            rotation=trans_angle,
            transform=ax.transData,
            bbox=bbox,
            zorder=1,
            clip_on=clip_on,
        )
        text_items[(n1, n2)] = t

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    return text_items

def visualize_graph(G, transaction_number, succesfull_transactions, fee, capacity, pos=None,  s=None, t=None, fail = False, no_path = False, anot = True, save = False, show = False, G_reference = None, type = None, selected_states = None, state_probabilities = None):
    if pos is None:
        pos = nx.spring_layout(G)
    M = G.number_of_edges()
    if type is None:
        type = ''
    if G_reference is None:
        G_reference = G.copy()
    if type == 'star':
        if M > 50:
            figsize = (8*1.3, 6*1.3)
            anot = False
            ss=8
        elif M > 10:
            figsize = (8, 6)
            anot = False
            ss = 10
        else:
            figsize = (8, 6)
            ss=10
    else:
        if M >= 20:
            figsize = (8*2.5, 6*2.5)
            anot = False
            ss=8
        elif M >= 9:
            figsize = (8*1.3, 6*1.3)
            ss=10
        elif M >= 7:
            figsize = (8, 6)
            ss=8
        else:
            figsize = (8, 6)
            ss=10
    node = G.number_of_nodes()
    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightskyblue', edgecolors='black')

    if s is not None and t is not None:
        # Draw the source and target nodes in different colors
        nx.draw_networkx_nodes(G, pos, nodelist=[s], ax=ax, node_color='palegreen', edgecolors='black')
        nx.draw_networkx_nodes(G, pos, nodelist=[t], ax=ax, node_color='lightcoral', edgecolors='black')


    def get_edge_colors(G, G_reference, capacity):
        norm = mcolors.Normalize(vmin=-capacity, vmax=capacity)
        cmap_forward = plt.cm.coolwarm
        cmap_reverse = plt.cm.coolwarm_r
        edge_colors = {}

        for u, v in G.edges():
            weight = G[u][v]['capacity']
            is_reverse = G_reference.has_edge(v, u) and not G_reference.has_edge(u, v)
            cmap = cmap_reverse if is_reverse else cmap_forward
            color = cmap(norm(weight))
            edge_colors[(u, v)] = color

        return edge_colors

    nx.draw_networkx_labels(G, pos, ax=ax)
    # Get edge colors
    edge_colors = get_edge_colors(G, G_reference, capacity)

    # Determine edge types
    straight_edges = [edge for edge in G.edges() if not G.has_edge(edge[1], edge[0])]
    curved_edges = [edge for edge in G.edges() if G.has_edge(edge[1], edge[0])]

    # Draw straight edges
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=straight_edges, edge_color=[edge_colors[e] for e in straight_edges])

    # Draw curved edges
    arc_rad = 0.25
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=curved_edges, connectionstyle=f'arc3, rad = {arc_rad}',
                           edge_color=[edge_colors[e] for e in curved_edges])
    # Create a color map scalar mappable object for the color bar
    norm = mcolors.Normalize(vmin=-capacity, vmax=capacity)
    cmap = plt.cm.coolwarm
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # You can also pass in your range of edge weights here

    # Add the color bar to the figure
    cbar = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.04)  # Make colorbar narrower with a smaller fraction
    # cbar.set_label('Edge Capacity', labelpad=10, y=1.12,
    #                rotation=0)  # Adjust labelpad and y for better placement of "Edge Capacity"
    cbar.set_ticks([])  # Remove the ticks
    cbar.set_ticks([0])  # Add ticks at min, zero, and max
    cbar.ax.set_yticklabels(['0'])  # Label the min, zero, and max ticks

    # Set custom labels for the min and max of the colorbar
    cbar.ax.text(0.5, -0.01, f'Reverse {capacity}', transform=cbar.ax.transAxes, ha='center', va='top')
    cbar.ax.text(0.5, 1.01, f'Forward {capacity}', transform=cbar.ax.transAxes, ha='center', va='bottom')
    cbar.ax.set_title('Capacity', loc='center', pad=17, fontsize = 13)

    if anot:
        bbox_props = dict(boxstyle='round,pad=0.1', ec='white', fc='white', alpha=0.7)

        edge_weights = nx.get_edge_attributes(G, 'capacity')
        curved_edge_labels = {edge: edge_weights[edge] for edge in curved_edges}
        straight_edge_labels = {edge: edge_weights[edge] for edge in straight_edges}
        my_draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=curved_edge_labels, rotate=False, rad=arc_rad, font_size = ss, bbox=bbox_props)
        nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=straight_edge_labels, rotate=False, font_size = ss,  bbox=bbox_props)

    if transaction_number == 0:
        title = f'Initial {type} graph, f = {fee}, c = {capacity}, n = {node}'
    elif fail:
        title = f'{type} graph after {transaction_number} transactions, {transaction_number - succesfull_transactions} failed, f = {fee}, c = {capacity}, n = {node}'
    elif no_path:
        title = f'{type} graph after {transaction_number} transactions, no path, {transaction_number - succesfull_transactions} failed, f = {fee}, c = {capacity}, n = {node}'
    else:
        title = f'Final {type} graph, success prob = {round(succesfull_transactions/transaction_number*100, 2)}%, f = {fee}, c = {capacity}, n = {node}'

    ax.set_title(title, fontsize=14)
    filename = title.replace(" ", "_").replace(",", "").replace("=","") + ".png"  # Replace spaces with underscores and remove commas and equals signs for filenam
    plt.axis('off')
    file_path = os.path.join(new_directory_path, filename)
    plt.tight_layout()
    if show:
        plt.show()
    if save:
        fig.savefig((file_path), dpi=300)
    plt.close(fig)
    if selected_states is not None and state_probabilities is not None:
        num_states = len(selected_states) if selected_states else 1

        figsize = (num_states * 8/1.8, 6/1.8) if num_states <= 4 else (32, 6 * (num_states // 4))

        fig, axes = plt.subplots(nrows=1 if num_states <= 4 else num_states // 4, ncols=min(num_states, 4),
                                 figsize=figsize, dpi=300)
        if len(selected_states) == 1:  # Adjust if there's only one state to plot
            axes = [axes]

        for ax, state in zip(axes.flatten(), selected_states):
            H = nx.DiGraph()

            for (u, v, capacity) in state:
                H.add_edge(u, v, capacity=capacity)
            nx.draw_networkx_labels(H, pos, ax=ax)

            # Create a graph for the current state
            nx.draw_networkx_nodes(H, pos, ax=ax, node_color='lightskyblue', edgecolors='black')
            # Get edge colors

            # Determine edge types
            straight_edges = [edge for edge in H.edges() if not H.has_edge(edge[1], edge[0])]
            curved_edges = [edge for edge in H.edges() if H.has_edge(edge[1], edge[0])]

            # Draw straight edges
            edge_colors = get_edge_colors(H, H, capacity)

            nx.draw_networkx_edges(H, pos, ax=ax, edgelist=straight_edges,
                                   edge_color=[edge_colors[e] for e in straight_edges])

            # Draw edge labels (capacities)
            # Draw curved edges
            # nx.draw(G, pos, with_labels=True, node_size=700, node_color='skyblue')

            arc_rad = 0.25
            nx.draw_networkx_edges(H, pos, ax=ax, edgelist=curved_edges, connectionstyle=f'arc3, rad = {arc_rad}',
                                   edge_color=[edge_colors[e] for e in curved_edges])
            # Create a color map scalar mappable object for the color bar

            # edge_labels = nx.get_edge_attributes(H, 'capacity')
            # nx.draw_networkx_edge_labels(H, pos, edge_labels=edge_labels, ax=ax)
            bbox_props = dict(boxstyle='round,pad=0.1', ec='white', fc='white', alpha=0.7)

            edge_weights = nx.get_edge_attributes(H, 'capacity')
            curved_edge_labels = {edge: edge_weights[edge] for edge in curved_edges}
            straight_edge_labels = {edge: edge_weights[edge] for edge in straight_edges}
            my_draw_networkx_edge_labels(H, pos, ax=ax, edge_labels=curved_edge_labels, rotate=False, rad=arc_rad,
                                         font_size=ss, bbox=bbox_props)
            nx.draw_networkx_edge_labels(H, pos, ax=ax, edge_labels=straight_edge_labels, rotate=False, font_size=ss,
                                         bbox=bbox_props)

            # Title with state info and probability
            probability = state_probabilities[state]
            state_label = ', '.join([f'({u},{v},{c})' for u, v, c in state])
            ax.set_title(f"State: {state_label}\nProbability: {probability:.4f}")
            ax.axis('off')
    # ax.set_title(title, fontsize=14)
    plt.axis('off')
    new_f = 'MK' + filename +".png"
    file_path = os.path.join(new_directory_path, new_f)
    plt.tight_layout()
    if show:
        plt.show()
    if save:
        fig.savefig((file_path), dpi=300)
    plt.close(fig)

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, node_radius=0.0027, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs
        self.node_radius = node_radius  # Radius of the node sphere

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        if len(xs) == 1:
            # Return an early depth value (e.g., the z-coordinate)
            return zs[0]
        # Calculate the direction vector from start to end point
        direction = np.array([xs[1] - xs[0], ys[1] - ys[0], zs[1] - zs[0]])
        length = np.linalg.norm(direction)

        direction /= length

        # Subtract the node radius from the arrow's length
        # to get the new endpoint that stops at the node's surface
        xs[1] -= direction[0] * self.node_radius
        ys[1] -= direction[1] * self.node_radius
        zs[1] -= direction[2] * self.node_radius

        # arrow_start_point = end_point - direction * node_radius * 32
        xs[0] = xs[1] - direction[0] * (self.node_radius)*4
        ys[0] = ys[1] - direction[1] * (self.node_radius )*4
        zs[0] = zs[1] - direction[2] * (self.node_radius )*4

        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


# Function to draw an arrow with varying opacity based on its length
# Function to draw an arrow with varying opacity based on its length
def draw_gradient_arrow(ax, start_point, end_point, node_radius, cmap, minz, maxz, lww):
    # Calculate the direction vector from start to end point
    direction = np.array(end_point) - np.array(start_point)
    length = np.linalg.norm(direction)

    if length > 0:  # Only perform the division if the length is non-zero
        direction /= length

        # Subtract the node radius from the arrow's length to get the new endpoint
        # that stops at the node's surface, and add it to the start point
        # end_point -= direction * node_radius
        # start_point += direction * node_radius

    num_points = 100
    ee = end_point - direction * node_radius*30
    ss = start_point + direction * node_radius*20
    x = np.linspace(ss[0], ee[0], num_points)
    y = np.linspace(ss[1], ee[1], num_points)
    z = np.linspace(ss[2], ee[2], num_points)

    # Normalize the colors along the line based on the z-coordinates
    norm = Normalize(vmin=minz, vmax=maxz)
    colors = [cmap(norm(zi)) for zi in z]
    segments = [np.array([[x[i], y[i], z[i]], [x[i+1], y[i+1], z[i+1]]]) for i in range(num_points - 1)]
    lc = Line3DCollection(segments, colors=colors, linewidth=lww, alpha = 0.9)
    ax.add_collection3d(lc)
    lc.set_zorder(1)

#----------
def plot_3d(G, ccc = False):
    sns.set_style()
    if G.number_of_nodes() > 16:
        lww = 1
    elif G.number_of_nodes() > 8:
        lww =2
    else:
        lww = 3

    if G.number_of_nodes() >=9:
        figsize = (12*1.5, 9*1.5)
    else:
        figsize = (12, 9)
    pos_3dd = nx.spring_layout(G, dim=3, seed=779)
    node_0_pos = pos_3dd[0]
    translation_vector = -np.array(node_0_pos)

    # Translate all nodes
    pos_3d = {node: pos + translation_vector for node, pos in pos_3dd.items()}

    # Now let's visualize the graph in 3D
    fig, ax = plt.subplots(figsize=figsize, dpi=300 ,  constrained_layout=True, subplot_kw={'projection': '3d'},
                           gridspec_kw=dict(top=1.07, bottom=0.02, left=0, right=1))
    fig.patch.set_facecolor('white')  # Set the figure background to white
    ax.set_facecolor('white')  # Set the axes background to white

    cmap = plt.get_cmap('coolwarm')
    # Get the range of z-coordinates
    z_values = [pos[2] for pos in pos_3d.values()]
    edge_lengths = [np.linalg.norm(np.array(pos_3d[v]) - np.array(pos_3d[u])) for u, v in G.edges()]

    min_z, max_z = min(z_values), max(z_values)

    # Normalize z-coordinates for colormap
    norm = plt.Normalize(min_z, max_z)

    minz = min(pos_3d.values(), key=lambda x: x[2])[2]  # min z-value from your nodes
    maxz = max(pos_3d.values(), key=lambda x: x[2])[2]
    # Draw edges first
    for u, v in G.edges():
        draw_gradient_arrow(ax, pos_3d[u], pos_3d[v], node_radius=0.0027, cmap=cmap, minz=minz, maxz=maxz, lww=lww)

    # Then draw arrows
    for u, v in G.edges():
        cc = cmap(norm(pos_3d[v][2]))
        x = np.array((pos_3d[u][0], pos_3d[v][0]))
        y = np.array((pos_3d[u][1], pos_3d[v][1]))
        z = np.array((pos_3d[u][2], pos_3d[v][2]))
        arrow = Arrow3D(x, y, z, mutation_scale=30, lw=lww, arrowstyle="-|>", color=cc)
        ax.add_artist(arrow)
        arrow.set_zorder(100000)  # Attempt to put arrows on top
    ax.auto_scale_xyz([pos_3d[v][0] for v in sorted(G)], [pos_3d[v][1] for v in sorted(G)], [pos_3d[v][2] for v in sorted(G)])

    # Add edge labels - adjust font size and color to match your 2D graph
    if ccc:
        for (u, v), label in nx.get_edge_attributes(G, 'capacity').items():
            x_mid = (pos_3d[u][0] + pos_3d[v][0]) / 2
            y_mid = (pos_3d[u][1] + pos_3d[v][1]) / 2
            z_mid = (pos_3d[u][2] + pos_3d[v][2]) / 2
            ax.text(x_mid, y_mid, z_mid, str(label), size=16, color='black', ha='center', va='center',
                    bbox=dict(facecolor=(0.95, 0.95, 0.95), edgecolor='none', pad=1), zorder = 6000)
    # Plot the nodes with color based on z-coordinate
    for node, (x, y, z) in pos_3d.items():
        if node == 0:
            zzz = 200000
        else:
            zzz = 200000
        color = cmap(norm(z))
        ax.scatter(x, y, z, s=400, color=color, edgecolors="black", depthshade=True, alpha=0.9, zorder = zzz-1)
        ax.text(x, y, z, str(node), color='black', size=16, ha='center', va='center', zorder=zzz)


    # ax.view_init(elev=20, azim=-60)  # You can adjust these angles
    # If the graph still appears small, try adjusting the axis limits or the viewpoint
    ax.set_xlim([min(x[0] for x in pos_3d.values()), max(x[0] for x in pos_3d.values())])
    ax.set_ylim([min(x[1] for x in pos_3d.values()), max(x[1] for x in pos_3d.values())])
    ax.set_zlim([min(x[2] for x in pos_3d.values()), max(x[2] for x in pos_3d.values())])
    ax.view_init(elev=20, azim=-60)

    ax.set_box_aspect([1,1,1])  # Equal aspect ratio

    # Remove gridlines and axis ticks, and set labels
    # ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # Adjust the viewpoint to enhance depth perception
    # ax.view_init(elev=30, azim=120)

    # Plot gridlines
    ax.xaxis._axinfo['grid'].update(color = 'dimgray', linestyle = '--', linewidth = 0.5)
    ax.yaxis._axinfo['grid'].update(color = 'dimgray', linestyle = '--', linewidth = 0.5)
    ax.zaxis._axinfo['grid'].update(color = 'dimgray', linestyle = '--', linewidth = 0.5)

    # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    # plt.draw()

    # Display the plot
    plt.show()
