"""
Visualization utilities for the E. coli tracking and analysis pipeline.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle
from matplotlib.collections import PatchCollection
import seaborn as sns
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union


def plot_property_distribution(data: np.ndarray, property_name: str = "",
                             bins: int = 30, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Generate distribution plots for cell properties.
    
    Args:
        data: Array of property values
        property_name: Name of the property (for title)
        bins: Number of histogram bins
        figsize: Figure size
        
    Returns:
        Figure: The matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.hist(data, bins=bins, alpha=0.7)
    
    if property_name:
        ax.set_title(f"Distribution of {property_name}")
        ax.set_xlabel(property_name)
    
    ax.set_ylabel("Frequency")
    
    plt.tight_layout()
    return fig


def create_property_overlay(image: np.ndarray, masks: np.ndarray, 
                          properties: Dict[int, Dict[str, Any]], 
                          property_name: str,
                          cmap: str = "viridis",
                          figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Create property visualizations overlaid on an image.
    
    Args:
        image: Input image
        masks: Segmentation masks
        properties: Dictionary of properties per label
        property_name: Name of the property to visualize
        cmap: Colormap to use
        figsize: Figure size
        
    Returns:
        Figure: The matplotlib figure
    """
    from skimage.color import label2rgb
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the image
    ax.imshow(image, cmap="gray")
    
    # Create colormap for the property
    labels = np.unique(masks)
    labels = labels[labels > 0]  # Exclude background
    
    # Get property values
    values = [properties[label][property_name] for label in labels]
    
    # Normalize values
    vmin, vmax = np.min(values), np.max(values)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    
    # Create new mask with color based on property
    colored_mask = np.zeros_like(masks)
    for i, label in enumerate(labels):
        colored_mask[masks == label] = norm(values[i])
    
    # Overlay
    overlay = label2rgb(masks, image=colored_mask, bg_label=0, alpha=0.7, cmap=cmap)
    ax.imshow(overlay, alpha=0.7)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(property_name)
    
    ax.set_title(f"{property_name} Overlay")
    ax.axis('off')
    
    plt.tight_layout()
    return fig


def create_growth_curve(tracks: Dict[int, Dict[str, Any]], 
                      property_name: str = "area",
                      figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Generate growth curves from tracking data.
    
    Args:
        tracks: Dictionary of tracking data
        property_name: Property to plot (e.g., 'area')
        figsize: Figure size
        
    Returns:
        Figure: The matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for cell_id, track in tracks.items():
        if property_name in track and 'frames' in track:
            frames = track['frames']
            values = track[property_name]
            
            ax.plot(frames, values, marker='o', markersize=4, linewidth=1, alpha=0.7,
                   label=f"Cell {cell_id}")
    
    ax.set_xlabel("Frame")
    ax.set_ylabel(property_name.capitalize())
    ax.set_title(f"{property_name.capitalize()} Over Time")
    
    if len(tracks) < 10:
        ax.legend()
    
    plt.tight_layout()
    return fig


def create_lineage_tree(tracks: Dict[int, Dict[str, Any]], 
                      root_ids: Optional[List[int]] = None,
                      property_name: Optional[str] = None,
                      figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Create lineage trees from tracking data.
    
    Args:
        tracks: Dictionary of tracking data
        root_ids: IDs of root cells (if None, all cells without parents are roots)
        property_name: Property to encode in node color (optional)
        figsize: Figure size
        
    Returns:
        Figure: The matplotlib figure
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("networkx is required for lineage tree plotting")
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add all cells as nodes
    for cell_id, track in tracks.items():
        G.add_node(cell_id)
        
        # Add attributes
        if 'frames' in track:
            G.nodes[cell_id]['start_frame'] = min(track['frames'])
            G.nodes[cell_id]['end_frame'] = max(track['frames'])
        
        if property_name and property_name in track:
            G.nodes[cell_id][property_name] = track[property_name]
    
    # Add edges based on parent-child relationships
    for cell_id, track in tracks.items():
        if 'parent_id' in track and track['parent_id'] in tracks:
            G.add_edge(track['parent_id'], cell_id)
    
    # Find root cells
    if root_ids is None:
        root_ids = [n for n in G.nodes() if G.in_degree(n) == 0]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use hierarchical layout
    pos = {}
    
    # Position each root and its descendants
    for i, root_id in enumerate(root_ids):
        # Get subtree
        subtree = nx.descendants(G, root_id) | {root_id}
        subgraph = G.subgraph(subtree)
        
        # Position with hierarchical layout
        hierarchy_pos = nx.drawing.nx_agraph.graphviz_layout(subgraph, prog='dot')
        
        # Offset horizontally to separate trees
        for node, (x, y) in hierarchy_pos.items():
            pos[node] = (x + i * 500, y)
    
    # Draw the graph
    nx.draw_networkx(G, pos=pos, with_labels=True, node_size=500, 
                    node_color='skyblue', font_size=10, arrows=True,
                    ax=ax)
    
    # Add property-based coloring if requested
    if property_name:
        # Get property values
        values = [G.nodes[n].get(property_name, 0) for n in G.nodes()]
        
        # Map to colors
        cmap = plt.cm.viridis
        nx.draw_networkx_nodes(G, pos=pos, node_color=values, cmap=cmap,
                              node_size=500, ax=ax)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label(property_name)
    
    ax.set_title("Cell Lineage Tree")
    ax.axis('off')
    
    plt.tight_layout()
    return fig


def plot_segmentation_results(image: np.ndarray, masks: np.ndarray, 
                            ground_truth: Optional[np.ndarray] = None,
                            figsize: Tuple[int, int] = (18, 6)) -> plt.Figure:
    """
    Plot segmentation results with optional ground truth comparison.
    
    Args:
        image: Input image
        masks: Predicted segmentation masks
        ground_truth: Ground truth masks (optional)
        figsize: Figure size
        
    Returns:
        Figure: The matplotlib figure
    """
    from skimage.segmentation import mark_boundaries
    
    if ground_truth is not None:
        # Three-panel figure: image, prediction, ground truth
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Plot original image
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Plot predicted masks
        overlay = mark_boundaries(
            np.repeat(image[:, :, np.newaxis], 3, axis=2) if len(image.shape) == 2 else image,
            masks, color=(1, 0, 0)
        )
        axes[1].imshow(overlay)
        axes[1].set_title(f'Predicted Masks ({len(np.unique(masks))-1} cells)')
        axes[1].axis('off')
        
        # Plot ground truth
        gt_overlay = mark_boundaries(
            np.repeat(image[:, :, np.newaxis], 3, axis=2) if len(image.shape) == 2 else image,
            ground_truth, color=(0, 1, 0)
        )
        axes[2].imshow(gt_overlay)
        axes[2].set_title(f'Ground Truth ({len(np.unique(ground_truth))-1} cells)')
        axes[2].axis('off')
    else:
        # Two-panel figure: image and prediction
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot original image
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Plot predicted masks
        overlay = mark_boundaries(
            np.repeat(image[:, :, np.newaxis], 3, axis=2) if len(image.shape) == 2 else image,
            masks, color=(1, 0, 0)
        )
        axes[1].imshow(overlay)
        axes[1].set_title(f'Predicted Masks ({len(np.unique(masks))-1} cells)')
        axes[1].axis('off')
    
    plt.tight_layout()
    return fig 


def create_color_map(n_colors, colormap='tab20'):
    """
    Create a color map with specified number of colors
    
    Parameters:
    -----------
    n_colors : int
        Number of unique colors needed
    colormap : str
        Name of matplotlib colormap to use
        
    Returns:
    --------
    dict
        Dictionary mapping indices to colors
    """
    cmap = plt.cm.get_cmap(colormap, n_colors)
    colors = {}
    
    for i in range(n_colors):
        colors[i] = cmap(i)
    
    return colors


def plot_tracks(tracks, image_shape=None, bg_img=None, figsize=(10, 10), cmap='tab20',
               show_labels=True, alpha=0.7, linewidth=1.5, markersize=5, title=None):
    """
    Plot cell tracks as trajectories on an image
    
    Parameters:
    -----------
    tracks : pd.DataFrame
        DataFrame containing track information (track_id, frame, x, y)
    image_shape : tuple
        Shape of the image (height, width)
    bg_img : numpy.ndarray
        Background image to overlay tracks on
    figsize : tuple
        Figure size
    cmap : str
        Matplotlib colormap for track coloring
    show_labels : bool
        Whether to show track IDs
    alpha : float
        Transparency of tracks
    linewidth : float
        Line width for track trajectories
    markersize : float
        Size of track position markers
    title : str
        Title for the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure with track visualizations
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Show background image if provided
    if bg_img is not None:
        if len(bg_img.shape) == 2:  # Grayscale image
            ax.imshow(bg_img, cmap='gray')
        else:  # RGB image
            ax.imshow(bg_img)
    
    # Get unique track IDs
    unique_tracks = tracks['track_id'].unique()
    n_tracks = len(unique_tracks)
    
    # Create color map
    color_map = create_color_map(n_tracks, colormap=cmap)
    
    # Plot each track
    for i, track_id in enumerate(unique_tracks):
        track_data = tracks[tracks['track_id'] == track_id].sort_values('frame')
        
        # Skip if only one point
        if len(track_data) <= 1:
            continue
        
        # Get track color
        color = color_map[i % n_tracks]
        
        # Plot trajectory
        ax.plot(track_data['x'], track_data['y'], 
                linestyle='-', linewidth=linewidth, 
                color=color, alpha=alpha)
        
        # Plot points
        ax.scatter(track_data['x'], track_data['y'], 
                  s=markersize, color=color, alpha=alpha)
        
        # Add track ID at the first position
        if show_labels:
            first_point = track_data.iloc[0]
            ax.text(first_point['x'], first_point['y'], 
                    str(int(track_id)), color=color, fontweight='bold')
    
    # Set plot limits if image shape is provided
    if image_shape is not None:
        ax.set_xlim(0, image_shape[1])
        ax.set_ylim(image_shape[0], 0)  # Flip y-axis to match image coordinates
    
    # Set title
    if title is None:
        title = f'Cell Tracks (n={n_tracks})'
    ax.set_title(title)
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    return fig


def plot_track_stats(tracks, metric='displacement', figsize=(12, 6), bins=30):
    """
    Plot statistics about the tracks
    
    Parameters:
    -----------
    tracks : pd.DataFrame
        DataFrame containing track information (track_id, frame, x, y)
    metric : str
        Metric to plot ('displacement', 'lifespan', 'speed', or 'length')
    figsize : tuple
        Figure size
    bins : int
        Number of bins for histogram
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure with statistical plot
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    track_stats = []
    unique_tracks = tracks['track_id'].unique()
    
    for track_id in unique_tracks:
        track_data = tracks[tracks['track_id'] == track_id].sort_values('frame')
        
        # Skip if only one point
        if len(track_data) <= 1:
            continue
        
        # Calculate track lifespan (frames)
        start_frame = track_data['frame'].min()
        end_frame = track_data['frame'].max()
        lifespan = end_frame - start_frame + 1
        
        # Calculate displacement (start to end distance)
        start_pos = track_data.iloc[0][['x', 'y']].values
        end_pos = track_data.iloc[-1][['x', 'y']].values
        displacement = np.linalg.norm(end_pos - start_pos)
        
        # Calculate track length (sum of step distances)
        length = 0
        for i in range(1, len(track_data)):
            pos1 = track_data.iloc[i-1][['x', 'y']].values
            pos2 = track_data.iloc[i][['x', 'y']].values
            step = np.linalg.norm(pos2 - pos1)
            length += step
        
        # Calculate average speed
        speed = length / lifespan if lifespan > 0 else 0
        
        track_stats.append({
            'track_id': track_id,
            'lifespan': lifespan,
            'displacement': displacement,
            'length': length,
            'speed': speed
        })
    
    if not track_stats:
        ax.text(0.5, 0.5, "No valid tracks to analyze", 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
        return fig
    
    stats_df = pd.DataFrame(track_stats)
    
    if metric == 'displacement':
        sns.histplot(stats_df['displacement'], bins=bins, ax=ax)
        ax.set_xlabel('Displacement (pixels)')
        ax.set_title('Distribution of Cell Displacements')
    elif metric == 'lifespan':
        sns.histplot(stats_df['lifespan'], bins=bins, ax=ax)
        ax.set_xlabel('Lifespan (frames)')
        ax.set_title('Distribution of Track Lifespans')
    elif metric == 'speed':
        sns.histplot(stats_df['speed'], bins=bins, ax=ax)
        ax.set_xlabel('Speed (pixels/frame)')
        ax.set_title('Distribution of Cell Speeds')
    elif metric == 'length':
        sns.histplot(stats_df['length'], bins=bins, ax=ax)
        ax.set_xlabel('Track Length (pixels)')
        ax.set_title('Distribution of Total Track Lengths')
    else:
        ax.text(0.5, 0.5, f"Unknown metric: {metric}", 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
    
    ax.set_ylabel('Count')
    plt.tight_layout()
    
    return fig


def visualize_frame_with_tracks(image, masks=None, tracks=None, frame_id=0, 
                               ax=None, figsize=(10, 8), cmap='tab20', alpha=0.7,
                               show_ids=True, mask_alpha=0.3):
    """
    Visualize a single frame with cell masks and tracks
    
    Parameters:
    -----------
    image : numpy.ndarray
        Image to visualize
    masks : numpy.ndarray
        Segmentation masks (instance segmentation)
    tracks : pd.DataFrame
        DataFrame containing track information
    frame_id : int
        Frame ID to visualize
    ax : matplotlib.axes.Axes
        Axes to plot on
    figsize : tuple
        Figure size
    cmap : str
        Colormap for track visualization
    alpha : float
        Transparency of track markers
    show_ids : bool
        Whether to show track IDs
    mask_alpha : float
        Transparency of segmentation masks
        
    Returns:
    --------
    matplotlib.axes.Axes
        Axes with visualization
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Show the background image
    if image is not None:
        if len(image.shape) == 2:  # Grayscale image
            ax.imshow(image, cmap='gray')
        else:  # RGB image
            ax.imshow(image)
    
    # Show masks if available
    if masks is not None:
        # Create a color overlay for the masks
        mask_overlay = np.zeros((*masks.shape, 4))
        
        unique_labels = np.unique(masks)
        unique_labels = unique_labels[unique_labels > 0]  # Skip background
        
        # Random colors for masks
        for label in unique_labels:
            mask = masks == label
            color = np.array([random.random(), random.random(), random.random(), mask_alpha])
            mask_overlay[mask] = color
        
        ax.imshow(mask_overlay)
    
    # Show tracks if available
    if tracks is not None:
        # Filter tracks for the current frame
        current_tracks = tracks[tracks['frame'] == frame_id]
        
        if len(current_tracks) > 0:
            # Create a colormap for the tracks
            unique_track_ids = tracks['track_id'].unique()
            n_tracks = len(unique_track_ids)
            track_cmap = create_color_map(n_tracks, colormap=cmap)
            
            # Create mapping from track_id to color index
            id_to_idx = {tid: i for i, tid in enumerate(unique_track_ids)}
            
            # Plot each track
            for _, track in current_tracks.iterrows():
                track_id = track['track_id']
                color_idx = id_to_idx[track_id] % n_tracks
                color = track_cmap[color_idx]
                
                # Plot marker at cell position
                ax.scatter(track['x'], track['y'], s=40, color=color, alpha=alpha)
                
                # Show track ID
                if show_ids:
                    ax.text(track['x'] + 5, track['y'] + 5, str(int(track_id)), 
                           color=color, fontweight='bold')
    
    ax.set_title(f'Frame {frame_id}')
    ax.axis('off')
    
    return ax


def create_tracking_visualization(images, masks=None, tracks=None, fps=5, 
                                 figsize=(10, 8), cmap='tab20', show_ids=True,
                                 mask_alpha=0.3, save_path=None):
    """
    Create an animation of tracking results
    
    Parameters:
    -----------
    images : list or numpy.ndarray
        List of images or 3D array (frames, height, width)
    masks : list or numpy.ndarray
        List of segmentation masks or 3D array
    tracks : pd.DataFrame
        DataFrame containing track information
    fps : int
        Frames per second for the animation
    figsize : tuple
        Figure size
    cmap : str
        Colormap for track visualization
    show_ids : bool
        Whether to show track IDs
    mask_alpha : float
        Transparency of segmentation masks
    save_path : str
        Path to save the animation (optional)
        
    Returns:
    --------
    matplotlib.animation.Animation
        Animation object
    """
    # Setup figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert to lists if arrays
    if isinstance(images, np.ndarray) and images.ndim == 3:
        images = [images[i] for i in range(images.shape[0])]
    
    if masks is not None and isinstance(masks, np.ndarray) and masks.ndim == 3:
        masks = [masks[i] for i in range(masks.shape[0])]
    
    # Determine number of frames
    n_frames = len(images)
    
    # Setup animation function
    def animate(i):
        ax.clear()
        frame_masks = masks[i] if masks is not None else None
        visualize_frame_with_tracks(images[i], frame_masks, tracks, i, 
                                   ax=ax, cmap=cmap, show_ids=show_ids,
                                   mask_alpha=mask_alpha)
        return ax,
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=n_frames,
                                 interval=1000/fps, blit=True)
    
    # Save if path provided
    if save_path:
        writer = animation.FFMpegWriter(fps=fps)
        anim.save(save_path, writer=writer)
    
    plt.close()
    return anim


def plot_tracking_comparison(gt_tracks, pred_tracks, image_shape=None, 
                            figsize=(15, 7), title=None):
    """
    Plot ground truth tracks versus predicted tracks for comparison
    
    Parameters:
    -----------
    gt_tracks : pd.DataFrame
        Ground truth tracking data
    pred_tracks : pd.DataFrame
        Predicted tracking data
    image_shape : tuple
        Shape of the image (height, width)
    figsize : tuple
        Figure size
    title : str
        Title for the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure with track comparisons
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot ground truth tracks
    n_gt_tracks = len(gt_tracks['track_id'].unique())
    gt_color_map = create_color_map(n_gt_tracks, colormap='viridis')
    
    for i, track_id in enumerate(gt_tracks['track_id'].unique()):
        track_data = gt_tracks[gt_tracks['track_id'] == track_id].sort_values('frame')
        
        if len(track_data) <= 1:
            continue
        
        color = gt_color_map[i % n_gt_tracks]
        
        ax1.plot(track_data['x'], track_data['y'], '-', linewidth=1.5, color=color, alpha=0.7)
        ax1.scatter(track_data['x'], track_data['y'], s=5, color=color, alpha=0.7)
    
    # Plot predicted tracks
    n_pred_tracks = len(pred_tracks['track_id'].unique())
    pred_color_map = create_color_map(n_pred_tracks, colormap='viridis')
    
    for i, track_id in enumerate(pred_tracks['track_id'].unique()):
        track_data = pred_tracks[pred_tracks['track_id'] == track_id].sort_values('frame')
        
        if len(track_data) <= 1:
            continue
        
        color = pred_color_map[i % n_pred_tracks]
        
        ax2.plot(track_data['x'], track_data['y'], '-', linewidth=1.5, color=color, alpha=0.7)
        ax2.scatter(track_data['x'], track_data['y'], s=5, color=color, alpha=0.7)
    
    # Set plot limits if image shape is provided
    if image_shape is not None:
        ax1.set_xlim(0, image_shape[1])
        ax1.set_ylim(image_shape[0], 0)
        ax2.set_xlim(0, image_shape[1])
        ax2.set_ylim(image_shape[0], 0)
    
    # Set titles
    ax1.set_title(f'Ground Truth Tracks (n={n_gt_tracks})')
    ax2.set_title(f'Predicted Tracks (n={n_pred_tracks})')
    
    # Remove axis ticks
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    if title:
        fig.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    return fig


def plot_lineage_tree(tracks, root_id=None, figsize=(12, 8), 
                     node_size=100, colormap='tab20', title=None):
    """
    Plot a cell lineage tree showing cell divisions
    
    Parameters:
    -----------
    tracks : pd.DataFrame
        DataFrame containing track information with parent_id column
    root_id : int
        ID of the root track to start the tree (optional)
    figsize : tuple
        Figure size
    node_size : int
        Size of nodes in the tree
    colormap : str
        Colormap for node coloring
    title : str
        Title for the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure with lineage tree
    """
    # Check if required column exists
    if 'parent_id' not in tracks.columns:
        raise ValueError("Tracks DataFrame must contain 'parent_id' column for lineage tree visualization")
    
    # Import networkx here to avoid dependency for other functions
    try:
        import networkx as nx
    except ImportError:
        print("NetworkX is required for lineage tree visualization")
        print("Install with: pip install networkx")
        return None
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Get unique tracks and their parent relationships
    unique_tracks = tracks['track_id'].unique()
    
    # Add nodes
    for track_id in unique_tracks:
        # Get track data
        track_data = tracks[tracks['track_id'] == track_id]
        
        # Get track lifespan
        start_frame = track_data['frame'].min()
        end_frame = track_data['frame'].max()
        lifespan = end_frame - start_frame + 1
        
        # Add node with attributes
        G.add_node(track_id, lifespan=lifespan, start_frame=start_frame, end_frame=end_frame)
    
    # Add edges based on parent relationships
    for track_id in unique_tracks:
        track_data = tracks[tracks['track_id'] == track_id]
        parent_id = track_data['parent_id'].iloc[0]
        
        if parent_id > 0 and parent_id in unique_tracks:
            G.add_edge(parent_id, track_id)
    
    # If root_id not specified, find roots (nodes without parents)
    if root_id is None:
        roots = [n for n, d in G.in_degree() if d == 0]
        
        if not roots:
            # No clear roots, use node with lowest ID
            root_id = min(unique_tracks)
        elif len(roots) == 1:
            root_id = roots[0]
        else:
            # Multiple roots, use the one with the earliest start frame
            root_id = min(roots, key=lambda r: G.nodes[r]['start_frame'])
    
    # Create subgraph starting from root_id
    descendants = nx.descendants(G, root_id)
    descendants.add(root_id)
    G = G.subgraph(descendants)
    
    # Setup figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Layout for tree visualization (top-to-bottom)
    pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot')
    
    # Get node color based on track start time
    start_frames = nx.get_node_attributes(G, 'start_frame')
    
    if start_frames:
        # Normalize start frames for colormap
        min_frame = min(start_frames.values())
        max_frame = max(start_frames.values())
        frame_range = max_frame - min_frame
        
        if frame_range > 0:
            node_colors = [(start_frames[n] - min_frame) / frame_range for n in G.nodes()]
        else:
            node_colors = [0] * len(G.nodes())
    else:
        node_colors = list(range(len(G.nodes())))
    
    # Draw nodes and edges
    cmap = plt.cm.get_cmap(colormap)
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_colors, cmap=cmap, alpha=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=15, ax=ax)
    
    # Add labels
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    
    # Add colorbar
    if start_frames:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(min_frame, max_frame))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Frame')
    
    # Set title
    if title is None:
        title = f'Cell Lineage Tree (root: {root_id})'
    plt.title(title)
    
    # Remove axes
    ax.set_axis_off()
    
    plt.tight_layout()
    return fig 