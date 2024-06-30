import itertools
import numpy as np
import random
import math
import pandas as pd
import os 
import seaborn as sns
import torch

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import colorcet as cc

import umap.umap_ as umap
import networkx as nx

from hmmlearn import hmm
from sklearn.model_selection import KFold
from scipy.cluster.hierarchy import linkage, fcluster

from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

from model import HMM_ensemble
from concurrent.futures import ThreadPoolExecutor, as_completed

from scipy.sparse.csgraph import connected_components

import copy

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import davies_bouldin_score    
from sklearn.metrics import accuracy_score

def simulate_head(num_neuron, back_firing, peak_firing, tuning_width, data_length):

    hd_sim = np.random.uniform(0, 2 * math.pi, data_length)
    rf_hd = np.random.uniform(0, 2 * math.pi, num_neuron)
    rate_hd = np.zeros((num_neuron, data_length))
    spikes_hd = np.zeros((num_neuron, data_length))

    # Calculate the absolute difference between rf_hd and hd_sim for all neurons and time steps
    distances = np.abs(rf_hd[:, np.newaxis] - hd_sim)

    # Wrap distances greater than pi
    distances = np.minimum(distances, 2 * np.pi - distances)

    # Calculate the squared distances for all neurons and time steps
    distances_squared = distances ** 2

    # Calculate the response for all neurons and time steps using vectorized operations
    response = np.log(back_firing) + (np.log(peak_firing / back_firing)) * np.exp(-distances_squared / (2 * tuning_width))

    # Calculate rate_hd for all neurons and time steps   
    rate_hd = np.exp(response)

    # Generate spikes_hd using vectorized operations
    spikes_hd = np.random.poisson(lam=rate_hd)
    
    df_head = pd.DataFrame(spikes_hd.T)
    column_mapping = {col: f'head_neuron_{col}' for col in df_head.columns}
    # Rename the columns using the mapping
    df_head.rename(columns=column_mapping, inplace=True)
        
    return df_head, hd_sim

def simulate_state(num_neuron, num_states, frequency, data_length, noise_rate = 0.1):
    
    if isinstance(frequency, (int, float, complex)):
        state_fr = np.repeat(frequency, num_states)
    else:
        state_fr = np.array(frequency)

    state_sim = np.random.choice(np.arange(1, num_states + 1), data_length, replace=True)
    state_tuned = np.random.choice(np.arange(1, num_states + 1), num_neuron, replace=True)
    state_mat = np.zeros((num_neuron, num_states))
    state_mat[np.arange(num_neuron), state_tuned - 1] = 1

    rate_mat = state_mat * state_fr

    rate_state = np.zeros((num_neuron, data_length))
    spikes_state = np.zeros((num_neuron, data_length))
    

    rate_state = rate_mat[:, state_sim - 1]
    noise = np.random.poisson(lam=noise_rate, size=rate_state.shape)
    spikes_state = np.random.poisson(lam=rate_state) + noise
    
    df_states = pd.DataFrame(spikes_state.T)
    df_rate = pd.DataFrame(rate_mat.T)
    column_mapping = {col: f'state_neuron_{col}' for col in df_states.columns}
    # Rename the columns using the mapping
    df_states.rename(columns=column_mapping, inplace=True)

    return df_states, state_sim, df_rate 

def compute_absolute_correlation(df, target_column):
    """
    Compute the absolute Pearson correlation coefficient between the target column
    and all other columns in the DataFrame.

    Parameters:
    - df: pandas DataFrame
      The DataFrame containing the data.
    - target_column: str
      The name of the target column for correlation measurement.

    Returns:
    - correlation_distances: pandas Series
      A Series containing the absolute correlation distances for each column
      (excluding the target column).
    """
    # Check if the target column exists in the DataFrame
    if target_column not in df.columns:
        raise ValueError(f"'{target_column}' not found in the DataFrame.")

    # Calculate the absolute correlation distances
    correlation_distances = df.corr().abs()[target_column].drop(target_column)

    return correlation_distances

def plot_correlation_distances(correlation_distances, target_column):
    """
    Plot the correlation distances between the target column and all other columns
    using the same style as in the provided code.

    Parameters:
    - correlation_distances: pandas Series
      A Series containing the absolute correlation distances for each column.
    - df: pandas DataFrame
      The DataFrame containing the data.
    - target_column: str
      The name of the target column for correlation measurement.
    """
    # Sort the distances in descending order and get the corresponding column names
    sorted_correlation_distances = correlation_distances.sort_values(ascending=False)
    sorted_columns = sorted_correlation_distances.index

    # Define colors for "head" and "state" variables
    colors = []
    for column in sorted_columns:
        if column.startswith("head"):
            colors.append("blue")  # You can choose any color you prefer for "head" variables
        elif column.startswith("state"):
            colors.append("red")  # You can choose any color you prefer for "state" variables
        else:
            colors.append("gray")  # You can set a default color for other variables

    # Plot the sorted absolute correlation distances with different colors
    plt.figure(figsize=(8, 6))
    bars = plt.bar(sorted_columns, sorted_correlation_distances, color=colors)
    plt.xlabel('Neurons')
    plt.ylabel('Absolute Correlation Distance')
    plt.title(f'Sorted Absolute Correlation Distance of {target_column} to Other Columns (Descending)')
    plt.xticks(rotation=45)
    
    # Replace x-axis labels with empty strings
    plt.gca().set_xticklabels(['' for _ in sorted_columns])

    # Create custom legends for "head" and "state" variables
    legend_elements = [
        Line2D([0], [0], color='blue', lw=4, label='Head Neurons'),
        Line2D([0], [0], color='red', lw=4, label='State Neurons'),
    ]

    plt.legend(handles=legend_elements)

    plt.show()


def get_sim_id(file):
    file_end = str.split(file,"/")[1]
    file_final = str.split(file_end, ".")[:-1]
    return ".".join(file_final)


def plot_mse_distances(target_column, df):
    # Create a StandardScaler to normalize the DataFrame
    scaler = StandardScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    # Calculate the mean squared error (MSE) distances from the target column
    mse_distances = []
    for column in df_normalized.columns:
        if column != target_column:
            mse = mean_squared_error(df_normalized[target_column], df_normalized[column])
            mse_distances.append(mse)

    # Sort the distances in ascending order and get the corresponding column names
    sorted_mse_distances, sorted_columns = zip(*sorted(zip(mse_distances, df_normalized.columns)))

    # Define colors for "head" and "state" variables
    colors = []
    for column in sorted_columns:
        if column.startswith("head"):
            colors.append("blue")  # You can choose any color you prefer for "head" variables
        elif column.startswith("state"):
            colors.append("red")  # You can choose any color you prefer for "state" variables
        else:
            colors.append("gray")  # You can set a default color for other variables

    # Plot the sorted mean MSE distances with different colors
    plt.figure(figsize=(8, 6))
    bars = plt.bar(sorted_columns, sorted_mse_distances, color=colors)
    plt.xlabel('Columns')
    plt.ylabel('Mean Squared Error Distance')
    plt.title(f'Sorted MSE Distance of {target_column} to Other Columns (Ascending)')
    plt.xticks(rotation=45)
    
    # Replace x-axis labels with empty strings
    plt.gca().set_xticklabels(['' for _ in sorted_columns])

    legend_elements = [
        Line2D([0], [0], color='blue', lw=4, label='Head Neurons'),
        Line2D([0], [0], color='red', lw=4, label='State Neurons'),
    ]

    plt.legend(handles=legend_elements)

    plt.show()



def plot_transition_graph(transition_matrix, path):
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes corresponding to the states
    num_states = transition_matrix.shape[0]
    for i in range(num_states):
        G.add_node(i)

    for i in range(num_states):
        for j in range(num_states):
            if transition_matrix[i, j] > 0:
                G.add_edge(i, j, weight=transition_matrix[i, j])

    pos = nx.spring_layout(G)  # Layout for positioning nodes
    labels = {i: f'State {i+1}' for i in range(num_states)}  # Node labels

    # Draw the graph
    nx.draw(G, pos, with_labels=True, labels=labels, node_size=800, node_color='skyblue',
            font_size=10, font_color='black',
            font_weight='bold', width=[d['weight'] * 4 for (u, v, d) in G.edges(data=True)],
            arrowstyle='-')

    # Save plot
    if path:
        file_path = f'models/{path}/graph.png'
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path, dpi=300)  # Example: PNG format with 300 DPI

    # Display the plot
    plt.show()



def scatterplot_with_color(y_head, states_head, state_labels=None, path=''):
    use_color = True
    if not state_labels:
        use_color = False
        state_labels = np.ones(len(y_head))
    
    data_df = pd.DataFrame({
        'HeadDirection': y_head,
        'StateHead': states_head,
        "StateCategory" : state_labels
    })
    
    # Create custom x-axis labels
    if use_color:
        palette = sns.color_palette("Set1", n_colors=len(set(state_labels)))
        state_labels = [f'State {i}' for i in range(1, len(set(states_head)) + 1)]

    # Create the plot using Seaborn with coloring by 'StateHead'
    plt.figure(figsize=(10, 6))
    if use_color:
        sns.scatterplot(data=data_df, x='StateHead', y='HeadDirection', hue='StateCategory', palette=palette,
                    alpha=0.7, s=80)  # Adjust the size (s) and alpha for aesthetics
    else: 
        sns.scatterplot(data=data_df, x='StateHead', y='HeadDirection',
                    alpha=0.7, s=80)  # Adjust the size (s) and alpha for aesthetics      
    plt.xlabel('State')
    plt.ylabel('Head Direction (Radians)')
    plt.title('Head Direction vs. State Categorization')

    # Save plot
    if path:
        file_path = f'models/{path}/scatterplot.png'
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path, dpi=300, bbox_inches='tight')  # Example: PNG format with 300 DPI

    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_head_direction_vs_state(y_head, states_head, path=''):
    data_df = pd.DataFrame({
        'HeadDirection': y_head,
        'StateCategory': states_head
    })

    # Create custom x-axis labels
    n_states = len(set(states_head))
    state_labels = [f'State {i}' for i in range(1, n_states + 1)]
    palette = sns.color_palette("husl", n_colors=len(set(states_head)))


    # Create the plot using Seaborn
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data_df, x='StateCategory', y='HeadDirection', palette=palette, alpha=0.7, s=80)  # Adjust the size (s) and alpha for aesthetics
    plt.xticks(range(n_states), labels=state_labels, rotation=45, ha='right')  # Customize x-axis labels
    plt.xlabel('State')
    plt.ylabel('Head Direction (Radians)')
    plt.title('Head Direction vs. State Categorization')

    # Save plot
    if path:
        file_path = f'models/{path}/scatterplot.png'
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path, dpi=300, bbox_inches='tight')  # Example: PNG format with 300 DPI

    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_polar_scatter(y_head, states_head, path=''):
    states = states_head
    n_states = len(set(states))
    angle_data = y_head
    t_end = len(states)
    palette = sns.color_palette(cc.glasbey, n_states)
    legend_labels = [f"State {i + 1}" for i in range(n_states)]

    # Create a time vector
    t = np.arange(1, t_end + 1) / t_end + 0.05

    # Create a polar scatter plot
    plt.figure(figsize=(7, 6))
    ax = plt.subplot(111, polar=True)
    scatter = ax.scatter(y_head, t, c=[palette[x] for x in states], s=20, alpha=1)

    # Remove axis labels and ticks
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Set x and y labels to blank
    ax.set_xlabel(" ")
    ax.set_ylabel(" ")

    # Create legend handles with the specified colors
    legend_handles = [Patch(color=palette[i]) for i in range(n_states)]

    # Create the legend
    plt.legend(handles=legend_handles, labels=legend_labels, loc='upper right', title="States", bbox_to_anchor=(1.2, 1))

    plt.title("Polar Scatter Plot")

    # Optionally, save the plot to a file
    if path:
        plt.savefig(path, bbox_inches='tight')

    # Show the plot
    plt.show()

# Example usage:
# plot_polar_scatter(y_head, states_head, path='polar_scatter.png')


def upper_triangular_values(matrix):
    values_dict = {}
    n = matrix.shape[0]

    for i in range(n):
        for j in range(i + 1, n):
            values_dict[(i, j)] = matrix[i, j]

    sorted_values_dict = dict(sorted(values_dict.items(), key=lambda item: item[1], reverse = False))

    return sorted_values_dict


def correlation_linkage(X):
    corr_matrix =  1 - np.abs(np.corrcoef(X))
    
    sorted_dict = upper_triangular_values(corr_matrix)
    n = X.shape[0]
    num_members = { i :  1 for i in range(n)}
    index_map = {i : i for i in range(n)}
    
    linkage_matrix = np.zeros((n-1, 4), dtype=float)
    idx = n
    for k in range(len(sorted_dict)):
        # Find the next closest pair
        i, j = list(sorted_dict.keys())[k]
        
        # Find which cluster the data points are member of
        i_new = index_map[i]
        j_new = index_map[j]
        
        # If they are member of the same cluster, continue to next itteration
        if i_new == j_new:
            print("cool")
            continue
        
        # Save that they are member of the next cluster that is to be made
        index_map[i] = idx
        index_map[j] = idx
        
        # Save the new cluster and it's children
        num_members[idx] = num_members[i_new] + num_members[j_new]
        
        # Update linkage matrix
        linkage_matrix[idx - n , 0] = i_new
        linkage_matrix[idx - n , 1] = j_new
        linkage_matrix[idx - n , 2] = list(sorted_dict.values())[k]
        linkage_matrix[idx - n , 3] = num_members[idx]
        
        # Update index
        idx += 1
    
    return linkage_matrix


def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))

def ravel_index(indices, shape):
    flattened_index = 0
    for i in range(len(indices)):
        flattened_index += indices[i] * (torch.prod(torch.tensor(shape[i+1:])).item() if i+1 < len(shape) else 1)
    return flattened_index

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    exp_scores = np.exp(x - np.max(x))  # Subtracting the maximum value for numerical stability
    return exp_scores / np.sum(exp_scores, axis=0)

def head_direction_trans_mat(num_states):
    trans_mat = np.zeros((num_states,num_states))
    if num_states == 2:
        for i in range(num_states):
            vector = np.ones(2) / 2
            trans_mat[i,:] = vector
    else:
        for i in range(num_states):
            vector = np.zeros(num_states)
            rand_vec = np.ones(3) / 3
            if i == 0:
                vector[-1] = rand_vec[0]
                vector[0] = rand_vec[1]
                vector[1] = rand_vec[2]
            elif i==num_states-1:
                vector[0] = rand_vec[0]
                vector[(i-1)] = rand_vec[1]
                vector[i] = rand_vec[2]
            else:
                vector[(i - 1):(i + 2)] = rand_vec
            trans_mat[i,:] = vector

    return trans_mat

def simulate_markov_chain(transition_matrix, t):
    num_states = len(transition_matrix)
    initial_state = np.random.randint(0, num_states)
    states = [initial_state]

    current_state = initial_state
    for _ in range(t - 1):
        next_state = np.random.choice(np.arange(num_states), p=transition_matrix[current_state])
        states.append(next_state)
        current_state = next_state

    return np.array(states)


def simulate_state_head(num_neuron, num_states, frequency, data_length, noise_rate = 0.1):
    
    if isinstance(frequency, (int, float, complex)):
        state_fr = np.repeat(frequency, num_states)
    else:
        state_fr = np.array(frequency)

    trans_mat = head_direction_trans_mat(num_states=num_states)
    state_sim = simulate_markov_chain(trans_mat, data_length)
    
    state_tuned = np.random.choice(np.arange(1, num_states + 1), num_neuron, replace=True)
    state_mat = np.zeros((num_neuron, num_states))
    state_mat[np.arange(num_neuron), state_tuned - 1] = 1

    rate_mat = state_mat * state_fr

    rate_state = np.zeros((num_neuron, data_length))
    spikes_state = np.zeros((num_neuron, data_length))
    

    rate_state = rate_mat[:, state_sim - 1]
    noise = np.random.poisson(lam=noise_rate, size=rate_state.shape)
    spikes_state = np.random.poisson(lam=rate_state) + noise
    
    df_states = pd.DataFrame(spikes_state.T)
    df_rate = pd.DataFrame(rate_mat.T)
    column_mapping = {col: f'state_neuron_{col}' for col in df_states.columns}
    # Rename the columns using the mapping
    df_states.rename(columns=column_mapping, inplace=True)

    return df_states, state_sim, df_rate 


def count_instances(arr):
    """
    Counts the number of instances of each element in an array and presents it in a table.
    
    Parameters:
    arr (numpy.ndarray or list): Input array.
    
    Returns:
    pandas.DataFrame: DataFrame containing counts of each unique element.
    """
    # Convert array to pandas Series for counting
    series = pd.Series(arr)
    
    # Count occurrences of each unique element
    counts = series.value_counts().reset_index()
    counts.columns = ['Element', 'Count']
    
    return counts

from collections import Counter
def find_non_unique_elements(lst):
    # Count all elements in the list
    counts = Counter(lst)
    # Filter and return only elements that have a count greater than 1
    non_unique = [item for item, count in counts.items() if count > 1]
    return non_unique


def find_non_unique_elements(lst):
    return [item for item, count in Counter(lst).items() if count > 1]

def best_mapping(y_true, y_pred):
    unique_pred = np.unique(y_pred)
    unique_true = np.unique(y_true)
    
    best_mapping = {}
    used_true_labels = set()

    for pred_label in unique_pred:
        min_dist = float("inf")
        best_true_label = None

        for true_label in unique_true:
            if true_label in used_true_labels:
                continue
            cat_state = (y_pred == pred_label).astype(int)
            real_state = (y_true == true_label).astype(int)
            dist = np.linalg.norm(real_state - cat_state)

            if dist < min_dist:
                min_dist = dist
                best_true_label = true_label

        if best_true_label is not None:
            best_mapping[pred_label] = best_true_label
            used_true_labels.add(best_true_label)

    # Handle extra states in y_pred not present in y_true
    unused_true_labels = set(unique_true) - used_true_labels
    unused_pred_labels = set(unique_pred) - set(best_mapping.keys())

    for pred_label in unused_pred_labels:
        if unused_true_labels:
            best_mapping[pred_label] = unused_true_labels.pop()
        else:
            best_mapping[pred_label] = pred_label

    y_pred_mapped = np.array([best_mapping[var] for var in y_pred])

    return y_pred_mapped

def best_mapping_new(y_true, y_pred):
    unique_pred = np.unique(y_pred)
    unique_true = np.unique(y_true)
    
    best_mapping = {}

    for pred_label in unique_pred:
        min_dist = float("inf")
        best_true_label = None

        for true_label in unique_true:
            cat_state = (y_pred == pred_label).astype(int)
            real_state = (y_true == true_label).astype(int)
            dist = np.linalg.norm(real_state - cat_state)

            if dist < min_dist:
                min_dist = dist
                best_true_label = true_label

        best_mapping[pred_label] = best_true_label
    
    # Finding the most important mapping
    # All mappings that are not unique and not best is mapped to itself
    not_unique = find_non_unique_elements(best_mapping.values())
    for label in not_unique:
        min_dist = float("inf")
        best_label = None
        for key in best_mapping.keys():
            if best_mapping[key] != label:
                continue
            cat_state = (y_pred == key).astype(int)
            real_state = (y_true == label).astype(int)
            dist = np.linalg.norm(real_state - cat_state)
            if dist < min_dist:
                min_dist = dist
                best_label = key
            
        for key in best_mapping.keys():
            if best_mapping[key] != label:
                continue
            if key is not best_label:
                best_mapping[key] = key
            


    y_pred_mapped = np.array([best_mapping[var] for var in y_pred])

    return y_pred_mapped, best_mapping



def sample_points_on_circle(radius, num_points):
    angles = np.random.uniform(0, 2 * np.pi, num_points)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    points = np.column_stack((x, y))
    return points

def pairwise_distances(points):
    num_points = len(points)
    distances = np.zeros((num_points, num_points))
    
    for i in range(num_points):
        for j in range(num_points):
            distances[i, j] = np.linalg.norm(points[i] - points[j])
    
    np.fill_diagonal(distances, float("inf"))
    return distances

def closest_points(distance_matrix, n):
    # Get indices of n closest points for each row
    closest_indices = np.argsort(distance_matrix, axis=1)[:, :n]
    
    # Initialize list to store boolean vectors
    closest_points_list = []
    
    # Iterate over each row
    for indices in closest_indices:
        # Create boolean vector
        closest_bool = np.zeros(distance_matrix.shape[1], dtype=bool)
        closest_bool[indices] = True
        closest_points_list.append(closest_bool)
    
    return closest_points_list

def transition_matrix(distances, closest_points_list):
    trans_mat = np.zeros_like(distances)
    for i, truth_vec in enumerate(closest_points_list):
        sub_dist = distances[i][truth_vec]
        prob_vec = softmax(sub_dist)
        trans_mat[i][truth_vec] = prob_vec
    return trans_mat

def create_circle_trans_mat(num_points, radius, n_closest):
    points = sample_points_on_circle(radius, num_points)
    distances = pairwise_distances(points)
    closest_points_list = closest_points(distances, n_closest)
    trans_mat = transition_matrix(distances, closest_points_list)
    return trans_mat


def sample_points_on_sphere(radius, num_points):
    # Generate random azimuthal angles (longitude) and polar angles (latitude)
    azimuthal_angles = np.random.uniform(0, 2 * np.pi, num_points)
    polar_angles = np.random.uniform(0, np.pi, num_points)
    
    # Convert spherical coordinates to Cartesian coordinates
    x = radius * np.sin(polar_angles) * np.cos(azimuthal_angles)
    y = radius * np.sin(polar_angles) * np.sin(azimuthal_angles)
    z = radius * np.cos(polar_angles)
    
    points = np.column_stack((x, y, z))
    return points


def create_sphere_trans_mat(num_points, radius, n_closest):
    points = sample_points_on_sphere(radius, num_points)
    distances = pairwise_distances(points)
    closest_points_list = closest_points(distances, n_closest)
    trans_mat = transition_matrix(distances, closest_points_list)
    return trans_mat



def simulate_state_circle(num_neuron, num_states, frequency, data_length, radius = 1, n_closest = 3, noise_rate = 0.1):
    
    if isinstance(frequency, (int, float, complex)):
        state_fr = np.repeat(frequency, num_states)
    else:
        state_fr = np.array(frequency)

    trans_mat = create_circle_trans_mat(num_states, radius, n_closest)
    state_sim = simulate_markov_chain(trans_mat, data_length)
    
    state_tuned = np.random.choice(np.arange(1, num_states + 1  ), num_neuron, replace=True)
    state_mat = np.zeros((num_neuron, num_states))
    state_mat[np.arange(num_neuron), state_tuned - 1] = 1

    rate_mat = state_mat * state_fr

    rate_state = np.zeros((num_neuron, data_length))
    spikes_state = np.zeros((num_neuron, data_length))
    

    rate_state = rate_mat[:, state_sim - 1]
    noise = np.random.poisson(lam=noise_rate, size=rate_state.shape)
    spikes_state = np.random.poisson(lam=rate_state) + noise
    
    df_states = pd.DataFrame(spikes_state.T)
    df_rate = pd.DataFrame(rate_mat.T)
    column_mapping = {col: f'state_neuron_{col}' for col in df_states.columns}
    # Rename the columns using the mapping
    df_states.rename(columns=column_mapping, inplace=True)

    return df_states, state_sim, df_rate 


def simulate_state_sphere(num_neuron, num_states, frequency, data_length, radius = 1, n_closest = 3, noise_rate = 0.1):
    
    if isinstance(frequency, (int, float, complex)):
        state_fr = np.repeat(frequency, num_states)
    else:
        state_fr = np.array(frequency)

    trans_mat = create_sphere_trans_mat(num_states, radius, n_closest)
    state_sim = simulate_markov_chain(trans_mat, data_length)
    
    state_tuned = np.random.choice(np.arange(1, num_states + 1), num_neuron, replace=True)
    state_mat = np.zeros((num_neuron, num_states))
    state_mat[np.arange(num_neuron), state_tuned - 1] = 1

    rate_mat = state_mat * state_fr

    rate_state = np.zeros((num_neuron, data_length))
    spikes_state = np.zeros((num_neuron, data_length))
    

    rate_state = rate_mat[:, state_sim - 1]
    noise = np.random.poisson(lam=noise_rate, size=rate_state.shape)
    spikes_state = np.random.poisson(lam=rate_state) + noise
    
    df_states = pd.DataFrame(spikes_state.T)
    df_rate = pd.DataFrame(rate_mat.T)
    column_mapping = {col: f'state_neuron_{col}' for col in df_states.columns}
    # Rename the columns using the mapping
    df_states.rename(columns=column_mapping, inplace=True)

    return df_states, state_sim, df_rate 


def simulate_ensemble(num_state_list, num_neuron_list, frequency_list, data_length, method = "head", radius = 1, n_closest_list = None, noise_rate = 0.1):
    df_ensamble = pd.DataFrame([])
    y_ensamble = np.array([])
    for i in range(len(num_state_list)):
        if n_closest_list is not None:
            n_closest = n_closest_list[i]   
        if method == "head":
            df, y, _ = simulate_state_head(num_neuron_list[i], num_state_list[i], frequency_list[i], data_length, noise_rate = noise_rate)
        elif method == "circle":
            df, y, _ = simulate_state_circle(num_neuron_list[i], num_state_list[i], frequency_list[i], data_length, radius = radius, n_closest=n_closest, noise_rate = noise_rate)
        elif method == "sphere":
            df, y, _ = simulate_state_sphere(num_neuron_list[i], num_state_list[i], frequency_list[i], 
                                   data_length, radius = radius, n_closest=n_closest, noise_rate = noise_rate)
        else:
            df, y, _ = simulate_state(num_neuron_list[i], num_state_list[i], frequency_list[i], data_length, noise_rate = noise_rate)
        if df_ensamble.empty:
            df_ensamble = df
            y_ensamble = y.reshape(-1, 1)
            ensemble_asignment = np.repeat(i, num_neuron_list[i])
        else:
            df_ensamble = pd.concat([df_ensamble, df], axis = 1)
            y_ensamble = np.concatenate((y_ensamble, y.reshape(-1, 1)), axis = 1)
            new_asignment = np.repeat(i, num_neuron_list[i])
            ensemble_asignment = np.concatenate((ensemble_asignment, new_asignment))
    
    return df_ensamble, y_ensamble, ensemble_asignment

# Generates a random list 
def generate_random_list(min_len=2, max_len=5, min_state=3, max_state=20, uniform=False):
    """
    Generates a random list 
    """
    # Define the range of integers and their probabilities
    integers = np.arange(min_state, max_state+1) 
    len_int = np.arange(min_len, max_len+1) 
    
    # Higher numbers is less likely
    if not uniform:
        # Decreasing probabilities for non-uniform distribution
        probabilities = np.linspace(1, 0.1, len(integers))  
        prob_len = np.linspace(1, 0.1, len(len_int))

        # Generate a random list length
        length = np.random.choice(len_int, p=prob_len/prob_len.sum())
        
        # Generate the random list
        random_list = np.random.choice(integers, size=length, p=probabilities/probabilities.sum())
    else:
        # Uniform probabilities
        length = np.random.choice(len_int)
        random_list = np.random.choice(integers, size=length)

    return random_list.tolist() 

def plot_umap(states, X_umap, s = 0.5, n_neighbors = 10, min_dist = 1):
    X_scaled = StandardScaler().fit_transform(X_umap)

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist)

    embedding = reducer.fit_transform(X_scaled)

    custom_palette = sns.color_palette(cc.glasbey, max(states) + 1)

    plt.scatter(
        embedding[:, 0],
        embedding[:, 1], c=[custom_palette[x] for x in states], s=s)
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projection of neuron activity')

    legend_patches = [plt.Line2D([0], [0], marker='o', color='w', label='State {}'.format(state), markersize=10, markerfacecolor=custom_palette[state]) for state in np.unique(states)]
    # Adjust the bbox_to_anchor to change the legend position
    plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.show()



def get_ensemble_state(vector1, vector2):
    combo_to_number = {}
    counter = 0
    combination = []

    for i in range(len(vector1)):
        combo = (vector1[i], vector2[i])
        combination.append(combo)
        if combo not in combo_to_number:
            combo_to_number[combo] = counter
            counter += 1

    return [combo_to_number[combo] for combo in combination]

def get_ensemble_state_multi(vector_list):
    ensemble_state = vector_list[0]
    for i in range(1, len(vector_list)):
        ensemble_state = get_ensemble_state(ensemble_state, vector_list[i])
    return ensemble_state

def find_combinations(n, start=2, path=[], result=[]):
    # Base case: If n is 1, add the current path to the result
    if n == 1:
        if len(path) > 1:  # Exclude single numbers
            result.append(path[:])
        return
    
    # Explore all possible factors starting from 'start'
    for i in range(start, n + 1):
        if n % i == 0:
            # Add current factor to the path
            path.append(i)
            # Recursively find combinations for the remaining part
            find_combinations(n // i, i, path, result)
            # Backtrack: Remove the last added factor
            path.pop()

    return result



def sparse_transition_matrix(num_states, sparsity=0.5):
    """
    Create a transition matrix that is both sparse and ensures that all states are connected.
    """
    connected = False
    while not connected:
        matrix = np.zeros((num_states, num_states))
        for i in range(num_states):
            for j in range(num_states):
                if i != j and np.random.rand() > sparsity:
                    matrix[i, j] = np.random.rand()
            if not np.any(matrix[i]):
                j = np.random.randint(0, num_states)
                while j == i:
                    j = np.random.randint(0, num_states)
                matrix[i, j] = np.random.rand()
        for i in range(num_states):
            matrix[i, i] = np.random.uniform(0.01, 0.1)
            matrix[i] /= matrix[i].sum()
        G = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
        connected = nx.is_strongly_connected(G)
    return matrix


def simulate_neuron_data(num_time_steps, ensemble_sizes, ensemble_states, rate_interval, sparsity=0.5, noise_level=0.1):
    num_neurons = sum(ensemble_sizes)
    ensemble_transition_matrices = []
    neuron_rate_parameters = []

    for ensemble_index, (size, states) in enumerate(zip(ensemble_sizes, ensemble_states)):
        transition_matrix = sparse_transition_matrix(states, sparsity)
        ensemble_transition_matrices.append(transition_matrix)
        
        # Generate neuron-specific rate parameters within this ensemble
        for _ in range(size):
            active_state = np.random.randint(states)
            rate_params = np.zeros(states)
            # High rate for the active state
            rate_params[active_state] = np.random.uniform(*rate_interval)

            # Lower rates for states with non-zero transition probability to the active state
            for s in range(states):
                if transition_matrix[s, active_state] > 0 and s != active_state:
                    rate_params[s] = rate_params[active_state] * np.random.uniform(0.1, 0.5)  # Smaller fraction of the main rate

            neuron_rate_parameters.append(rate_params)

    all_firing_counts_matrix = np.zeros((num_neurons, num_time_steps))
    all_states_matrix = np.zeros((len(ensemble_sizes), num_time_steps), dtype=int)
    neuron_index = 0

    for ensemble_index, size in enumerate(ensemble_sizes):
        transition_matrix = ensemble_transition_matrices[ensemble_index]
        states = ensemble_states[ensemble_index]

        # Generate one state sequence for the entire ensemble
        current_state = np.random.randint(states)
        ensemble_state_sequence = []
        for t in range(num_time_steps):
            new_state = np.random.choice(states, p=transition_matrix[current_state])
            ensemble_state_sequence.append(new_state)
            current_state = new_state

        # Apply this state sequence to the matrix for this ensemble
        all_states_matrix[ensemble_index] = ensemble_state_sequence

        # Apply this state sequence to all neurons in the ensemble
        for i in range(size):
            for t in range(num_time_steps):
                rate_param = neuron_rate_parameters[neuron_index][ensemble_state_sequence[t]]
                all_firing_counts_matrix[neuron_index, t] = np.random.poisson(rate_param)
            neuron_index += 1

    ensemble_membership = np.concatenate([np.full(size, i) for i, size in enumerate(ensemble_sizes)])
    noisy_firing_counts_matrix = all_firing_counts_matrix + np.random.normal(0, noise_level, all_firing_counts_matrix.shape)
    noisy_firing_counts_matrix = np.maximum(0, np.round(noisy_firing_counts_matrix))

    return noisy_firing_counts_matrix.T, all_states_matrix, ensemble_membership


def create_cooccurrence_matrix(labels):
    n_samples = labels.shape[0]
    cooccurrence_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            if labels[i] == labels[j]:
                cooccurrence_matrix[i][j] += 1
    return cooccurrence_matrix

def create_cooccurrence_matrix_old(labels):
    n_samples = labels.shape[1]
    cooccurrence_matrix = np.zeros((n_samples, n_samples))
    for label in labels:
        for i in range(n_samples):
            for j in range(n_samples):
                if label[i] == label[j]:
                    cooccurrence_matrix[i][j] += 1
    return cooccurrence_matrix



def find_best_model(data, n_components_range=range(2, 10), num_retrains=3, n_splits=5):
    # Adjust zero values to avoid -inf in predictions
    if np.any(data == 0):
        data += 1

    results = []
    kf = KFold(n_splits=n_splits)  # 5-fold cross-validation

    for n_components in n_components_range:
        cv_likelihoods = []
        for train_index, test_index in kf.split(data):
            train_data, test_data = data[train_index], data[test_index]
            best_log_likelihood = -np.inf  # Initialize with a very low log likelihood
            best_model = None

            # Using ThreadPoolExecutor to parallelize retraining
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(train_model, train_data, test_data, n_components) for _ in range(num_retrains)]
                for future in as_completed(futures):
                    log_likelihood, model = future.result()
                    if log_likelihood > best_log_likelihood:
                        best_log_likelihood = log_likelihood
                        best_model = model

            if best_log_likelihood > -np.inf:
                cv_likelihoods.append(best_log_likelihood)

        cv_likelihood = np.mean(cv_likelihoods)
        results.append((n_components, cv_likelihood))
        print((n_components, cv_likelihood))
    print(results)
        

    # Select the model with the best criteria
    best_model = max(results, key=lambda x: x[1])  # Example: prioritize log_like, then BIC, then AIC
    return best_model


def odd_even_cv(data, n_components_range=range(2, 10), num_retrains=3):
    """
    Perform odd-even cross-validation for a range of component numbers in an HMM.

    Parameters:
    - data (np.ndarray): The dataset to model with HMM.
    - n_components_range (range): A range of component numbers to evaluate.
    - num_retrains (int): Number of retraining iterations for each model configuration.

    Returns:
    - tuple: The best model configuration (n_components, average log likelihood).
    """
    if np.any(data == 0):
        data += 1
    results = []
    indices = np.arange(data.shape[0])  # Generate an array of indices from 0 to n-1
    even_indices = indices[indices % 2 == 0]  # Filter even indices
    odd_indices = indices[indices % 2 != 0]  # Filter odd indices
    
    odd_data = data[odd_indices]
    even_data = data[even_indices]

    for n_components in n_components_range:
        cv_likelihoods = []
        for _ in range(num_retrains):
            # Train on odd, score on even
            model_odd = hmm.PoissonHMM(n_components=n_components)
            model_odd.fit(odd_data)
            log_likelihood_odd = model_odd.score(even_data)

            # Train on even, score on odd
            model_even = hmm.PoissonHMM(n_components=n_components)
            model_even.fit(even_data)
            log_likelihood_even = model_even.score(odd_data)

            # Aggregate results
            cv_likelihoods.extend([log_likelihood_odd, log_likelihood_even])

        # Calculate the average cross-validated likelihood
        cv_likelihood = np.mean(cv_likelihoods)
        print((n_components, cv_likelihood))
        results.append((n_components, cv_likelihood))

    # Select the model with the highest average log likelihood
    best_model = max(results, key=lambda x: x[1])  # Changed to max if we assume higher is better
    return best_model



def find_best_model(data, n_components_range = range(2, 10), num_retrains = 3, n_splits = 5):
    results = []
    kf = KFold(n_splits=n_splits)  # 5-fold cross-validation
    for n_components in n_components_range:
        model = hmm.PoissonHMM(n_components=n_components)
        cv_likelihoods = []
        
        for train_index, test_index in kf.split(data):
            train_data, test_data = data[train_index], data[test_index]
            best_log_likelihood = -np.inf  # Initialize with a very low log likelihood
            for _ in range(num_retrains):
                model = hmm.PoissonHMM(n_components=n_components)
                model.fit(train_data)
                log_likelihood = model.score(test_data)
                if log_likelihood > best_log_likelihood:
                    best_log_likelihood = log_likelihood
                    best_model = model
            if best_log_likelihood > -np.inf:
                cv_likelihoods.append(best_log_likelihood)
            
        cv_likelihood = np.mean(cv_likelihoods)
        print((n_components, cv_likelihood))
        
        results.append((n_components, cv_likelihood))


    # Select the model with the best criteria
    best_model = max(results, key=lambda x: x[1]) 
    return best_model

def number_of_clusters(data):
    Z = linkage(data)
    dist_range = np.arange(5.0, 150, .5)
    silhouette_scores = []

    for dist in dist_range:
        labels = fcluster(Z, dist, criterion='distance')
        if 1 < np.unique(labels).size < len(data):  # Check if the number of clusters is within the valid range
            score = silhouette_score(data, labels, metric='euclidean')
            silhouette_scores.append(score)
        else:
            silhouette_scores.append(-1)  # Append a negative score to indicate an invalid clustering scenario


    optimal_dist = dist_range[np.argmax(silhouette_scores)]
    clusters = fcluster(Z, optimal_dist, criterion='distance')
    number_of_clusters = np.unique(clusters).size
    return number_of_clusters

def seperate_and_predict(data, n_retrain = 5):
    num_cluster_full = number_of_clusters(data)

    data_list = seperate_data(data, num_cluster_full)
    model_list = []
    prediction_list = []
    score_list = []
    for d in data_list:
        best_states = find_best_model(d)
        for _ in range(n_retrain):
            model = hmm.PoissonHMM(n_components=best_states[0])
            model.fit(d)
            log_likelihood = model.score(d)
            if log_likelihood > best_log_likelihood:
                best_log_likelihood = log_likelihood
                best_model = model
        model_list.append(best_model)
        prediction_list.append(best_model.predict(d))
        score_list.append(best_log_likelihood)
    
    return data_list, model_list, prediction_list, score_list


# def train_hmm_ensemble(data, n_state_list, m_dimensions, max_iterations, tolerance, confidence_rate, certainty, n_retraining, n_splits):
#     """
#     Trains HMM ensembles on provided data using specified training parameters.
#     """
#     ensemble_assignments = []
#     kfold = KFold(n_splits=n_splits)

#     X = torch.tensor(data)
#     m_dimensions = X.shape[1]
#     for train_index, _ in kfold.split(X):
#         batch_data = X[train_index]
#         m_dimensions = batch_data.shape[1]

#         for _ in range(n_retraining):
#             model = HMM_ensemble(n_state_list=n_state_list, m_dimensions=m_dimensions, max_iterations=max_iterations, tolerance=tolerance, confidence_rate=confidence_rate, certainty=certainty)
#             model = model.to('cpu')
#             with torch.no_grad():
#                 model.fit(batch_data)
#             ensemble_assignments.append(model.ensemble_assignment.numpy())
#             del model
#             torch.cuda.empty_cache()
#     return np.array(ensemble_assignments)

def train_hmm_ensemble(data, n_state_list, m_dimensions, max_iterations, tolerance, confidence_rate, certainty, n_retraining, n_splits):
    """
    Trains HMM ensembles on provided data using specified training parameters.
    """
    ensemble_assignments = []
    kfold = KFold(n_splits=n_splits)

    # Prepare data tensor
    X = torch.tensor(data)
    
    def train_model(batch_data):
        # Local model training function to be run in parallel
        local_assignments = []
        for _ in range(n_retraining):
            model = HMM_ensemble(n_state_list=n_state_list, m_dimensions=batch_data.shape[1], max_iterations=max_iterations, tolerance=tolerance, confidence_rate=confidence_rate, certainty=certainty)
            model = model.to('cpu')
            with torch.no_grad():
                model.fit(batch_data)
            local_assignments.append(model.ensemble_assignment.numpy())
            del model
            torch.cuda.empty_cache()
        return local_assignments
    
    # Use ThreadPoolExecutor to train models in parallel
    with ThreadPoolExecutor() as executor:
        future_to_model = {executor.submit(train_model, X[index]): index for index, _ in kfold.split(X)}
        
        for future in as_completed(future_to_model):
            ensemble_assignments.extend(future.result())
    
    return np.array(ensemble_assignments)

def handle_outliers(data, y_ensemble, model_dict):
    """
    Reassigns outliers to the best fitting models.
    """
    mean_model_dict = {}
    for index, model in model_dict.items():
        new_model = copy.copy(model)
        new_model.lambdas_ = np.mean(model.lambdas_, axis=1).reshape(-1,1)
        new_model.n_features = 1

        mean_model_dict[index] = new_model

    for index in np.where(y_ensemble == -1)[0]:
        data_point = data[index]
        best_score = float('-inf')
        best_model = -1
        for j, model in mean_model_dict.items():
            score = model.score(data_point.reshape(-1, 1))
            if score > best_score:
                best_score = score
                best_model = j
        y_ensemble[index] = best_model
    return y_ensemble

def handle_outliers_max(data, y_ensemble, model_dict):
    """
    Reassigns outliers to the best fitting models.
    """
    max_model_dict = {}
    for index, model in model_dict.items():
        model_list = []
        for lam in model.lambdas_.T:
            # Reinitiate the model to fit a single feature
            new_model = copy.copy(model)
            new_model.lambdas_ = lam.reshape(-1,1)
            new_model.n_features = 1
            
            model_list.append(new_model)
        max_model_dict[index] = model_list

    for index in np.where(y_ensemble == -1)[0]:
        data_point = data[index]
        best_score = float('-inf')
        best_model = -1
        for j, model_list in max_model_dict.items():
            # Find the best individual model in a ensemble
            max_score = float('-inf')
            for model in model_list:
                score = model.score(data_point.reshape(-1, 1))
                if score > max_score:
                    max_score = score
            
            if max_score > best_score:
                best_score = max_score
                best_model = j
        y_ensemble[index] = best_model
    return y_ensemble


def seperate_data(data, states = 12, num_latent = 5, max_iterations = 120, 
                    tolerance = 1e-6, n_retraining = 2, confidence_rate = 1.1,
                    certainty = 0.1, n_splits = 10, min_prob = 0.8, threshold = 0.1,
                    num_retrains = 1, n_components_range = range(2,16)):

    n_state_list = num_latent*[states]
    m_dimensions = data.shape[0]

    all_labels = train_hmm_ensemble(data, n_state_list, m_dimensions,
                                     max_iterations, tolerance, confidence_rate,
                                     certainty, n_retraining, n_splits=n_splits)
    
    cooccurrence_matrix = create_cooccurrence_matrix(all_labels)

    y_ensemble = ensemble_clustering(cooccurrence_matrix, min_prob, threshold)

    num_clusters = len(np.unique(y_ensemble)) - 1*(np.any(y_ensemble == -1))

    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(cooccurrence_matrix)  

    if num_clusters > 1:
        gmm = GaussianMixture(n_components=num_clusters, random_state=0)
        # Fit the GMM to the data
        gmm.fit(data_scaled)
        best_clusters = gmm.predict(data_scaled)
    else:
        # Prepare to store results
        results = {}
        # Loop over a range of cluster counts
        for n in range(2, 15):
            # Initialize the Gaussian Mixture Model
            gmm = GaussianMixture(n_components=n, random_state=0)
            
            # Fit the GMM to the data
            gmm.fit(data_scaled)
            
            # Predict the cluster for each data point
            y_ensemble = gmm.predict(data_scaled)
            
            # Calculate the silhouette score
            score = davies_bouldin_score(data_scaled, y_ensemble)
            # score = silhouette_score(data_scaled, y_ensemble)
            
            # Store results
            results[n] = {
                'score': score,
                'cluster_labels': y_ensemble
            }

        # To retrieve the best model's data:
        best_n = max(results, key=lambda x: results[x]['score'])
        y_ensemble = results[best_n]['cluster_labels']
        
        good_cluster = []
        for i in range(np.max(y_ensemble)+1):
            if sum(y_ensemble == i)/n > threshold:
                good_cluster.append(i)

        for i in range(len(y_ensemble)):
            if y_ensemble[i] not in good_cluster:
                y_ensemble[i] = -1

        num_clusters = max(len(np.unique(y_ensemble)) - 1*(np.any(y_ensemble == -1)),2)

        gmm = GaussianMixture(n_components=num_clusters, random_state=0)
        # Fit the GMM to the data
        gmm.fit(data_scaled)
        best_clusters = gmm.predict(data_scaled)



    # # For each ensemble, train optimal HMM models
    # model_dict = {}
    # for i in np.unique(y_ensemble):
    #     if i < 0:
    #         continue
    #     sub_data = data[:,y_ensemble == i]
    #     n_components = find_best_model(sub_data, n_components_range=n_components_range,
    #                                     num_retrains=num_retrains, n_splits = n_splits)[0]

    #     best_log_likelihood = -np.inf  
    #     best_model = None
    #     with ThreadPoolExecutor() as executor:
    #         futures = [executor.submit(train_model, sub_data, sub_data, n_components) for _ in range(num_retrains)]
    #         for future in as_completed(futures):
    #             log_likelihood, model = future.result()
    #             if log_likelihood > best_log_likelihood:
    #                 best_log_likelihood = log_likelihood
    #                 best_model = model

    #     model_dict[i] = best_model    

    # # For each outlier, cluster to ensemble by evaluating the score
    # y_ensemble = handle_outliers_max(data, y_ensemble, model_dict)
    
    return best_clusters, cooccurrence_matrix



def train_model(train_data, test_data, n_components):
    """Train a single model and compute its log likelihood."""
    model = hmm.PoissonHMM(n_components=n_components)
    model.fit(train_data)
    log_likelihood = model.score(test_data)
    return log_likelihood, model


def combine_label_list(labels1, labels2):
    # Find the maximum label in the first list
    max_label = np.max(labels1)
    
    # Increment labels in the second list to ensure uniqueness
    adjusted_labels2 = labels2 + max_label + 1
    
    # Combine both lists into a new list
    combined_labels = np.concatenate((labels1, adjusted_labels2))
    
    return combined_labels


def ensemble_clustering(co_mat, min_prob = 0.8, threshold = 0.1):
    co_mat = co_mat.copy()
    co_mat_norm = co_mat/co_mat.diagonal().astype(int)
    co_mat_binary = (co_mat_norm > min_prob).astype(int)

    _, y_ensemble = connected_components( co_mat_binary, directed=False, return_labels=True)
    n = y_ensemble.shape[0]
    good_cluster = []
    for i in range(np.max(y_ensemble)+1):
        if sum(y_ensemble == i)/n > threshold:
            good_cluster.append(i)

    for i in range(len(y_ensemble)):
        if y_ensemble[i] not in good_cluster:
            y_ensemble[i] = -1

    return y_ensemble


def create_accuracy_mat(pred):
    """
    Create an accuracy matrix based on pairwise comparisons of predicted latent variables.
    
    Args:
    pred (np.ndarray): 2D array where each column represents predicted latent variables for samples.
    
    Returns:
    np.ndarray: Symmetric accuracy matrix.
    """
    num_latent = pred.shape[1]
    accuracy_mat = np.ones((num_latent, num_latent))  # Initialize with ones
    
    for i in range(num_latent):
        pred_i = pred[:, i]
        for j in range(i + 1, num_latent):
            pred_j = pred[:, j]
            
            # Map predictions
            pred_i_mapped = best_mapping(pred_i, pred_j)
            pred_j_mapped = best_mapping(pred_j, pred_i)
            
            # Calculate accuracies
            accuracy_i_to_j = accuracy_score(pred_i, pred_i_mapped)
            accuracy_j_to_i = accuracy_score(pred_j, pred_j_mapped)
            
            # Compute mean accuracy
            min_accuracy = min(accuracy_i_to_j, accuracy_j_to_i)
            accuracy_mat[i, j] = min_accuracy
            accuracy_mat[j, i] = min_accuracy
    
    return accuracy_mat


from sklearn.metrics import adjusted_mutual_info_score



def create_similarity_matrix(pred):
    """
    Create an accuracy matrix based on pairwise comparisons of predicted latent variables.
    
    Args:
    pred (np.ndarray): 2D array where each column represents predicted latent variables for samples.
    
    Returns:
    np.ndarray: Symmetric accuracy matrix.
    """
    num_latent = pred.shape[1]
    sim_mat = np.ones((num_latent, num_latent))  # Initialize with ones
    
    for i in range(num_latent):
        pred_i = pred[:, i]
        for j in range(i,num_latent):
            pred_j = pred[:, j]
            
            mut_info = adjusted_mutual_info_score(pred_i, pred_j)

            sim_mat[i, j] = mut_info
            sim_mat[j, i] = mut_info
    
    return sim_mat


def update_co_occurance_mat(co_mat, pred, accuracy_mat):
    """
    Update a co-occurrence matrix based on the accuracy matrix and predicted latent variables.
    
    Args:
    co_mat (np.ndarray): The co-occurrence matrix to be updated.
    pred (np.ndarray): 1D array of predicted latent variables for each sample.
    accuracy_mat (np.ndarray): Symmetric accuracy matrix.
    
    Returns:
    np.ndarray: Updated co-occurrence matrix.
    """
    m_dimensions = len(pred)
    
    for i in range(m_dimensions):
        for j in range(i, m_dimensions):
            if i == j:
                co_mat[i, j] += 1  # Increment diagonal elements
            else:
                lat_i = pred[i]
                lat_j = pred[j]
                co_mat[i, j] += accuracy_mat[lat_i, lat_j]
                co_mat[j, i] += accuracy_mat[lat_i, lat_j]  # Symmetric update
    
    return co_mat


