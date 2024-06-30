import numpy as np
from utils import generate_random_list, simulate_neuron_data, create_similarity_matrix, update_co_occurance_mat, create_cooccurrence_matrix, best_mapping_new
import torch
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Model
from model import HMM_ensemble

import datetime

from scipy.cluster.hierarchy import linkage, fcluster

from hmmlearn import hmm 

num_test = 20
num_latent_test = 3

cluster_criteria = "pca" # wcss, reduction or silhouette (or a combo of wcss and silhouette)
# pca now also works as a criteria. Is found to be best
num_latent_estimate_list = []

folder = f"accuracy_co_mat_test_{num_latent_test}"

try:
    orig_arr = np.loadtxt(f"{folder}/accuracy_arr.csv", delimiter=",")
    accuracy_list = list(orig_arr)
except:
    accuracy_list = []

max_len = num_latent_test
min_len = max_len

try:
    orig_arr = np.loadtxt(f"{folder}/ensemble_ami_list.csv", delimiter=",")
    ami_ensemble_score_list = list(orig_arr)
except:
    ami_ensemble_score_list = []

try:
    orig_arr = np.loadtxt(f"{folder}/consensus_ami_list.csv", delimiter=",")
    consensus_ami_list = list(orig_arr)
except:
    consensus_ami_list = []

try: 
    orig_arr = np.loadtxt(f"{folder}/num_latent_list.csv", delimiter=",")
    num_latent_list = list(orig_arr)
except:
    num_latent_list = []


num_prev_test = len(ami_ensemble_score_list)

for j in range(num_test):
    max_state = 12
    min_state = 3
    num_time_steps = 1250
    max_neurons = 200

    min_ensemble_neuron = round(max_neurons/num_latent_test) - 5
    max_ensemble_neuron = min_ensemble_neuron + 10

    ensemble_states = generate_random_list(max_len=max_len, min_len = min_len,
                                            max_state=max_state, min_state=min_state)
    L = len(ensemble_states)

    ensemble_sizes = generate_random_list(min_len=L,max_len=L,
                                            min_state=min_ensemble_neuron,
                                            max_state=max_ensemble_neuron)
    rate_interval = (5.0, 10.0)  # Define the interval for firing rates

    data_full, true_states, ensemble_assignments_true = simulate_neuron_data(num_time_steps, ensemble_sizes, ensemble_states, rate_interval)

    data , data_test = train_test_split(data_full, test_size=0.2, random_state=42, shuffle=False)

    m_dimensions=data_full.shape[1]

    # Config 
    ## Model
    states = 15 # Number of states for each latent in ME-HMM
    num_latent = 12 # Number of latent in ME-HMM
    num_lat_init = num_latent
    ## Stopping criteria
    max_iteration = 120 # Max iterations for ME-HMM
    tolerance = 1e-6 # Tolerance for log-likelihood change
    ## Hyper Parameters
    certainty = 0.1 # Hyper parameter for ME-HMM. slows initial convergence in ensemble probabilities
    confidence_rate = 1.1 # Hyper parameter for ME-HMM. Slows rate of convergence in ensemble probabilities
    ## Number of Ensembles tuning
    n_splits = 4 # Number of data splits for training of multiple ME-HMM
    n_retraining = 10 # Number of ratrains of ME-HMM model for each split
    ## Ensemble Clustering 
    min_prob = 0.8 # Probability cut-off range in normalized cooccurance matrix
    threshold = 0.05 # Cut-off to decide outliers. Each ensemble cluster need a number of neurons above this threshold, else is outlier
    ## HMM training
    n_components_range = range(2,16) # Range of number of states to test when training normal HMM
    num_retrains = 10 # Number of ratrains of HMM model for each split


    co_mat = np.zeros((m_dimensions, m_dimensions))

    num_latent_estimate_list = []
    for t in range(num_retrains):
        current_datetime = datetime.datetime.now()
        formatted_datetime = current_datetime.strftime("%Y,%m,%d;%H,%M")

        n_state_list = num_latent*[states]
        m_dimensions = data.shape[1]

        # Model
        X = torch.tensor(data)
        X_test = torch.tensor(data_test)
        model = HMM_ensemble(n_state_list=n_state_list, m_dimensions=data.shape[1],
                            max_iterations=max_iteration, tolerance=tolerance,
                            confidence_rate=confidence_rate, certainty=certainty)
        # Check if CUDA is available
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move your model to the device
        model.to("cpu")
        with torch.no_grad():
            model.fit(X)
        torch.cuda.empty_cache()
        prediction = model.ensemble_assignment.numpy()

        score = model.forward(X)
        score_test = model.forward(X_test)

        y_pred = model.predict(X).numpy()
        y_pred_test = model.predict(X_test).numpy()

        is_inactive = (model.ensemble_priors < 1e-5).numpy()
        num_inactive =  is_inactive.sum()

        is_active = [not value for value in is_inactive]

        sim_mat = create_similarity_matrix(y_pred_test)
        sim_mat_sub = sim_mat[:,is_active][is_active,:]
        
        # Check which clusters have a ami over the threshold
        # Remove a number of latents according to the number of above threshold clusters
        # Cluster with highers above threshold AMIs are removed first
        num_shared = 0
        ami_threshold = 0.15
        used_arr = np.repeat(False, sim_mat_sub.shape[0])
        shared_sum_list = np.zeros(sim_mat_sub.shape[0])
        for i in range(sim_mat_sub.shape[0]):
            shared_sum_list[i] = (sim_mat_sub[i] > ami_threshold).sum()
        
        sorted_indices = np.argsort(-shared_sum_list)
        for i in sorted_indices:
            if used_arr[i]:
                continue
            shared_sum = shared_sum_list[i]
            if shared_sum > 1:
                used_arr = used_arr | (sim_mat_sub[i] > ami_threshold)
                num_shared += 1
        

        co_mat = update_co_occurance_mat(co_mat, prediction, sim_mat)

        prev_num_latent = num_latent 

        # In general, to cluster correctly we want to be as close to the truth as possible.
        # Howevrer, sightly offerfitting is not usualy a problem
        # Therefor we reduce by over estimate of unncecessary latents
        # and add som slack in case we are too strict.
        # This is due to the fact that we do not to get stuck at a underfitting scenerio
        
        slack = 2
        num_latent_estimate = int(max(num_latent - num_inactive - num_shared, 2))
        num_latent_estimate_list.append(num_latent_estimate)

        num_latent = num_latent_estimate + slack

        del model


        # Save result
        # prediction_mapped = best_mapping(ensemble_assignments_true, prediction)
        accuracy = adjusted_mutual_info_score(ensemble_assignments_true, prediction)
        accuracy_list.append(accuracy)
        accuracy_arr = np.array(accuracy_list)
        mean_accuracy = np.mean(accuracy_arr)

        path = f"test_num_{num_prev_test+j+1}_n{ensemble_sizes}_s{ensemble_states}/retrain_{t+1}"

        df_path = f'{folder}/{path}/df'+ ".csv"
        df_test_path = f'{folder}/{path}/df_test'+ ".csv"

        assignment_path = f'{folder}/{path}/assignment'+ ".csv"
        true_assignment_path = f'{folder}/{path}/true_assignment'+ ".csv"

        co_mat_path = f"{folder}/{path}/co_mat.csv"
        sim_mat_path = f"{folder}/{path}/sim_mat.csv"
        is_inactive_path = f"{folder}/{path}/is_inactive.csv"

        y_path = f"{folder}/{path}/y.csv"
        y_pred_path = f"{folder}/{path}/y_pred.csv"
        y_pred_test_path = f"{folder}/{path}/y_pred_test.csv"

        run_info_path = f'{folder}/{path}/run_info.txt'
        os.makedirs(os.path.dirname(run_info_path), exist_ok=True)

        with open(run_info_path, 'a') as f:
            f.write(f"Accuracy: {accuracy}\n")
            f.write(f"Score Train: {score}\n")
            f.write(f"Score Test: {score_test}\n")
            f.write(f"Path Name: {path}\n")
            f.write(f"Number of latent inactive: {num_inactive}\n")
            f.write(f"Number of latent that share ensemble: {num_shared}\n")
            f.write(f"Current number of active ensembles: {prev_num_latent}\n")
            f.close()   

        np.savetxt(df_path, data, delimiter = ",")
        np.savetxt(df_test_path, data_test, delimiter = ",")

        np.savetxt(true_assignment_path, ensemble_assignments_true, delimiter = ",")
        np.savetxt(assignment_path, prediction, delimiter = ",")

        np.savetxt(y_path, true_states, delimiter=",")
        np.savetxt(y_pred_path, y_pred, delimiter=",")
        np.savetxt(y_pred_test_path, y_pred_test, delimiter=",")

        np.savetxt(co_mat_path, co_mat, delimiter = ",")
        np.savetxt(sim_mat_path, sim_mat, delimiter = ",")
        np.savetxt(is_inactive_path, is_inactive, delimiter = ",")
        
        # Number of latent estimated in each iteration of HMMM
        num_latent_estimate_list_path = f"{folder}/{path}/num_latent_estimate_list.csv"
        np.savetxt(num_latent_estimate_list_path, num_latent_estimate_list, delimiter = ",")

        simulation_info_path = f'{folder}/simulation_info.txt'
        accuracy_arr_path = f'{folder}/accuracy_arr.csv'

        with open(simulation_info_path, "a") as f:
            f.write(f"Mean Accuracy: {mean_accuracy}\n")
        
        np.savetxt(accuracy_arr_path, accuracy_arr, delimiter=",")

    # Predict States
    ## Create linkage matrix from mean AMI matrix
    Z = linkage(co_mat, 'ward')

    if cluster_criteria == "wcss":
        ## Use elbow method to find optimal number of clusters
        last = Z[-num_latent:, 2] # within-cluster sum of squares (WCSS) after each merge  

        acceleration = np.diff(last, 2) - np.diff(last,1)[:-1]  # 2nd derivative of the WCSS
        acceleration_rev = acceleration[::-1] # Reverse the order

        k = acceleration_rev[1:].argmax() + 3 # if k = 0 means 1 clusters
    
    elif cluster_criteria == "reduction":
        k = np.round(np.mean(num_latent_estimate_list))+1
        # k = stats.mode(num_latent_estimate_list)[0]
    

    elif cluster_criteria == "silhouette":
        silhouette_avg = []
        range_n_clusters = np.arange(2, 16)

        for n_clusters in range_n_clusters:
            # Form flat clusters from the hierarchical clustering
            cluster_labels = fcluster(Z, n_clusters, criterion='maxclust')
            
            # Compute the silhouette score
            silhouette_avg.append(silhouette_score(co_mat, cluster_labels))

        acceleration = np.diff(silhouette_avg, 2) + np.diff(silhouette_avg, 1)[:-1]

        k = acceleration.argmax() + 2
     
    elif cluster_criteria == "pca":
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(co_mat)

        # Fit PCA on the standardized data
        pca = PCA()
        principal_components = pca.fit_transform(scaled_data)

        # Extract explained variance
        explained_variance = pca.explained_variance_ratio_

        # Calculate first and second differences
        diff = np.diff(explained_variance,2)
        diff2 = np.diff(diff)

        # Choose number of ensembles by elbow method
        k = np.argmax(diff2) + 2


    else:
        # Combine wscc and silhouette
        Z = linkage(co_mat, 'ward')
        last = Z[-num_lat_init:-1, 2]
        last_rev = last[::-1]
        range_n_clusters = np.arange(2, num_lat_init)

        acceleration =  np.diff(last, 2) - np.diff(last,1)[:-1] # 2nd derivative of the distances
        acceleration_rev = acceleration[::-1]
        acceleration_rev = acceleration_rev/np.linalg.norm(acceleration_rev)

        range_n_clusters = np.arange(2, num_lat_init+1)

        for n_clusters in range_n_clusters:
            # Form flat clusters from the hierarchical clustering
            cluster_labels = fcluster(Z, n_clusters, criterion='maxclust')
            
            # Compute the silhouette score
            silhouette_avg.append(silhouette_score(co_mat, cluster_labels))

        acceleration_rev = acceleration_rev/np.linalg.norm(acceleration_rev)

        # Adjusting the range of clusters to match the length of last_rev
        range_n_clusters = np.arange(1, 1 + len(last_rev))

        acceleration = np.diff(silhouette_avg, 2) + np.diff(silhouette_avg, 1)[:-1]
        acceleration = acceleration/np.linalg.norm(acceleration)

        acceleration = (acceleration_rev - acceleration)/2

        k = acceleration.argmax() + 2
    
    num_latent_list.append(k)

    ## Find best clustering
    best_clusters = fcluster(Z, t=k, criterion="maxclust")
    consensus_clustering_ami = adjusted_mutual_info_score(ensemble_assignments_true, best_clusters)
    consensus_ami_list.append(consensus_clustering_ami)

    ami_pred_score_list = []
    optimal_cluster_list = []
    state_assignment_dict = {}
    num_retrains=50
    n_components=15
    dropout_rate=0.5
    state_ami_threshold=0.3
    num_latent=10

    unique_clusters = np.unique(best_clusters)
    print("unique clusters:", unique_clusters)
    for cluster_nr in unique_clusters:
        sub_data = data[:, (best_clusters == cluster_nr)]
        sub_data_test = data_test[:, (best_clusters == cluster_nr)]
        sub_data_full = np.concatenate([sub_data, sub_data_test], axis=0)
        
        pred_co_mat = None
        tot_features = sub_data.shape[1]
        num_features = int(np.round(tot_features * dropout_rate))
        
        for n in range(num_retrains):
            sampled_indices = np.random.choice(tot_features, size=num_features, replace=False)
            boot_data = sub_data_full[:, sampled_indices]

            model = hmm.PoissonHMM(n_components=n_components, n_iter=10)
            try:
                model.fit(boot_data)
                pred = model.predict(boot_data)
            except:
                print("Error in prediction for cluster nr:", cluster_nr, "at retrain", n)
                continue

            if pred_co_mat is None:
                pred_co_mat = create_cooccurrence_matrix(pred)
            else:
                pred_co_mat += create_cooccurrence_matrix(pred)
        
        Z = linkage(pred_co_mat, 'ward')
        range_n_clusters = range(2, num_latent + 1)

        silhouette_avg = [silhouette_score(pred_co_mat, fcluster(Z, n, criterion='maxclust')) for n in range_n_clusters]
        
        optimal_clusters = range_n_clusters[np.argmax(silhouette_avg)]
        optimal_cluster_list.append(optimal_clusters)

        state_assignment = fcluster(Z, t=optimal_clusters, criterion="maxclust")
        state_assignment_dict[cluster_nr] = state_assignment
    
    used_map = {}
    for cluster_nr, state_assignment in state_assignment_dict.items():
        for second_cluster_nr, second_state_assignment in state_assignment_dict.items():
            if cluster_nr == second_cluster_nr or used_map.get(cluster_nr) == second_cluster_nr:
                continue
            
            ami_score = adjusted_mutual_info_score(state_assignment, second_state_assignment)
            print("ami_score:", ami_score)
            if ami_score > state_ami_threshold:
                used_map[second_cluster_nr] = cluster_nr
                best_clusters[best_clusters == second_cluster_nr] = cluster_nr
    
    
    
    consensus_clustering_ami = adjusted_mutual_info_score(ensemble_assignments_true, best_clusters)
    ami_ensemble_score_list.append(consensus_clustering_ami)
    ## Map best clustering for easier evaluation. Does not improve performance!
    best_clusters_mapped, mapping = best_mapping_new(ensemble_assignments_true, best_clusters)
    print(mapping)
    print(state_assignment_dict.keys())
    # Remap the state assignment dictionary  the mapping of best clusters to true assignment
    # If a value has been mapped, use create a new key with that value
    state_assignment_dict_mapped = {mapping.get(cluster_nr, cluster_nr): states for cluster_nr, states in state_assignment_dict.items()} 
    print(state_assignment_dict_mapped.keys())
    for cluster_nr in np.unique(best_clusters_mapped):
        try:
            state_assignment = state_assignment_dict_mapped.get(cluster_nr)
            ami_score = adjusted_mutual_info_score(true_states[cluster_nr,:], state_assignment)
            ami_pred_score_list.append(ami_score)
        except:
            # If there is too many clusters, return 0
            print(f"Error with key {cluster_nr}")
            ami_pred_score_list.append(0)
    
    sub_path = f"test_num_{num_prev_test+j+1}_n{ensemble_sizes}_s{ensemble_states}"

    ami_score_list_path = f"{folder}/{sub_path}/ami_score_list.csv"
    optimal_cluster_list_path = f"{folder}/{sub_path}/optimal_cluster_list.csv"
    
    ami_score_mean = np.mean(ami_pred_score_list)
    test_info_path = f"{folder}/{sub_path}/test_info.txt"

    with open(run_info_path, 'a') as f:
        f.write(f"Average State AMI: {ami_score_mean}\n")
        f.write(f"Number of latents: {k}\n")
        f.close()   

    # AMI score of each prediction
    np.savetxt(ami_score_list_path , ami_pred_score_list, delimiter = ",")
    
    # Optimal number of clusters after each iteration
    np.savetxt(optimal_cluster_list_path, optimal_cluster_list, delimiter = ",")

    # Final AMI clustering score after merging og ensembles
    ensemble_ami_list_path = f"{folder}/ensemble_ami_list.csv"
    np.savetxt(ensemble_ami_list_path, ami_ensemble_score_list, delimiter = ",")

    # AMI score after consensus clustering of mean AMI matrix
    consensus_ami_list_path = f"{folder}/consensus_ami_list.csv"
    np.savetxt(consensus_ami_list_path, consensus_ami_list, delimiter = ",")

    # Number of latent estimated when clustering mean AMI matrix
    num_latent_list_path = f"{folder}/num_latent_list.csv"
    np.savetxt(num_latent_list_path, num_latent_list, delimiter = ",")
    
    # best_clustering after merging of ensembles
    best_clusters_path = f"{folder}/{sub_path}/best_clusters.csv"
    np.savetxt(best_clusters_path, best_clusters, delimiter = ",")






