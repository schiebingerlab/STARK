import numpy as np
import pandas as pd
import sklearn
import sklearn.metrics
import sklearn.neighbors
import sklearn.pipeline
import matplotlib.pyplot as plt
import scanpy as sc
import seaborn as sns



def normalize_matrix(mymat, axis=1):
    
    return np.nan_to_num(mymat/mymat.sum(axis=axis, keepdims=True))


def get_log1pCPM_matrix(mymat, scaling_factor=1e+4, axis=0):
    
    mymat = scaling_factor * normalize_matrix(mymat, axis=axis)
    return np.log1p(mymat)


# Graph weights
    
def get_purely_spatial_markov_graph(spatial_locations, gph_rho_abs, gph_s2):


    pairwise_distances_sp = sklearn.metrics.pairwise_distances(spatial_locations, metric='euclidean')

    WAM = np.exp(-(pairwise_distances_sp/gph_s2)**2) * (pairwise_distances_sp < gph_rho_abs)

    return normalize_matrix(WAM, axis=1)


def get_markov_graph(spatial_locations, F_mat, gph_rho_abs, gph_s2, gph_s1):

    pairwise_distances_sp = sklearn.metrics.pairwise_distances(spatial_locations, metric='euclidean') # In physical space

    pairwise_distances_gex = sklearn.metrics.pairwise_distances(F_mat, metric='euclidean') # In gene expression space

    WAM = np.exp(-(pairwise_distances_gex/gph_s1)**2 - (pairwise_distances_sp/gph_s2)**2) * (pairwise_distances_sp < gph_rho_abs)

    return normalize_matrix(WAM, axis=1)

# Other utilities

def compute_distances_along_edges(data_mat, adjacency_mat, metric='euclidean'):
    
    m = data_mat.shape[0]
    all_dists_data = sklearn.metrics.pairwise_distances(data_mat, metric=metric)
    non_diagonals = (np.eye(m)==0)
    return all_dists_data[adjacency_mat&non_diagonals]


def get_average_neighbour_radius(intended_num_neighbours, training_locations, relative_buffer=0.1, dist_reltol=1e-5):
    # training_locations must be (m_training)-by-2
    # relative_buffer is about the tolerance in the average num of neighbours
    # dist_reltol is about the error in the radius value


    distances_mat = sklearn.metrics.pairwise_distances(training_locations, metric='euclidean')

    min_distance = np.min(distances_mat[distances_mat>0])
    max_distance = np.max(distances_mat)
    lb = min_distance
    # lb = min_distance/2 # Try this later?
    ub = max_distance/5
    
    success_flag = False
    while not success_flag:
        my_iterate = (lb + ub)/2
        num_neighbours = (distances_mat <= my_iterate).sum(axis=1) - 1
        ave_num_neighbours = num_neighbours.mean()
        if ((ub - lb)/my_iterate < dist_reltol):
            print('Bisection search converged:')
            print(f"Ave num neighbours = {ave_num_neighbours:.3f}")
            break
        
        if (ave_num_neighbours > (1+relative_buffer)*intended_num_neighbours): # iterate too large
            ub = my_iterate
        elif (ave_num_neighbours < (1-relative_buffer)*intended_num_neighbours): # iterate too small
            lb = my_iterate
        else:
            success_flag = True
            print('Success criterion achieved:')
            print(f"Ave num neighbours = {ave_num_neighbours:.3f}")

    print("--------------------------------------------------------------")
    return my_iterate
    

# Evaluation metrics

def relative_error_numpy(MatGT, MatRC):
    err = np.linalg.norm(MatRC-MatGT)/np.linalg.norm(MatGT)
    print("Relative error = " + str(err))
    return err

def compute_label_transfer_accuracy(adata_orig, adata_reference, denoised_matrix, classifier_pipe, 
                                    method_name='STARK', pretrained=False, compute_transferred_annotations_on_orig=False, 
                                    spot_size=2, plotting=False, fullfile_classification_plot=None, fullfile_pca_plot=None):

    
    all_labels_reference = adata_reference.obs['annotation'].unique().tolist()
    all_labels_orig = adata_orig.obs['annotation'].unique().tolist()

    all_labels = sorted(list(set(all_labels_reference) | set(all_labels_orig)))

    ######################### Train if needed

    if not pretrained:
        reference_features_mat = get_log1pCPM_matrix(adata_reference.X.toarray(), axis=1)
        reference_features_mat -= reference_features_mat.mean(axis=0, keepdims=True)
        reference_targets_vec = pd.Categorical(adata_reference.obs['annotation'].values, categories=all_labels)

        # Training
        classifier_pipe.fit(reference_features_mat, reference_targets_vec)


        predicted_vec_on_reference_set = pd.Categorical(classifier_pipe.predict(reference_features_mat), categories=all_labels)

        cls_report_reference_set = sklearn.metrics.classification_report(reference_targets_vec, predicted_vec_on_reference_set, labels=all_labels)
        cls_scores_reference_set = sklearn.metrics.precision_recall_fscore_support(reference_targets_vec, predicted_vec_on_reference_set, labels=all_labels, average='micro')
        classification_accuracy_reference_set = cls_scores_reference_set[0]
        # print("Classification accuracy on reference set: ", classification_accuracy_reference_set)

    else:
        print("Using pretrained classifier")
        cls_report_reference_set = None
        classification_accuracy_reference_set = None


    # Compute GT features (needed later for plotting)

    orig_features_mat = get_log1pCPM_matrix(adata_orig.X.toarray(), axis=1)
    orig_features_mat -= orig_features_mat.mean(axis=0, keepdims=True)

    ##################### Compute transferred annotations on the ground truth if required
    if compute_transferred_annotations_on_orig:
        # This refers to transferring annotations onto the GROUND TRUTH, and using these to test
        
        predicted_vec_on_orig = pd.Categorical(classifier_pipe.predict(orig_features_mat), categories=all_labels)
        adata_orig.obs["annotation_to_test"] = predicted_vec_on_orig

    else:
        # Use the original annotations of the ground truth
        adata_orig.obs["annotation_to_test"] = pd.Categorical(adata_orig.obs["annotation"], categories=all_labels)



    #################### Reclassificaton test (label transfer)
        
    current_features_mat = get_log1pCPM_matrix(denoised_matrix, axis=1)
    current_features_mat -= current_features_mat.mean(axis=0, keepdims=True)
    current_targets_vec = adata_orig.obs["annotation_to_test"].values # This is a series, and we extract an array

    predicted_vec_on_current_set = pd.Categorical(classifier_pipe.predict(current_features_mat), categories=all_labels)
    adata_orig.obs["annotation_estimate_"+method_name] = predicted_vec_on_current_set

    cls_report_current_set = sklearn.metrics.classification_report(current_targets_vec, predicted_vec_on_current_set, labels=all_labels)
    cls_scores_current_set = sklearn.metrics.precision_recall_fscore_support(current_targets_vec, predicted_vec_on_current_set, labels=all_labels, average='micro')
    classification_accuracy_current_set = cls_scores_current_set[0]
    print("Label transfer accuracy on denoised: ", classification_accuracy_current_set)
    
    ########################## Visualization

    pcs_orig = None
    pcs_estimate = None

    if plotting:

        # Log PCA space plot

        pcs_orig = classifier_pipe.named_steps['pca'].transform(orig_features_mat)
        pcs_estimate = classifier_pipe.named_steps['pca'].transform(current_features_mat)


        xlimmin = min(pcs_orig[:,0].min(), pcs_estimate[:,0].min())
        xlimmax = max(pcs_orig[:,0].max(), pcs_estimate[:,0].max())
        ylimmin = min(pcs_orig[:,1].min(), pcs_estimate[:,1].min())
        ylimmax = max(pcs_orig[:,1].max(), pcs_estimate[:,1].max())

        fh = plt.figure(figsize=(10,4))

        plt.subplot(1,2,1)
        ax1 = sns.scatterplot(x=pcs_orig[:,0], y=pcs_orig[:,1], hue=current_targets_vec, hue_order=all_labels, s=3, rasterized=True, legend=False)
        # sns.move_legend(ax1, "upper left", bbox_to_anchor=(1, 1))
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.xlim([xlimmin,xlimmax])
        plt.ylim([ylimmin, ylimmax])
        plt.title("GT")
        
        plt.subplot(1,2,2)
        ax2 = sns.scatterplot(x=pcs_estimate[:,0], y=pcs_estimate[:,1], hue=current_targets_vec, hue_order=all_labels, s=3, rasterized=True, legend=True)
        sns.move_legend(ax2, "upper left", bbox_to_anchor=(1, 1))
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.xlim([xlimmin,xlimmax])
        plt.ylim([ylimmin, ylimmax])
        plt.title("Estimate")

        fh.tight_layout()

        if fullfile_pca_plot is not None:
            plt.savefig(fullfile_pca_plot, bbox_inches="tight")


        # Classification plot
        sc.pl.spatial(adata_orig, color=['annotation_to_test', "annotation_estimate_"+method_name], spot_size=spot_size)

        if fullfile_classification_plot is not None:
            plt.savefig(fullfile_classification_plot, bbox_inches="tight")


    return {"classification_accuracy_estimate": classification_accuracy_current_set,
            "cls_report_estimate": cls_report_current_set,
            "pcs_orig": pcs_orig,
            "pcs_estimate": pcs_estimate,
            "classification_accuracy_reference_set": classification_accuracy_reference_set,
            "cls_report_reference_set": cls_report_reference_set}


def compute_kNN_overlap(adata_orig, denoised_matrix, k_val=50, num_pcs=30):

    reference_features_mat = get_log1pCPM_matrix(adata_orig.X.toarray(), axis=1)
    reference_features_mat -= reference_features_mat.mean(axis=0, keepdims=True)

    current_features_mat = get_log1pCPM_matrix(denoised_matrix, axis=1)
    current_features_mat -= current_features_mat.mean(axis=0, keepdims=True)

    pca_obj_orig = sklearn.decomposition.PCA(n_components=num_pcs, random_state=0)
    
    transformed_features_mat = pca_obj_orig.fit_transform(reference_features_mat)
    connectivity_mat_orig = sklearn.neighbors.kneighbors_graph(transformed_features_mat, n_neighbors=k_val)

    transformed_features_mat = pca_obj_orig.transform(current_features_mat)
    connectivity_mat_rec = sklearn.neighbors.kneighbors_graph(transformed_features_mat, n_neighbors=k_val)

    m_cells = connectivity_mat_orig.shape[0]
    
    return (connectivity_mat_orig.multiply(connectivity_mat_rec)).sum()/m_cells

