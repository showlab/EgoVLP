import numpy as np


def calculate_mAP(sim_mat, relevancy_matrix):
    """
    Computes the mean average precision according to the following formula of
    average precision:
    \frac{\sum_{k=1}^n p(k) x rel(k)}{num_rel_docs}

    where p(k) is the precision at k, rel(k) is an indicator function
    determining whether the kth returned item is relevant or not and
    num_rel_docs is the number of relevant items to find within the search.

    The mean average precision is the mean of the average precision for each
    query item (i.e row in the matrix)

    This function takes in two parameters:
        - sim_mat: a NxM matrix which represents the similarity between two
        modalities (with modality 1 being of size N and modality 2 of size M).
        - relevancy_matrix: an NxM matrix which represents the relevancy between two
        modalities of items (with modality 1 being of size N and modality 2 of
        size M).
    """
    #Find the order of the items in modality 2 according to modality 1
    ranked_order = (-sim_mat).argsort()
    ranked_sim_mat = sim_mat[np.arange(sim_mat.shape[0])[:, None], ranked_order]
    #re-order the relevancy matrix to accommodate the proposals
    ranked_rel_mat = relevancy_matrix[np.arange(relevancy_matrix.shape[0])[:, None], ranked_order]

    #find the number of relevant items found at each k
    cumulative_rel_mat = np.cumsum(ranked_rel_mat, axis=1)
    #Mask this ensuring that it is non zero if the kth term is 1 (rel(k) above)
    cumulative_rel_mat[ranked_rel_mat != 1] = 0
    #find the divisor for p(k)
    divisor = np.arange(ranked_rel_mat.shape[1]) + 1

    #find the number of relevant docs per query item
    number_rel_docs = np.sum(ranked_rel_mat==1, axis=1)

    #find the average precision per query, within np.sum finds p(k) * rel(k)
    avg_precision = np.sum(cumulative_rel_mat / divisor, axis=1) / number_rel_docs
    mAP = np.mean(avg_precision)
    return mAP
