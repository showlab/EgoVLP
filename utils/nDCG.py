import numpy as np

def calculate_DCG(similarity_matrix, relevancy_matrix, k_counts):
    """
    Calculates the Discounted Cumulative Gain (DCG) between two modalities for
    the first modality.
    DCG = \sum_{i=1}^k \frac{rel_i}{log_2(i + 1)}
    i.e. the sum of the k relevant retrievals which is calculated as the scaled
    relevancy for the ith item. The scale is designed such that early
    retrievals are more important than later retrievals.
    Params:
        - similarity_matrix: matrix of size n1 x n2 where n1 is the number of
          items in the first modality and n2 is the number of items in the
          second modality. The [ith,jth] element is the predicted similarity
          between the ith item from the first modality and the jth item from
          the second modality.
        - relevancy_matrix: matrix of size n1 x n2 (see similarity_matrix
          above). The [ith, jth] element is the semantic relevancy between the
          ith item from the first modality and the jth item from the second
          modality.
        - k_counts: matrix of size n1 x n2 (see similarity_matrix above) which
          includes information on which items to use to calculate the DCG for
          (see calculate_k_counts for more info on this matrix).
    Returns:
        - The DCG for each item in the first modality, a n1 length vector.
    """
    x_sz, y_sz = similarity_matrix.shape
    ranks = np.argsort(similarity_matrix)[:, ::-1]
    #Create vector of size (n,) where n is the length of the last dimension in
    #similarity matrix
    #This vector is of the form log(i+1)
    logs = np.log2(np.arange(y_sz) + 2)
    #Convert logs into the divisor for the DCG calculation, of size similarity
    #matrix
    divisors = np.repeat(np.expand_dims(logs, axis=0), x_sz, axis=0)

    #mask out the sorted relevancy matrix to only use the first k relevant
    #retrievals for each item.
    columns = np.repeat(np.expand_dims(np.arange(x_sz), axis=1), y_sz, axis=1)
    numerators = relevancy_matrix[columns, ranks] * k_counts
    #Calculate the final DCG score (note that this isn't expected to sum to 1)
    return np.sum(numerators / divisors, axis=1)

def calculate_k_counts(relevancy_matrix):
    """
    Works out the maximum number of allowed retrievals when working out the
    Discounted Cumulative Gain. For each query the DCG only uses the first k
    items retrieved which constitute the k relevant items for that query
    (otherwise the nDCG scores can be deceptively high for bad rankings).
    Params:
        - relevancy_matrix: matrix of size n1 x n2 where n1 is the number of
          items in the first modality and n2 is the number of items in the
          second modality.  The [ith, jth] element is the semantic relevancy
          between the ith item from the first modality and the jth item from
          the second modality.
    Returns:
        - Matrix of size n1 x n2 (see relevancy matrix for more info). This is
          created as a mask such that if the [ith, jth] element is 1 it
          represents a valid item to use for the calculation of DCG for the
          ith item after sorting. For example, if relevancy matrix of:
        [[1, 0.5, 0],
          [0, 0  , 1]]
          is given, then the k_counts matrix will be:
        [[1, 1, 0],
         [1, 0, 0]]
         i.e. the first row has 2 non-zero items, so the first two retrieved
         items should be used in the calculation. In the second row there is
         only 1 relevant item, therefore only the first retrieved item should
         be used for the DCG calculation.
    """
    return (np.sort(relevancy_matrix)[:, ::-1] > 0).astype(int)


def calculate_IDCG(relevancy_matrix, k_counts):
    """
    Calculates the Ideal Discounted Cumulative Gain (IDCG) which is the value
    of the Discounted Cumulative Gain (DCG) for a perfect retrieval, i.e. the
    items in the second modality were retrieved in order of their descending
    relevancy.
    Params:
        - relevancy_matrix: matrix of size n1 x n2 where n1 is the number of
          items in the first modality and n2 is the number of items in the
          second modality. The [ith, jth] element is the semantic relevancy
          between the ith item from the first modality and the jth item from
          the second modality.
        - k_counts: matrix of size n1 x n2 (see similarity_matrix above) which
          includes information on which items to use to calculate the DCG for
          (see calculate_k_counts for more info on this matrix).
    """
    return calculate_DCG(relevancy_matrix, relevancy_matrix, k_counts)

def calculate_nDCG(similarity_matrix, relevancy_matrix, k_counts=None, IDCG=None, reduction='mean'):
    """
    Calculates the normalised Discounted Cumulative Gain (nDCG) between two
    modalities for the first modality using the Discounted Cumulative Gain
    (DCG) and the Ideal Discounted Cumulative Gain (IDCG).

    nDCG = \frac{DCG}{IDCG}

    Params:
        - similarity_matrix: matrix of size n1 x n2 where n1 is the number of
          items in the first modality and n2 is the number of items in the second
          modality. The [ith,jth] element is the predicted similarity between
          the ith item from the first modality and the jth item from the second
          modality.
        - relevancy_matrix: matrix of size n1 x n2 (see similarity_matrix
          above). The [ith, jth] element is the semantic relevancy between the
          ith item from the first modality and the jth item from the second
          modality.
        - k_counts: optional parameter: matrix of size n1 x n2 (see
          similarity_matrix above) which includes information on which items to
          use to calculate the DCG for (see calculate_k_counts for more info on
          this matrix). This will be calculated using calculate_IDCG if not
          present, but should be pre-processed for efficiency.
        - IDCG: Optional parameter which includes the pre-processed Ideal
          Discounted Cumulative Gain (IDCG). This is a vector of size n1 (see
          similarity_matrix above) which contains the IDCG value for each item
          from the first modality. This will be calculated using calculate_IDCG
          if not present, but should be pre-processed for efficiency.
        - reduction: what to use to reduce the different nDCG scores. By
          default this applies np.mean across all different queries.
    Returns:
        - The nDCG values for the first modality.
    """
    if k_counts is None:
        k_counts = calculate_k_counts(relevancy_matrix)
    DCG = calculate_DCG(similarity_matrix, relevancy_matrix, k_counts)
    if IDCG is None:
        IDCG = calculate_IDCG(relevancy_matrix, k_counts)
    if reduction == 'mean':
        return np.mean(DCG / IDCG)
    elif reduction is None:
        return DCG / IDCG


if __name__ == '__main__':
    sim_matrix = np.array([
        [1.0, 0.7, 0.4, 0.0],
        [0.3, 0.9, 0.6, 0.1],
        [0.2, 0.5, 0.8, 0.4]
    ])
    rel_matrix = np.array([
        [1.0, 0.5, 0.25, 0.0],
        [0.0, 1.0, 0.4, 0.0],
        [0.5, 0.3, 1.0, 0.0]
    ])
    k_counts = np.array([
        [1, 1, 1, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0]
    ])
    assert (k_counts == calculate_k_counts(rel_matrix)).all()
    nDCG = calculate_nDCG(sim_matrix, rel_matrix, k_counts)
    assert nDCG == 0.9371789900735429
    DCG = calculate_DCG(sim_matrix, rel_matrix, k_counts)
    IDCG = calculate_IDCG(rel_matrix, k_counts)

    assert nDCG == np.mean(DCG / IDCG)

    pre_nDCG = calculate_nDCG(sim_matrix, rel_matrix, k_counts, IDCG=IDCG)
    assert pre_nDCG == nDCG

    post_mean_nDCG = calculate_nDCG(sim_matrix, rel_matrix, k_counts, IDCG=IDCG, reduction=None)
    assert np.mean(post_mean_nDCG) == pre_nDCG
