import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

def create_sparse_graph(points, k=10, max_distance=np.inf):
    """
    Create a sparse graph based on k-nearest neighbors within max_distance.

    :param points: numpy array of shape (n_points, n_dimensions)
    :param k: number of nearest neighbors to consider
    :param max_distance: maximum distance for connections
    :return: scipy.sparse.csr_matrix representing the graph
    """
    kdtree = cKDTree(points)
    distances, indices = kdtree.query(points, k=k + 1, distance_upper_bound=max_distance)

    # Remove self-connections (first column)
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    # Create row indices
    row_ind = np.repeat(np.arange(len(points)), k)

    # Flatten column indices and distances
    col_ind = indices.ravel()
    data = distances.ravel()

    # Remove any remaining infinite distances
    valid = np.isfinite(data)
    row_ind = row_ind[valid]
    col_ind = col_ind[valid]
    data = data[valid]

    graph = csr_matrix((data, (row_ind, col_ind)), shape=(len(points), len(points)))
    return (graph + graph.T) / 2  # Make symmetric


def applySmallClusterClean(pcd,max_gap_clean=0.0,min_points_clean=0,MIN_RES = 0.03):
    # assert(max_gap_clean>0)
    # assert(min_points_clean>0)

    pcd_ijk, pcd_ijk_reverse_index = np.unique(np.floor((pcd[:, :3] - np.min(pcd[:, :3], axis=0)) / MIN_RES), axis=0,return_inverse=True)
    csgraph = create_sparse_graph(pcd_ijk, k=10, max_distance=max_gap_clean)
    n_components, labels = connected_components(csgraph, connection='strong', return_labels=True)
    labels_unq, labels_unq_reverse_index, labels_counts = np.unique(labels, return_inverse=True, return_counts=True)

    labels_reverse_counts = labels_counts[labels_unq_reverse_index]
    labels_reverse_counts[labels_reverse_counts < min_points_clean] = 0

    return labels_reverse_counts[pcd_ijk_reverse_index]


if __name__ =='__main__':
    import os
    import laspy

    max_gap_clean=3.0
    min_points_clean=100
    MIN_RES = 0.03

    path_to_las = r"C:\Users\ZXI\Downloads\2023_NWT-10-7_REG-V2_SUB-2_SOR-6-3_ref_pred_stemcls.las"
    pcd_basename = os.path.basename(path_to_las)[:-4]
    las = laspy.read(path_to_las)
    pcd=np.transpose([las.x,las.y,las.z,las.stemcls])
    stem_ind=pcd[:,-1]==2
    pcd=pcd[stem_ind]

    pcd_ijk, pcd_ijk_reverse_index = np.unique(np.floor((pcd[:, :3] - np.min(pcd[:, :3], axis=0)) / MIN_RES), axis=0, return_inverse=True)
    csgraph = create_sparse_graph(pcd_ijk, k=10, max_distance=max_gap_clean)
    n_components, labels = connected_components(csgraph, connection='strong', return_labels=True)
    labels_unq, labels_unq_reverse_index,labels_counts = np.unique(labels, return_inverse=True, return_counts=True)

    labels_reverse_counts = labels_counts[labels_unq_reverse_index]
    labels_reverse_counts[labels_reverse_counts < min_points_clean] = 0

    np.savetxt(r"C:\Users\ZXI\Downloads\tmp.txt",np.concatenate([pcd,labels_reverse_counts[pcd_ijk_reverse_index,np.newaxis]],axis=-1))
