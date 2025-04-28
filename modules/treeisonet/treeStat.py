import numpy as np
import numpy_indexed as npi
import os
from scipy.spatial import cKDTree
from scipy.spatial import ConvexHull
import numpy_groupies as npg
from scipy.interpolate import griddata

def createDtm(columns,grd_class=None, dtm_resolution=np.array([1.0, 1.0])):#grd_mask: grd(<=1), veg(>1)
    def rasterize(points, res, shape, extent, func='max'):
        points_i = extent[3] - points[:, 1]
        points_j = points[:, 0] - extent[0]

        points_ij = np.floor(np.column_stack([points_i, points_j]) / res[[1, 0]]).astype(np.int32)
        valid_indices = (points_ij[:, 0] >= 0) & (points_ij[:, 0] < shape[0]) & \
                        (points_ij[:, 1] >= 0) & (points_ij[:, 1] < shape[1])

        points_ij = points_ij[valid_indices]
        values = points[valid_indices, 2]

        points_idx = np.ravel_multi_index((points_ij[:, 0], points_ij[:, 1]), shape)

        raster_values = npg.aggregate(points_idx, values, func=func, size=shape[0] * shape[1], fill_value=np.nan)
        return raster_values.reshape(shape)

    # Calculate overall extent and shape
    columns_min = np.min(columns[:, :2], axis=0)
    columns_max = np.max(columns[:, :2], axis=0)
    dtm_shape = np.ceil((columns_max - columns_min) / dtm_resolution).astype(int)
    dtm_shape = dtm_shape[::-1]  # Swap to (rows, cols)
    extent = (*columns_min, *columns_max)

    # Rasterize DTM
    if grd_class is not None:
        dtm = rasterize(columns[grd_class <= 1], dtm_resolution, dtm_shape, extent, func='min')
    else:
        dtm = rasterize(columns, dtm_resolution, dtm_shape, extent, func='min')

    # Interpolate missing values
    mask = np.isnan(dtm)
    coords = np.argwhere(~mask)
    values = dtm[~mask]
    dtm[mask] = griddata(coords, values, np.argwhere(mask), method='cubic', fill_value=np.nan)

    # Create scattered points for CHM
    i, j = np.indices(dtm_shape)
    scattered_dtm = np.column_stack((
        columns_min[0] + (j+0.5).ravel() * dtm_resolution[0],
        columns_max[1] - (i+0.5).ravel() * dtm_resolution[1],
        dtm.ravel()
    ))

    # Remove points with zero height
    scattered_dtm = scattered_dtm[~ np.isnan(scattered_dtm[:, 2])]
    return scattered_dtm


def treeStat(pcd,treeitc,pcd_min=None,treefilter=None,outpath=None,dtm_resolution=1.0,progress_callback=lambda x: None):
    progress_callback(0)

    columns = pcd.copy()#otherwise will be problems
    columns_itc=treeitc.copy()#otherwise will be problems

    # pcd_min=np.min(pcd[:,:3],axis=0)
    # columns[:, :3]=columns[:,:3]-pcd_min
    # print(pcd_min)
    if pcd_min is None:
        pcd_min=np.min(pcd[:,:3],axis=0)
        columns[:, :3] = columns[:, :3] - pcd_min


    if treefilter is not None:
        dtm = createDtm(columns, treefilter,np.array([dtm_resolution,dtm_resolution]))
        abg_ind=treefilter>1
        columns=columns[abg_ind,:]
        columns_itc=treeitc[abg_ind]
    else:
        dtm = createDtm(columns)

    progress_callback(15)

    kdtree=cKDTree(dtm[:,:2])
    _, indices = kdtree.query(columns[:,:2])
    dtm_z=dtm[indices, 2]

    columns=np.concatenate([columns,dtm_z[:,np.newaxis]],-1)
    itc_unq, itc_idx_groups = npi.group_by(columns_itc, np.arange(len(columns_itc)))
    itc_stats=np.zeros([len(itc_unq),6])#itc_id,x,y,z,height,crown_area
    for i,itc_idx in enumerate(itc_idx_groups):
        itc_pts = columns[itc_idx, :]
        itc_stats[i,0]=itc_unq[i]
        itc_stats[i,1:3] = np.mean(itc_pts[:,:2],axis=0)+pcd_min[:2]
        itc_stats[i,3] = np.max(itc_pts[:,2])+pcd_min[2]

        itc_stats[i,4]=np.max(itc_pts[:,2])-(np.min(itc_pts[:,2])+np.min(itc_pts[:,-1]))*0.5
        if itc_pts.shape[0]>6:
            itc_stats[i,5]=ConvexHull(itc_pts[:,:2]).volume

        progress_value = int(85 * i / len(itc_unq)) + 15
        progress_callback(progress_value)

    if outpath is not None:
        np.savetxt(outpath,itc_stats,delimiter=" ",fmt="%d %.3f %.3f %.3f %.3f %.3f",header="itc_id x y z height crown_area",comments="")

if __name__ == '__main__':
    import laspy

    # data_path=r"D:\xzx\prj\cc-TreeAIBox-plugin\data\stats"
    data_path=r"C:\CloudCompare\TreeAIBox\New folder"
    extract_list = [
        # "tree_farm_area1_sub_refined",
        "cynthia_site08_202410_grd",
    ]
    out_path=r"D:\xzx\prj\cc-TreeAIBox-plugin\modules\treeisonet\output"
    for las_fname in extract_list:

        print("Extracting stats for {}".format(las_fname))
        las = laspy.open(os.path.join(data_path, f"{las_fname}.laz")).read()
        columns = np.transpose(np.array([las.x, las.y, las.z]))
        treeStat(columns,las.treeoff,None,las.treefilter,out_path)