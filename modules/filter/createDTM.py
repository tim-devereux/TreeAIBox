import os
import numpy as np
import numpy_groupies as npg
from skimage.filters import median
from skimage.morphology import disk
from scipy.interpolate import griddata

def createDtm(pcd,resolution=np.array([1.0,1.0]),tile_size=np.array([100,100]),buffer_size=np.array([20,20])):#last dimension of columns
    columns = pcd
    if pcd.shape[1] > 3:
        columns = columns[pcd[:, -1] <=1]

    columns_max = np.max(columns[:, :3], axis=0)
    columns_min = np.min(columns[:, :3], axis=0)

    if tile_size is not None:
        dtm_patches=[]
        # Calculate number of tiles in each direction
        xy_tiles = np.ceil((columns_max[:2] - columns_min[:2]) / tile_size[:2]).astype(np.int32)
        for i in range(xy_tiles[0]):
            for j in range(xy_tiles[1]):
                # Bounding box corners for the tile
                x1 = columns_min[0] + i * tile_size[0] - buffer_size[0]
                x2 = columns_min[0] + (i + 1) * tile_size[0] + buffer_size[0]
                y1 = columns_min[1] + j * tile_size[1] - buffer_size[1]
                y2 = columns_min[1] + (j + 1) * tile_size[1] + buffer_size[1]


                pcd_tile = columns[(columns[:, 0] >= x1) & (columns[:, 0] <= x2) & (columns[:, 1] >= y1) & (columns[:, 1] <= y2)]
                if len(pcd_tile)>0:
                    pcd_tile_max = np.max(pcd_tile[:, :3], axis=0)
                    pcd_tile_min = np.min(pcd_tile[:, :3], axis=0)

                    pcd_tile_shape = (np.floor((pcd_tile_max[:2]-pcd_tile_min[:2]) / resolution[:2]) + 1).astype(np.int32)
                    pcd_tile_shape = pcd_tile_shape[[1, 0]]
                    pcd_tile_sz = pcd_tile_shape[0] * pcd_tile_shape[1]

                    pcd_tile_ij = np.floor(np.concatenate([(pcd_tile[:, 1] - pcd_tile_min[1])[:, np.newaxis], (pcd_tile[:, 0] - pcd_tile_min[0])[:, np.newaxis]], axis=1) / resolution[[1, 0]]).astype(np.int32)
                    pcd_tile_dtm_idx = np.ravel_multi_index((np.transpose(pcd_tile_ij)).astype(np.int32), pcd_tile_shape)
                    pcd_tile_dtm_idx_u, pcd_tile_dtm_idx_uidx, pcd_tile_dtm_inverse_idx = np.unique(pcd_tile_dtm_idx, return_index=True,return_inverse=True)

                    pcd_tile_z_min = npg.aggregate(pcd_tile_dtm_inverse_idx, pcd_tile[:,2], func='min', fill_value=np.nan)
                    pcd_tile_dtm = np.full(pcd_tile_sz, np.inf)
                    pcd_tile_dtm[pcd_tile_dtm_idx_u] = pcd_tile_z_min

                    pcd_tile_dtm = pcd_tile_dtm.reshape(pcd_tile_shape)
                    dtm_mask = np.isinf(pcd_tile_dtm)
                    dtm_coord = np.argwhere(~dtm_mask)
                    values = pcd_tile_dtm[~dtm_mask]

                    grid_indices = np.argwhere(dtm_mask)
                    dtm_filled = griddata(dtm_coord, values, grid_indices, method='cubic')
                    pcd_tile_dtm[grid_indices[:, 0], grid_indices[:, 1]] = dtm_filled
                    pcd_tile_dtm = median(pcd_tile_dtm, disk(5))
                    # plt.imshow(pcd_abg_dtm)
                    # plt.show()
                    dtm_y, dtm_x = np.meshgrid(np.arange(pcd_tile_shape[1]), np.arange(pcd_tile_shape[0]), indexing='ij')
                    dtm_xy = np.vstack([dtm_y.ravel(), dtm_x.ravel()]).T
                    dtm_value = pcd_tile_dtm.reshape(pcd_tile_sz, order='F')
                    dtm_xy = pcd_tile_min[:2] + resolution[:2] * dtm_xy
                    dtm_xyz = np.concatenate([dtm_xy, dtm_value[:, np.newaxis]], axis=-1)
                    dtm_xyz = dtm_xyz[~np.isnan(dtm_xyz[:, -1])]
                    dtm_xyz = dtm_xyz[~np.isinf(dtm_xyz[:, -1])]
                    dtm_patches.append(dtm_xyz)

        dtm_patches=np.concatenate(dtm_patches)
        dtm_patches_max=np.max(dtm_patches[:,:3],axis=0)
        dtm_patches_min=np.min(dtm_patches[:,:3],axis=0)
        dtm_patches_ij=np.floor(np.concatenate([(dtm_patches[:, 1] - columns_min[1])[:, np.newaxis], (dtm_patches[:, 0] - columns_min[0])[:, np.newaxis]],axis=1) / resolution[[1, 0]]).astype(np.int32)
        dtm_patches_shape = (np.floor((dtm_patches_max[:2] - dtm_patches_min[:2]) / resolution[:2]) + 1).astype(np.int32)
        dtm_patches_shape = dtm_patches_shape[[1, 0]]
        dtm_patches_idx = np.ravel_multi_index((np.transpose(dtm_patches_ij)).astype(np.int32), dtm_patches_shape)
        dtm_patches_u, dtm_patches_uidx, dtm_patches_inverse_idx = np.unique(dtm_patches_idx,return_index=True,return_inverse=True)
        dtm_patches_z=npg.aggregate(dtm_patches_inverse_idx, dtm_patches[:,2], func='mean', fill_value=np.nan)
        dtm_patches_xy=dtm_patches[dtm_patches_uidx,:2]
        dtm=np.concatenate([dtm_patches_xy,dtm_patches_z[:,np.newaxis]],axis=1)
    else:
        columns_shape = (np.floor((columns_max[:2] - columns_min[:2]) / resolution[:2]) + 1).astype(np.int32)
        columns_shape = columns_shape[[1, 0]]
        columns_sz = columns_shape[0] * columns_shape[1]
        columns_ij = np.floor(np.concatenate([(columns[:, 1] - columns_min[1])[:, np.newaxis], (columns[:, 0] - columns_min[0])[:, np.newaxis]],axis=1) / resolution[[1, 0]]).astype(np.int32)
        columns_dtm_idx = np.ravel_multi_index((np.transpose(columns_ij)).astype(np.int32), columns_shape)
        columns_dtm_idx_u, columns_dtm_idx_uidx, columns_dtm_inverse_idx = np.unique(columns_dtm_idx,return_index=True, return_inverse=True)
        columns_z_min = npg.aggregate(columns_dtm_inverse_idx, columns[:, 2], func='min', fill_value=np.nan)
        columns_dtm = np.full(columns_sz, np.inf)
        columns_dtm[columns_dtm_idx_u] = columns_z_min
        columns_dtm = columns_dtm.reshape(columns_shape)
        dtm_mask = np.isinf(columns_dtm)
        dtm_coord = np.argwhere(~dtm_mask)
        values = columns_dtm[~dtm_mask]
        grid_indices = np.argwhere(dtm_mask)
        dtm_filled = griddata(dtm_coord, values, grid_indices, method='cubic')
        columns_dtm[grid_indices[:, 0], grid_indices[:, 1]] = dtm_filled
        columns_dtm = median(columns_dtm, disk(5))
        dtm_y, dtm_x = np.meshgrid(np.arange(columns_shape[1]), np.arange(columns_shape[0]), indexing='ij')
        dtm_xy = np.vstack([dtm_y.ravel(), dtm_x.ravel()]).T
        dtm_value = columns_dtm.reshape(columns_sz, order='F')
        dtm_xy = columns_min[:2] + resolution[:2] * dtm_xy
        dtm_xyz = np.concatenate([dtm_xy, dtm_value[:, np.newaxis]], axis=-1)
        dtm_xyz = dtm_xyz[~np.isnan(dtm_xyz[:, -1])]
        dtm = dtm_xyz[~np.isinf(dtm_xyz[:, -1])]
        # print('done!')

    return dtm


if __name__ == '__main__':
    import laspy

    # path_to_las = r"D:\xzx\prj\test_data\tls\Mixed_plot1_sep.laz"
    path_to_las = r"D:\xzx\prj\test_data\als\pred_JNP23_S_478000_5792000_T317.laz"
    pcd_basename = os.path.basename(path_to_las)[:-4]
    las = laspy.read(path_to_las)
    # columns=np.transpose([las.x,las.y,las.z,las.trees])
    columns=np.transpose([las.x,las.y,las.z,las.Pred-1])

    # dtm_xyz=create_dtm(columns,resolution=np.array([1.0,1.0]))
    dtm_xyz=createDtm(columns,resolution=np.array([1.0,1.0]),tile_size=None,buffer_size=None)
    np.savetxt('tmp.txt',dtm_xyz)
