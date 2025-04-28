import numpy as np
import os
import laspy
import glob
import numpy_indexed as npi
from scipy.spatial import cKDTree

inlier_min_R=0.4
inlier_min_K=15

if __name__ == '__main__':

    pcd_dir=r"..\data"
    out_dir=r"..\data"

    pcd_names=[
        # "tree_farm_area1_sub_refined_itc",
        "GP_262_leafon_itc1"
    ]

    for pcd_name in pcd_names:
        las = laspy.open(glob.glob(os.path.join(pcd_dir, pcd_name + '.la*'))[0]).read()
        columns = np.transpose(np.array([las.x, las.y, las.z, las.itc_edit]))  # las.point_format.dimension_names
        seg_tops = np.zeros([len(columns),2])

        abg_ind=columns[:,-1]>0
        columns=columns[abg_ind]

        tree = cKDTree(columns[:, :3])
        indices = tree.query_ball_point(columns[:, :3], r=inlier_min_R,p=2)
        neighbor_counts = np.array([len(idx) for idx in indices])
        inlier_logi=neighbor_counts>inlier_min_K

        itc_unq, itc_idx_groups = npi.group_by(columns[:, -1], np.arange(len(columns[:, -1])))
        seg_peak_poses = np.zeros([len(itc_unq),4])
        seg_confidences = np.zeros(len(columns))
        seg_radii = np.zeros(len(columns))

        for i,itc_idx in enumerate(itc_idx_groups):
            segs_pos = columns[itc_idx, :]
            n_segs_pos=len(segs_pos)

            inlier_mask=inlier_logi[itc_idx]

            segs_pos=segs_pos[inlier_mask]
            if len(segs_pos)>0:
                peak_i=np.argmax(segs_pos[:,2])
                segs_pos_hmin=np.min(segs_pos[:,2],axis=0)
                segs_pos_hmax=segs_pos[peak_i,2]

                seg_radius=np.mean(np.max(segs_pos[:,:2],axis=0)-np.min(segs_pos[:,:2],axis=0))*0.5
                seg_peak_poses[i,:]=np.append(segs_pos[peak_i,:3],seg_radius)

                segs_pos_dists_2d=np.sqrt(np.sum(np.power(segs_pos[:,:2]-segs_pos[peak_i,:2],2),1))
                segs_pos_dists_h=(segs_pos[:,2]-segs_pos_hmin)/(segs_pos_hmax-segs_pos_hmin)

                sigma = np.mean(segs_pos_dists_2d) * 0.5 # adjustable parameter for smoother/faster gradients
                S = np.exp(-segs_pos_dists_2d ** 2 / (2 * sigma ** 2))

                seg_top_metric=np.zeros(n_segs_pos)
                seg_top_metric[inlier_mask]=segs_pos_dists_h * S
                seg_confidences[itc_idx] = seg_top_metric
                seg_radii[itc_idx] = seg_radius*np.ones(n_segs_pos)
        seg_peak_poses=seg_peak_poses[np.sum(seg_peak_poses[:,:3],axis=1)>0.01]
        outfname = os.path.join(out_dir, f"{pcd_name}_treeloc.csv")
        np.savetxt(outfname, seg_peak_poses, fmt='%.3f')
        print(f"converted {pcd_name}")

        # seg_tops[abg_ind,0]=seg_confidences
        # seg_tops[abg_ind,1]=seg_radii

        # fname = glob.glob(os.path.join(pcd_dir, pcd_name + '.la*'))[0]
        # las = laspy.open(fname).read()
        # las.add_extra_dim(laspy.ExtraBytesParams(name="top_conf", type="float32", description="top_conf"))
        # las.add_extra_dim(laspy.ExtraBytesParams(name="top_radius", type="float32", description="top_radius"))
        # las.top_conf = seg_tops[:,0]
        # las.top_radius = seg_tops[:,1]
        # las.write(os.path.join(pcd_dir, "{}_treetop.laz".format(pcd_name)))

