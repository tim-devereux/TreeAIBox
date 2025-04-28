import warnings
warnings.filterwarnings("ignore")

import numpy as np

import torch

import json
import numpy_indexed as npi
from scipy.spatial import cKDTree


# pip install scikit-learn laspy[lazrs] scipy numpy_indexed numpy_groupies commentjson timm scikit-image matplotlib einops plotly pandas
def mergeshift(points,treelocs,treeids):
    tree = cKDTree(treelocs[:,:2])
    d, idx = tree.query(points[:,:2])
    seg_labels=treeids[idx]
    return seg_labels

def treeOff(config_file, pcd, treeloc, model_path, use_cuda=True, progress_callback=lambda x: None):
    progress_callback(0)
    try:
        with open(config_file) as json_file:
            configs = json.load(json_file)
    except Exception as e:
        print(config_file)
        print("Cannot load config file:", e)
        return

    nbmat_sz = np.array(configs["model"]["voxel_number_in_block"])
    min_res = np.array(configs["model"]["voxel_resolution_in_meter"])

    try:
        from vox3DSegFormerRegression import Segformer
    except:
        from .vox3DSegFormerRegression import Segformer

    model = Segformer(
        block3d_size=nbmat_sz,
        in_chans=2,
        out_chans=2,
        patch_size=configs["model"]["patch_size"],
        decoder_dim=configs["model"]["decoder_dim"],
        embed_dims=configs["model"]["channel_dims"],
        num_heads=configs["model"]["num_heads"],
        mlp_ratios=configs["model"]["MLP_ratios"],
        qkv_bias=configs["model"]["qkv_bias"],
        depths=configs["model"]["depths"],
        sr_ratios=configs["model"]["SR_ratios"],
        drop_rate=0., drop_path_rate=0.,
    )
    print("Num params (SegFormer): ", sum(p.numel() for p in model.parameters()))

    device = "cuda" if use_cuda else "cpu"

    if use_cuda:
        model = model.cuda()
        state_dict = torch.load(model_path)
    else:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    model.max_accu = state_dict.get('max_accu', 0.0)
    if 'max_accu' in state_dict:
        state_dict.pop('max_accu')

    model.best_mIoU = state_dict.get('best_mIoU', 0.0)
    if 'best_mIoU' in state_dict:
        state_dict.pop('best_mIoU')

    model.load_state_dict(state_dict)

    model.eval()

    progress_callback(5)

    nb_tsz = int(nbmat_sz[0] * nbmat_sz[1] * nbmat_sz[2])
    nb_tsz2d = int(nbmat_sz[0] * nbmat_sz[1])

    treelocs=np.concatenate([treeloc, np.arange(len(treeloc))[:, np.newaxis]], axis=-1)

    pcd_min = np.min(pcd[:, :3], axis=0)
    block_ij = np.floor((pcd[:, :2] - pcd_min[:2]) / min_res[:2] / nbmat_sz[:2]).astype(np.int32)
    _, block_idx_groups = npi.group_by(block_ij, np.arange(len(block_ij)))

    nb_idxs = []
    nb_pcd_idxs = []
    nb_inverse_idxs = []
    sp_mins = []
    treeloc_idxs = []

    for iter, idx in enumerate(block_idx_groups):
        columns_sp = pcd[idx, :]
        nb_pcd_idx = idx

        sp_min = np.min(columns_sp[:, :3], axis=0)
        sp_max = np.max(columns_sp[:, :3], axis=0)
        treelocs_sp = treelocs[np.all(sp_max[:2] - treelocs[:, :2] > 0, 1) & np.all(treelocs[:, :2] - sp_min[:2] > 0, 1)]

        nb_ijk = np.floor((columns_sp[:, :3] - sp_min) / min_res)
        nb_sel = np.all((nb_ijk < nbmat_sz) & (nb_ijk >= 0), axis=1)
        nb_ijk = nb_ijk[nb_sel]
        nb_pcd_idx = nb_pcd_idx[nb_sel]

        nb_idx = np.ravel_multi_index(np.transpose(nb_ijk.astype(np.int32)), nbmat_sz)
        nb_idx_u, nb_idx_uidx, nb_inverse_idx = np.unique(nb_idx, return_index=True, return_inverse=True)

        treeloc_ijk = np.floor((treelocs_sp[:, :3] - sp_min) / min_res)
        treeloc_within = np.all((treeloc_ijk < nbmat_sz) & (treeloc_ijk >= 0), axis=1)
        treeloc_ijk = treeloc_ijk[treeloc_within]

        treeloc_idx = np.ravel_multi_index(np.transpose(treeloc_ijk[:, :2].astype(np.int32)), nbmat_sz[:2])
        treeloc_idx_u, treeloc_idx_u_idx, treeloc_group_idx = np.unique(treeloc_idx, return_index=True,return_inverse=True)
        nb_idxs.append(nb_idx_u)
        nb_inverse_idxs.append(nb_inverse_idx)
        nb_pcd_idxs.append(nb_pcd_idx)
        sp_mins.append(sp_min)

        treeloc_idxs.append(treeloc_idx_u)

    progress_callback(15)

    pcd_xyz = pcd[:, :3]
    pcd_pred = np.zeros([len(pcd), 2], dtype=np.float32)

    for k, idx in enumerate(nb_idxs):
        x = torch.zeros(nb_tsz, 1)
        if len(idx) > 0:
            x[idx, :] = 1.0

        x = torch.moveaxis(x.reshape((1, *nbmat_sz, 1)).float(), -1, 1)
        x = torch.swapaxes(x, -1, 2)

        x_loc = torch.zeros(nb_tsz2d, 1)
        if len(treeloc_idxs[k]) > 0:
            x_loc[treeloc_idxs[k], :] = 1.0
        x_loc = torch.moveaxis(x_loc.reshape((1, *nbmat_sz[:2], 1)).float(), -1, 1)
        x_loc = torch.swapaxes(x_loc, -1, 2)
        x_loc = torch.unsqueeze(x_loc, 1).repeat(1, 1, nbmat_sz[2], 1, 1)
        x = torch.concatenate([x, x_loc], dim=1)

        with torch.no_grad():
            h = model(x.to(device))

        h_nonzero = torch.moveaxis(
            torch.unsqueeze(torch.moveaxis(torch.swapaxes(h, -1, 2), 1, -1).reshape((nb_tsz, 2))[idx, :],
                            0), -1, 1)
        nb_ann_pred = h_nonzero.cpu().detach().numpy()
        pcd_pred[nb_pcd_idxs[k], :] = np.moveaxis(nb_ann_pred[0], -1, 0)[nb_inverse_idxs[k]]

        progress_value = int(85 * k / len(nb_idxs)) + 15
        progress_callback(progress_value)

    pcd_within_idx = np.concatenate(nb_pcd_idxs)

    pcd_pred_labels = np.zeros(len(pcd))
    pcd_xyz_pred = np.copy(pcd_xyz)
    pcd_xyz_pred[:, :2] = pcd_xyz[:, :2] + pcd_pred[:, :2] * min_res[:2]
    pcd_pred_labels[pcd_within_idx] = mergeshift(pcd_xyz_pred[pcd_within_idx],treelocs=treelocs[:, :3],treeids=treelocs[:, -1])

    return pcd_pred_labels+1#starting from 1


if __name__ == '__main__':
    import os
    import glob
    import laspy
    import pandas as pd

    model_name="treeisonet_als_reclamation_treeoff_segformer3D_128_10cm(GPU4GB)"
    config_file=os.path.join(f"{model_name}.json")

    # data_dir="../../data"
    data_dir=r"D:\xzx\prj\cc-TreeAIBox-plugin\data\itc"

    pcd_fnames = glob.glob(os.path.join(data_dir, "*.la*"))
    model_path = f"../../models/{model_name}.pth"

    out_path="output"

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    for i, pcd_fname in enumerate(pcd_fnames):
        print(f"Processing {pcd_fname}...")
        las = laspy.open(pcd_fname).read()
        pcd = np.transpose(np.array([las.x, las.y, las.z]))  # las.point_format.dimension_names
        basename=os.path.basename(pcd_fname)[:-4]
        treeloc=pd.read_csv(os.path.join(data_dir,basename+"_treeloc.csv"),sep=" ",header=None).to_numpy()
        pred_itc=treeOff(config_file,pcd,treeloc,model_path,use_cuda=True)

        if i==0:
            free, total = torch.cuda.mem_get_info(torch.device("cuda"))
            mem_used_GB = (total - free) / 1024 ** 3
            print(f"Total GPU memory used: {mem_used_GB} GB")

        if len(pred_itc)>0:
            if hasattr(las, 'veg_pred'):
                abg_ind=las.veg_pred>1
                preds=np.zeros(len(las.x))
                preds[abg_ind]=pred_itc
            else:
                preds=pred_itc
            las.add_extra_dim(laspy.ExtraBytesParams(name="pred_itc",type="int32",description="pred_itc"))
            las.pred_itc=preds
            las.write(os.path.join(out_path, "{}_itc.laz".format(basename)))

            torch.cuda.empty_cache()