import numpy as np
import json
from os.path import join
from typing import List
import os
import trimesh


def compute_iou(pred, gt, face_areas=None):
    """Compute IoU with optional face area weighting"""
    if face_areas is None:
        # Original implementation - treat all faces equally
        intersection = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
    else:
        # Area-weighted implementation
        intersection_mask = np.logical_and(pred, gt)
        union_mask = np.logical_or(pred, gt)
        intersection = np.sum(face_areas[intersection_mask])
        union = np.sum(face_areas[union_mask])

    if union != 0:
        return (intersection / union) * 100
    else:
        return 0


def eval_single_gt_shape(gt_label, pred_masks, face_areas=None):
    # gt: [N,], label index
    # pred: [B, N], B is the number of predicted parts, binary label
    unique_gt_label = np.unique(gt_label)
    best_ious = []
    for label in unique_gt_label:
        best_iou = 0
        if label == -1:
            continue
        for mask in pred_masks:
            iou = compute_iou(mask, gt_label == label, face_areas)
            best_iou = max(best_iou, iou)
        best_ious.append(best_iou)
    return np.mean(best_ious)


def eval_whole_dataset(pred_folder, merge_parts=False):
    print(pred_folder)
    meta = json.load(open("blades_semantic.json", "r"))

    categories = meta.keys()
    results_per_cat = {}
    per_cat_mious = []
    overall_mious = []

    MAX_NUM_CLUSTERS = 20
    view_id = 0

    print("\n" + "=" * 80)
    print("BEST NUMBER OF CLUSTERS FOR EACH OBJECT")
    print("=" * 80)

    for cat in categories:
        results_per_cat[cat] = []
        print(f"\nCategory: {cat}")
        print("-" * 40)

        for shape_id in meta[cat].keys():
            try:
                all_pred_labels = []
                cluster_mious = {}  # Track mIoU for each cluster count

                for num_cluster in range(2, MAX_NUM_CLUSTERS):
                    ### load each label
                    fname_clustering = (
                        os.path.join(
                            pred_folder,
                            "cluster_out",
                            str(shape_id) + "_" + str(view_id) + "_" + str(num_cluster).zfill(2),
                        )
                        + ".npy"
                    )

                    if not os.path.exists(fname_clustering):
                        continue

                    pred_label = np.load(fname_clustering)
                    all_pred_labels.append((num_cluster, np.squeeze(pred_label)))

            except:
                continue

            if not all_pred_labels:
                continue

            #### Path for Blade Labels and Shapes
            gt_labels_path = "data/blades/gt"
            shapes_path = "data/blades/shapes"
            #################################

            gt_label = np.loadtxt(os.path.join(gt_labels_path, shape_id + ".seg"))
            gt_label = gt_label.astype(np.int32) - 1

            # Load mesh and compute face areas using trimesh
            mesh_file = os.path.join(shapes_path, shape_id + ".obj")
            try:
                mesh = trimesh.load(mesh_file)
                face_areas = mesh.area_faces
            except Exception as e:
                face_areas = None

            if merge_parts:
                pred_masks = []
                for num_cluster, result in all_pred_labels:
                    pred = result
                    assert pred.shape[0] == gt_label.shape[0]
                    for label in np.unique(pred):
                        pred_masks.append(pred == label)
                miou = eval_single_gt_shape(gt_label, np.array(pred_masks), face_areas)
                results_per_cat[cat].append(miou)
            else:
                best_miou = 0
                best_num_clusters = 0

                for num_cluster, result in all_pred_labels:
                    pred_masks = []
                    pred = result

                    for label in np.unique(pred):
                        pred_masks.append(pred == label)
                    miou = eval_single_gt_shape(gt_label, np.array(pred_masks), face_areas)
                    cluster_mious[num_cluster] = miou

                    if miou > best_miou:
                        best_miou = miou
                        best_num_clusters = num_cluster

                # Print results for this object
                print(f"Object: {shape_id}")
                print(f"  Best number of clusters: {best_num_clusters}")
                print(f"  Best mIoU: {best_miou:.2f}%")

                # Show top 3 cluster numbers for this object
                sorted_results = sorted(cluster_mious.items(), key=lambda x: x[1], reverse=True)
                print(f"  Top 3 cluster counts:")
                for i, (clusters, miou) in enumerate(sorted_results[:3]):
                    print(f"    {i + 1}. {clusters} clusters: {miou:.2f}% mIoU")
                print()

                results_per_cat[cat].append(best_miou)

        print(f"Category {cat} average mIoU: {np.mean(results_per_cat[cat]):.2f}%")
        per_cat_mious.append(np.mean(results_per_cat[cat]))
        overall_mious += results_per_cat[cat]

    print("=" * 80)
    print(f"Overall average mIoU: {np.mean(overall_mious):.2f}% ({len(overall_mious)} objects)")
    print(f"Per-category average: {np.mean(per_cat_mious):.2f}%")


if __name__ == "__main__":
    eval_whole_dataset("exp_results/clustering/blades")
