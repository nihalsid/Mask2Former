from scipy.sparse import csr_matrix
import numpy as np
import torch
from scipy.sparse.csgraph import connected_components
import time


class TTAHandler:

    def __init__(self, segment_probabilities, segment_masks, iou_threshold=0.5):
        assert (len(segment_masks) == len(segment_probabilities)), "number of probs and masks should be equal"
        self.segment_probabilities = segment_probabilities
        self.segment_masks = segment_masks
        self.num_segments = len(segment_masks)
        self.iou_threshold = iou_threshold

    def find_tta_probabilities_and_masks(self):
        graph = np.zeros((self.num_segments, self.num_segments), dtype=np.int32)
        segment_mask_gpu = torch.stack(self.segment_masks, 0).cuda()
        for i in range(self.num_segments):
            for j in range(self.num_segments):
                if i != j:
                    soft_iou = calculate_soft_iou(segment_mask_gpu[i], segment_mask_gpu[j])
                    graph[i, j] = int(soft_iou >= self.iou_threshold)
        graph = csr_matrix(graph)
        ccomponents = connected_components(csgraph=graph, directed=False, return_labels=True)[1]
        unique_comps, unique_counts = np.unique(ccomponents, return_counts=True)
        cluster_probabilities = [torch.zeros_like(self.segment_probabilities[0]) for _ in range(len(unique_comps))]
        cluster_masks = [torch.zeros_like(self.segment_masks[0]) for _ in range(len(unique_comps))]
        for segment_idx, cluster_idx in enumerate(ccomponents):
            cluster_masks[cluster_idx] += self.segment_masks[segment_idx]
            cluster_probabilities[cluster_idx] += self.segment_probabilities[segment_idx]
        for uidx, cluster_idx in enumerate(unique_comps):
            cluster_masks[cluster_idx] /= unique_counts[uidx]
            cluster_probabilities[cluster_idx] /= unique_counts[uidx]
        return torch.stack(cluster_probabilities, 0), torch.stack(cluster_masks, 0)


def calculate_soft_iou(mask_a, mask_b):
    return torch.min(mask_a, mask_b).sum() / torch.max(mask_a, mask_b).sum()


if __name__ == "__main__":
    pass