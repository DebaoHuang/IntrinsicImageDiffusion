import os

import torch
from batch import Batch

from iid.data import IIDDataset


class InteriorVerseDataset(IIDDataset):
    FEATURES = ["im", "albedo", "normal", "depth", "material", "mask"]
    DERIVED_FEATURES = []

    DEPTH_SCALING = 1000

    def load_dataset(self, allow_missing_features=False):
        # Collect the data
        data = Batch()
        data['samples'] = Batch(default=Batch)
        data['sample_ids'] = []

        self.module_logger.debug("Collecting features")

        for line in self.split_list:
            line = line.strip()
            if not line:
                continue

            paths = line.split()
            first_path = paths[0]

            if self.stage.name == "Test" or self.stage.name == "Validation":
                parts = first_path.split('/')
                filename = parts[-1]
                view_id = filename.split('.')[0]
                scene_folder = '/'.join(parts[:-1])
                sample_id = os.path.join(scene_folder, view_id)
            else:
                parts = first_path.split('/')
                filename = parts[-1]
                view_id = filename.split('_')[0]
                scene_folder = '/'.join(parts[:-1])
                sample_id = os.path.join(scene_folder, view_id)

            if sample_id not in data['samples']:
                data['sample_ids'].append(sample_id)

                if self.stage.name == "Test" or self.stage.name == "Validation":
                    for feature in self.features_to_include:
                        if feature == "mask":
                            continue
                        elif feature == "im":
                            feature_filename = f"{view_id}.png"
                        elif feature == "albedo":
                            feature_filename = f"{view_id}.exr"
                        feature_path = os.path.join(scene_folder, feature_filename)
                        data['samples'][sample_id][feature] = feature_path

                else:
                    for feature in self.features_to_include:
                        if feature == "mask":
                            feature_filename = f"{view_id}_{feature}.png"
                        else:
                            feature_filename = f"{view_id}_{feature}.exr"
                        feature_path = os.path.join(scene_folder, feature_filename)
                        data['samples'][sample_id][feature] = feature_path

        # Sanity check
        lengths = [len(list(data['samples'][sample_id].keys())) for sample_id in data['samples'].keys()]
        assert all([lengths[0] == l for l in lengths]), "Missing feature!"

        return data


class SubsetSequentialSampler(torch.utils.data.Sampler):
    """
    Samples elements sequentially from a given list of indices, without replacement.
    Adapted from https://github.com/Mephisto405/Learning-Loss-for-Active-Learning
    """

    def __init__(self, indices):
        """
        Creates new sampler
        :param indices: The indices to sample from sequentially
        """
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)
