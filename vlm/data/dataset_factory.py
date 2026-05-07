from data.vqa_dataset import MouseTrajDataset


def get_dataset_factory(dataset_type):
    if dataset_type == "mouse_traj_vqa":
        return MouseTrajDatasetFactory()
    raise ValueError(f"Unknown dataset type: {dataset_type}")


class MouseTrajDatasetFactory:
    def create_dataset(self, **kwargs):
        return MouseTrajDataset(**kwargs)
