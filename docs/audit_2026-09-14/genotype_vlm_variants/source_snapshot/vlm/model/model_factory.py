from model.viz_emb_trainer import VizEmbTrainer


def get_model_factory(model_type):
    if model_type == "viz_emb":
        return VizEmbTrainerFactory()
    raise ValueError(f"Unknown model type: {model_type}")


class VizEmbTrainerFactory:
    def create_model(self, exp_file=None):
        return VizEmbTrainer(exp_file=exp_file)
