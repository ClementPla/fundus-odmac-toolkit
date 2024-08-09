import torch.nn as nn
import torchseg
from huggingface_hub import PyTorchModelHubMixin, get_collection, list_repo_refs

ROOT_HF = "ClementP/fundus-odmac-segmentation-"
COLLECTION_ID = "66b663ed4e30b920790fd4c7"


class HuggingFaceModel(PyTorchModelHubMixin, nn.Module):
    def __init__(self, arch: str, encoder: str, in_channels: int = 3, classes: int = 3):
        super().__init__()
        self.model = torchseg.create_model(
            arch=arch,
            encoder_name=encoder,
            encoder_weights=None,
            in_channels=in_channels,
            classes=classes,
        )


def download_model(arch, encoder_name):
    model = HuggingFaceModel.from_pretrained(f"{ROOT_HF}{arch}-{encoder_name}", arch=arch, encoder=encoder_name).model
    return model


def list_models():
    collection = get_collection(ROOT_HF + COLLECTION_ID)
    print("Architecture | \033[94m Encoder | \033[92m Variants")
    for item in collection.items:
        if item.item_type == "model":
            name = item.item_id.split(ROOT_HF)[1]
            arch = name.split("-")[0]
            encoder = "_".join(name.split("-")[1:])
            branches = list_repo_refs(item.item_id).branches
            print(
                "\033[1m" + arch,
                "\033[94m" + encoder,
                f"\033[92m ({len(branches)} variants)",
            )
