from fundus_data_toolkit.datamodules.segmentation import (
    APTOSODMACDataModule,
    DDRODMACDataModule,
    EYEPACSODMACDataModule,
    IDRIDODMACDataModule,
)
from fundus_data_toolkit.datamodules.utils import merge_existing_datamodules


def get_datamodule_from_paths(paths: dict[str, str], config: dict):
    datamodules = []
    for dname, path in paths.items():
        match dname.upper():
            case "APTOS":
                datamodule = APTOSODMACDataModule(data_dir=path, **config)
            case "DDR":
                datamodule = DDRODMACDataModule(data_dir=path, **config)
            case "EYEPACS":
                datamodule = EYEPACSODMACDataModule(data_dir=path, **config)
            case "IDRID":
                datamodule = IDRIDODMACDataModule(data_dir=path, **config)

        datamodules.append(datamodule)

    dm = merge_existing_datamodules(datamodules)
    dm.setup_all()
    return dm


def get_datamodule_from_config(config):
    paths = config["paths"]
    dconfig = config["dataset"]

    return get_datamodule_from_paths(paths, dconfig)
