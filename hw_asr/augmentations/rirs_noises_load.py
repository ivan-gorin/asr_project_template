from speechbrain.utils.data_utils import download_file
import shutil
from hw_asr.utils import ROOT_PATH

URL_LINK = "https://www.openslr.org/resources/28/rirs_noises.zip"


def load_noise_data():
    data_dir = ROOT_PATH / "data" / "noise"
    data_dir.mkdir(exist_ok=True, parents=True)
    rirs_dir = data_dir / "RIRS_NOISES"
    if rirs_dir.exists():
        return rirs_dir
    arch_path = data_dir / "rir_noises.zip"
    print(f"Loading noise data")
    download_file(URL_LINK, arch_path)
    shutil.unpack_archive(arch_path, data_dir)
    return rirs_dir
