from warnings import warn
import torch


def get_device(gpu_index=None, debug=False):
    if torch.cuda.is_available() and gpu_index is not None:
        device = torch.device("cuda:%d" % gpu_index)

        if debug:
            print("Using GPU %d." % gpu_index)
    else:
        device = torch.device("cpu")

        if debug:
            warn("Warning: no GPU found; running with CPU.", UserWarning)

    return device
