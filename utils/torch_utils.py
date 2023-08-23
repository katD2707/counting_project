import torch


def select_device(device='cpu', batch_size=None):
    cpu = device.lower() == 'cpu'

    if not cpu and not torch.cuda.is_available():
        raise Exception('no CUDA installation found on this machine.')

    cuda = not cpu and torch.cuda.is_available()

    if cuda:
        n = torch.cuda.device_count()
        if n == 0:
            raise Exception('no GPU found on this machine.')
        if n > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'

    return torch.device('cuda:0' if cuda else 'cpu')