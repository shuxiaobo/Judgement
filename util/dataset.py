from torch.utils.data import  Dataset

class dataset(Dataset):

    def __init__(self, data):
        """
        data will be (samples)
        :param data:
        """
        self.data = data

    def __getitem__(self, index):


    def __len__(self):
        raise NotImplementedError