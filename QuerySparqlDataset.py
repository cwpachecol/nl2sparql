import pandas as pd
import torch
import torchvision.transforms as transforms  # Transformations we can perform on our dataset


class QuerySparqlDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.df_csv_file = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        return len(self.df_csv_file)

    def __getitem__(self, index):
       question = self.df_csv_file.iloc[index, 0]
       # question = torch.tensor(question)
       sparql = self.df_csv_file.iloc[index, 1]
       # sparql = torch.tensor(sparql)
       return (question, sparql)


# Load Data
DatasetTrain = QuerySparqlDataset(
    csv_file="data/DBNQA/DBNQA_total_wd.csv", root_dir="test123", transform=transforms.ToTensor()
)

batch_size = 1
train_len = int(0.7*len(DatasetTrain))
test_len = len(DatasetTrain) - train_len
train_set, test_set = torch.utils.data.random_split(DatasetTrain, [train_len, test_len])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

print(DatasetTrain.__getitem__(10))
# print(len(DatasetTrain))
# print(len(train_set))
# print(len(test_set))
# print(len(train_loader))
# print(len(test_loader))

