import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(640, args.num_classes)
        print(f"num classes is : {args.num_classes}")
        nn.init.xavier_normal_(self.fc.weight)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.softmax(x)
        return x
