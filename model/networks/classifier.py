import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(args.D, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, args.num_classes)
        print(f"num classes is : {args.num_classes}")

        nn.init.xavier_normal_(self.fc.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
