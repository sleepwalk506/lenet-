from torch import nn
import torch
import torch.nn.functional as F

# in_channels 输入张量 channels 如rgb图像就是3
# nn.ReLU(inplace=True)  参数为True是为了从上层网络Conv2d中传递下来的tensor直接进行修改，这样能够节省运算内存，不用多存储其他变量

class LeNet(nn.Module):
    def __init__(self,num_classes=10,init_weights=False):
        super(LeNet,self).__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.relu = nn.ReLU(True)

        self.conv2 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5,stride=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,num_classes)

        if init_weights:
            self._initialize_weights()

    def forward(self,x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        x = torch.flatten(x,start_dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):#如果m是卷积层
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias,0)

if __name__ == '__main__':
    x = torch.rand([1,1,28,28])
    model = LeNet()
    y = model(x)