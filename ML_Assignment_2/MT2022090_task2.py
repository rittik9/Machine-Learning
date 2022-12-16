STUDENT_ROLLNO = 'MT2022090' #yourRollNumberHere
STUDENT_NAME = 'Rittik Panda' #yourNameHere
#@PROTECTED_1
##DO NOT MODIFY THE BELOW CODE. NO OTHER IMPORTS ALLOWED. NO OTHER FILE LOADING OR SAVING ALLOWED.
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import sklearn.preprocessing as preprocessing
import sklearn.model_selection as model_selection
import torch.utils.data as data
import numpy as np
import torch
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_test = np.load("X_test.npy")
submission = np.load("sample_submission.npy")
#@PROTECTED_1
X_test1 = X_test
X_train,X_test,y_train,y_test =model_selection.train_test_split(X_train,y_train,test_size=0.2,random_state=77 )
X_train=torch.FloatTensor(X_train/255).cuda()
X_test=torch.FloatTensor(X_test/255).cuda()
y_train=torch.LongTensor(y_train).cuda()

# MODEL
class PandaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(3072, 1024)
        self.l2 = nn.Linear(1024, 925)
        self.l3 = nn.Linear(925, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.dropout(x)
        x = self.relu(self.l2(x))
        x = self.dropout(x)
        x = self.l3(x)
        return self.softmax(x)

torch.manual_seed(80)
model = PandaNet().cuda()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

#Model Train
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
Epoch = 2000
t = list(range(Epoch))
loss_history = []
model.train()
for i in range(Epoch):
    i = i + 1
    optimizer.zero_grad()
    y_pred = model.forward(X_train)
    loss = criterion(y_pred, y_train)
    loss_history.append(loss)
    loss.backward()
    optimizer.step()
    y_pred = y_pred.cpu()
    y_pred = y_pred.detach().numpy()
    predictions = np.argmax(y_pred, axis=1)

    y_train = y_train.cpu()
    y_train = y_train.detach().numpy()
    # metric = torchmetrics.classification.MulticlassAccuracy(num_classes=10).to(device)
    print("Epoch number: {} and the loss : {} and accuracy : {}".format(i, loss.item(),
                                                                        get_accuracy(predictions, y_train)))
    y_train = torch.LongTensor(y_train).cuda()

#Checking accuracy
y_pred = model.forward(X_test)
y_pred = y_pred.cpu()
y_pred = y_pred.detach().numpy()
predictions = np.argmax(y_pred, axis=1)

r = get_accuracy(predictions, y_test)
print(r)

#Predicting on test data
X_test1=torch.FloatTensor(X_test1/255).cuda()
y_pred1 = model.forward(X_test1)
y_pred1 = y_pred1.cpu()
y_pred1 = y_pred1.detach().numpy()
submission = np.argmax(y_pred1, axis=1)

#@PROTECTED_2
np.save("{}__{}".format(STUDENT_ROLLNO,STUDENT_NAME),submission)
#@PROTECTED_2