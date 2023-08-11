import sys
import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet152
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

'''---------------------------The following is the definition of the parameter section---------------------------------'''
#the tuning parameter only needs to be changed here, but not anywhere else

flag_experiments = 3 # Experimental group number
types = 1 # Type number
flag_gans = 1 # Gan structure type。（1: DCGan, 2: CycleGan, 3: StyleGan）。This variable works only when experiment 3/4/5 is conducted。
num_experiments = 1 # times of experiments

epochs = 100
patience = 30 # Early stop parameter. If no early stopping policy is required, set patience = epochs
batch_size = 400 # batch value

lr=0.001 # learning rate
momentum=0.8 # momentum parameter

save_interval = 20 # Saving the model every how many epochs

root_dataset = r'/work/zhining/data' # Dataset Path
outdir_models = r'/work/zhining/Masterarbeit/checkpoint/0.ResNet152V2' # Model Output Path
outdir_img = r'/work/zhining/Masterarbeit/generated_images' # Confusion Matrix Output Path

'''---------------------------Above is the definition parameter section------------------------------------------------'''

dataset_dict = {1: '/1.original_dataset', 2: '/2.hybrid_dataset_1_balance', 3: '/3.hybrid_dataset_2_augmentation'}
gan_dict = {1: '/DCGan', 2: '/CycleGan', 3: '/StyleGan'}
gan_dict_str = {1: 'DCGan', 2: 'CycleGan', 3: 'StyleGan'}
type_dict ={1 : '/type1', 2 : '/type2', 3 : '/type3', 4 : '/type4', 5 : '/type5'}

# Finding the location of a dataset
def root(flag_experiments, types, flag_gans):
    if flag_experiments in [3, 4, 5]:
        if flag_gans not in [1, 2, 3]:
            print('flag_gans error!')
            sys.exit()
        
    root_test = root_dataset + '/4.test_dataset' + type_dict[types]
    
    if flag_experiments == 1 or flag_experiments == 2:
        root_train = root_dataset + dataset_dict[1] + type_dict[types]
    elif flag_experiments == 3:
        root_train = root_dataset + dataset_dict[2] + gan_dict[flag_gans] + type_dict[types]
    elif flag_experiments == 4 or flag_experiments == 5:
        root_train = root_dataset + dataset_dict[3] + gan_dict[flag_gans] + type_dict[types]
    else:
        print('flag_experiments error!')
        sys.exit()
    
    return root_train, root_test
    
# Define the dataset class
class ResampledDataset(Dataset):
    def __init__(self, dataset, target_label, target_size):
        self.dataset = dataset
        self.target_label = target_label
        self.target_size = target_size
        self.resample_indices = self._resample_indices()

    def _resample_indices(self):
        target_indices = []
        for i in range(len(self.dataset)):
            if self.dataset[i][1] == self.target_label:
                target_indices.append(i)
        
        num_target_samples = len(target_indices)
        num_resamples = self.target_size - num_target_samples
        resample_indices = torch.randint(low=0, high=num_target_samples, size=(num_resamples,))
        return torch.tensor(target_indices)[resample_indices]

    def __getitem__(self, index):
        if index < len(self.dataset):
            return self.dataset[index]
        else:
            resample_index = self.resample_indices[index - len(self.dataset)]
            return self.dataset[resample_index][0], self.dataset[resample_index][1]  # 返回样本和标签

    def __len__(self):
        return len(self.dataset) + len(self.resample_indices)

# Observation dataset
def dataset_obs(train_dataset, test_dataset):
    print("-------------------------------")
    # Observation Training Set
    label_counts = {}
    for _, label in train_dataset:
        if label not in label_counts:
            label_counts[label] = 0
        label_counts[label] += 1
    
    for label, count in label_counts.items():
        print(f"train_dataset Label {label}: {count} samples")

    # Observation test set
    label_counts = {}
    for _, label in test_dataset:
        if label not in label_counts:
            label_counts[label] = 0
        label_counts[label] += 1

    for label, count in label_counts.items():
        print(f"test_dataset Label {label}: {count} samples")   
    print("-------------------------------")

# The tensor img has the shape [channels, height, width], where height and width denote the height and width of the image.
class Overlay(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, img):
        half_len = img.shape[2] // 2 # Calculate half of the image width
        left_part = img[:, :, :half_len] # Get the left half of the image
        right_part = img[:, :, -half_len:] # Get the right half of the image
        img = torch.cat([left_part, right_part], dim=1) # Splice the images along the high
        return img

flag_oversampling = 1 if flag_experiments in [2, 5] else 0
root_train, root_test = root(flag_experiments, types, flag_gans)

print('flag_oversampling = ',flag_oversampling)
print('root_train = ',root_train)
print('root_test = ',root_test)

transform = transforms.Compose([
    transforms.ToTensor(),
    Overlay(),                                    
    transforms.Resize((224, 224)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = DatasetFolder(root=root_train, loader=torchvision.datasets.folder.default_loader, extensions='.bmp', transform=transform)
test_dataset = DatasetFolder(root=root_test, loader=torchvision.datasets.folder.default_loader, extensions='.bmp', transform=transform)

dataset_obs(train_dataset, test_dataset)

if flag_oversampling == 0:
    print("no oversampling!")
else:
    print("Oversampling start！")
    label_counts = {}
    for _, label in train_dataset:
        if label not in label_counts:
            label_counts[label] = 0
        label_counts[label] += 1

    # Find the highest number of tags and the corresponding amount of data
    max_label = max(label_counts, key=label_counts.get)
    max_count = label_counts[max_label]

    # Define the dataset after oversampling
    target_label = 0
    target_size = max_count # Equal to the number of data in the "good" tag
    resampled_dataset = ResampledDataset(train_dataset, target_label, target_size)

    train_dataset = resampled_dataset
    
    print("Oversampling succeeded！")
    dataset_obs(train_dataset, test_dataset)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = resnet152(pretrained=True)
model.fc = nn.Linear(2048, 2)
for name, param in model.named_parameters():
    if 'fc' not in name:
        param.requires_grad = False

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

best_loss = float('inf')
counter = 0

for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 30 == 29:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0
    
    # Early Stop
    with torch.no_grad():
        val_loss = 0.0
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
        val_loss /= len(test_loader)
        
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                if flag_experiments in [3, 4, 5]:
                    torch.save(model.state_dict(), outdir_models + f'/E{flag_experiments}N{num_experiments}_{gan_dict_str[flag_gans]}_type{types}_{epoch + 1}.pt')    
                else:
                    torch.save(model.state_dict(), outdir_models + f'/E{flag_experiments}N{num_experiments}_type{types}_{epoch + 1}.pt')
                print("Validation loss did not improve for {} epochs. Stopping early.".format(patience))
                break
    
    if (epoch + 1) % save_interval == 0 or epoch == epochs - 1:
        if flag_experiments in [3, 4, 5]:
            torch.save(model.state_dict(), outdir_models + f'/E{flag_experiments}N{num_experiments}_{gan_dict_str[flag_gans]}_type{types}_{epoch + 1}.pt')    
        else:
            torch.save(model.state_dict(), outdir_models + f'/E{flag_experiments}N{num_experiments}_type{types}_{epoch + 1}.pt')

           
print('Finished Training')

model.eval()
y_true = []
y_pred = []
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))


cm = confusion_matrix(y_true, y_pred, labels=[1, 0])

cm_ratio = cm.astype(float)
cm_ratio[0] =cm_ratio[0]/sum(cm_ratio[0])
cm_ratio[1] =cm_ratio[1]/sum(cm_ratio[1])
cm_ratio = np.round(cm_ratio, 2) # Round to two decimal places.

tn, fp, fn, tp = cm.ravel()
tn_ratio, fp_ratio, fn_ratio, tp_ratio = cm_ratio.ravel()

# compute precision
precision = precision_score(y_true, y_pred, pos_label=0)
precision = round(precision, 2)

# compute recall
recall = recall_score(y_true, y_pred, pos_label=0)
recall = round(recall, 2)

# compute specificity
specificity = tn / (tn + fp)
specificity = round(specificity, 2)

# compute accuracy
accuracy = accuracy_score(y_true, y_pred)
accuracy = round(accuracy*100, 2)

# compute f1
f1 = f1_score(y_true, y_pred, pos_label=0)
f1 = round(f1, 2)

print("-------------------------------")
print("original confusion matrix")
print("TN:", tn)
print("FP:", fp)
print("FN:", fn)
print("TP:", tp)
print("-------------------------------")
print("normalized confusion matrix")
print("TN:", tn_ratio)
print("FP:", fp_ratio)
print("FN:", fn_ratio)
print("TP:", tp_ratio)
print("-------------------------------")
print("Precision:", precision)
print("Recall:", recall)
print("Specificity:", specificity)
print("Accuracy:", accuracy,"%")
print("F1 Score:", f1)
print("-------------------------------")

sns.set(font_scale= 1.1)
sns.heatmap(cm, annot=True, annot_kws={"size": 16}, cmap='Blues', fmt='g', 
            xticklabels=['Flawless', 'Faulty'], yticklabels=['Flawless', 'Faulty'])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
if flag_experiments in [3, 4, 5]:
    plt.savefig(outdir_img + f'/E{flag_experiments}type{types}n{num_experiments}_{gan_dict_str[flag_gans]}_original.png')
else:
    plt.savefig(outdir_img + f'/E{flag_experiments}type{types}n{num_experiments}_original.png')
#plt.show(block=False)
plt.close()

sns.set(font_scale= 1.1)
sns.heatmap(cm_ratio, annot=True, annot_kws={"size": 16}, cmap='Blues', fmt='g', 
            xticklabels=['Flawless', 'Faulty'], yticklabels=['Flawless', 'Faulty'])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
if flag_experiments in [3, 4, 5]:
    plt.savefig(outdir_img + f'/E{flag_experiments}type{types}n{num_experiments}_{gan_dict_str[flag_gans]}_normalized.png')
else:
    plt.savefig(outdir_img + f'/E{flag_experiments}type{types}n{num_experiments}_normalized.png')
#plt.show(block=False)

print("end of running!")