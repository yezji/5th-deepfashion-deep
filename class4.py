import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import os
import time
import copy

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) #mean, std
    ]),
    'test' : transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
}

NLABELS = 71

batch_size = 176

class_names = "red,orange,yellow,lime,green,sky,blue,purple,navy,pink,black,khaki,gray,white,brown,dark_beige,beige,casual,dandy,formal/office,lovely,luxury,modern,purity,sexy,sporty,street,vintage,All,spring/fall,summer,winter,blazer,blouse_long,blouse_short,cardigan,coach_jacket,coat,dress,dress_shirt_long,dress_shirt_short,flat_shoes,fleece_jacket,hightop,heel,hoody,hoody_jacket,jean,jogger,jumper,leather_jacket,leggings,loafer,long_boots,long_padding,mtm,running_shoes,sandal,short_boots,short_padding,skirt,slacks,sleeveless,sneakers,sweater,top_others,trench_coat,tshirt_long,tshirt_short,vest,walker".split(',')

class_color = class_names[:17]
class_style = class_names[17:28]
class_season = class_names[28:32]
class_category = class_names[32:]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATA_PATH = '/content/drive/My Drive/Colab Notebooks/images/images_v2.3/'
IMG_FILE = '_img.txt'
LABEL_FILE = '_label.txt'
phases = ['train', 'test']


# Convert to one-hot text file from csv file
for i in range(len(phases)):
  data_csv = pd.read_csv(DATA_PATH+phases[i]+".csv")
  data_csv['image_file_name'].to_csv(DATA_PATH+phases[i]+IMG_FILE, index=False)
  labels = pd.get_dummies(data_csv.iloc[:,1:])
  labels.to_csv(DATA_PATH+phases[i]+LABEL_FILE, header=False, index=False, sep=' ')
  labels = np.loadtxt(DATA_PATH+phases[i]+LABEL_FILE, dtype=np.int64)
  label1 = np.nonzero(labels[:,:17])[1]
  label2 = np.nonzero(labels[:,17:28])[1]
  label3 = np.nonzero(labels[:,28:32])[1]
  label4 = np.nonzero(labels[:,32:])[1]
  labels = np.column_stack([label1,label2,label3,label4])
  labels = pd.DataFrame(labels)
  labels.to_csv(DATA_PATH+phases[i]+LABEL_FILE, header=False, index=False, sep=' ')

# Processing Custom Dataset
class DatasetProcessing(Dataset):
    def __init__(self, data_path, img_path, img_filename, label_filename, transform=None):
        self.img_path = os.path.join(data_path, img_path)
        self.transform = transform
        # reading img file from file
        img_filepath = os.path.join(data_path, img_filename)
        fp = open(img_filepath, 'r')
        self.img_filename = [x.strip() for x in fp]
        fp.close()
        # reading labels from file
        label_filepath = os.path.join(data_path, label_filename)
        self.labels = np.loadtxt(label_filepath, dtype=np.int64)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img_filename[index]))
        img = img.convert('RGB')

        label1 = self.labels[index][0]
        label2 = self.labels[index][1]
        label3 = self.labels[index][2]
        label4 = self.labels[index][3]

        if self.transform is not None:
            img = self.transform(img)

        return img, label1, label2, label3, label4

    def __len__(self):
        return len(self.img_filename)


dset_train = DatasetProcessing(DATA_PATH, phases[0], phases[0]+IMG_FILE, phases[0]+LABEL_FILE, data_transforms[phases[0]])
dset_test = DatasetProcessing(DATA_PATH, phases[1], phases[1]+IMG_FILE, phases[1]+LABEL_FILE,  data_transforms[phases[1]])

train_loader = DataLoader(dset_train, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(dset_test, batch_size=batch_size, shuffle=False, num_workers=0)

dataloaders = {'train': train_loader, 'test': test_loader}
dataset_sizes = {'train': len(dset_train), 'test': len(dset_test)}


'''# Show Samples
def imshow(img):
  img = img.numpy().transpose((1, 2, 0))
  mean = np.array([0.485, 0.456, 0.406])
  std = np.array([0.229, 0.224, 0.225])
  img = std * img + mean # unnormalize
  img = np.clip(img, 0, 1)
  plt.imshow(img)

images, label1, label2, label3, label4 = next(iter(dataloaders['train'])) # get batch data from train set
imshow(torchvision.utils.make_grid(images)) # show sample images

for i in range(batch_size): # and print sample labels
  print("sample line %d" % (i+1), class_color[label1[i]], class_style[label2[i]], class_season[label3[i]], class_category[label4[i]])

plt.show()'''

from torchvision.models.mobilenet import mobilenet_v2
import torch.nn as nn
from torch.optim.adam import Adam


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        base = mobilenet_v2(pretrained=True)

        self.feature = base.features
        self.classification_1 = nn.Sequential(
            nn.Linear(1280 * 7 * 7, 17)
        )
        self.classification_2 = nn.Sequential(
            nn.Linear(1280 * 7 * 7, 11)
        )
        self.classification_3 = nn.Sequential(
            nn.Linear(1280 * 7 * 7, 4)
        )
        self.classification_4 = nn.Sequential(
            nn.Linear(1280 * 7 * 7, 39)
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)  # (batch_size, in_features)
        # print("feature(x).size():",x.size())
        out1 = self.classification_1(x)
        out2 = self.classification_2(x)
        out3 = self.classification_3(x)
        out4 = self.classification_4(x)
        print("forward:", out1.data.cpu().numpy().argmax(), out2.data.cpu().numpy().argmax(), out3.data.cpu().numpy().argmax(), out4.data.cpu().numpy().argmax())

        return out1, out2, out3, out4


losses_avg = []
losses1 = []
losses2 = []
losses3 = []
losses4 = []


def view_losses():
    plt.subplot(121)
    plt.title("loss color")
    plt.plot(losses_avg, 'g', losses1, 'b')
    plt.subplot(122)
    plt.title("loss style")
    plt.plot(losses_avg, 'g', losses2, 'b')
    plt.show()
    plt.subplot(121)
    plt.title("loss season")
    plt.plot(losses_avg, 'g', losses3, 'b')
    plt.subplot(122)
    plt.title("loss category")
    plt.plot(losses_avg, 'g', losses4, 'b')
    plt.show()


# model train with three classifier
def train_model(model, criterion, scheduler, optimizer, num_epochs=20):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss1 = 0.0
            running_loss2 = 0.0
            running_loss3 = 0.0
            running_loss4 = 0.0
            running_corrects1 = 0
            running_corrects2 = 0
            running_corrects3 = 0
            running_corrects4 = 0

            read_data_num = 0

            # get inputs and labels from dataloaders
            for inputs, label1, label2, label3, label4 in dataloaders[phase]:
                read_data_num += 1
                print(str(epoch) + " #" + str(read_data_num))

                inputs = inputs.to(device)
                label1 = label1.to(device)
                label2 = label2.to(device)
                label3 = label3.to(device)
                label4 = label4.to(device)
                result1, result2, result3, result4 = model(inputs)

                _, preds1 = result1.max(1)
                _, preds2 = result2.max(1)
                _, preds3 = result3.max(1)
                _, preds4 = result4.max(1)

                loss1 = criterion(result1, label1)
                loss2 = criterion(result2, label2)
                loss3 = criterion(result3, label3)
                loss4 = criterion(result4, label4)

                loss = loss1 + loss2 + loss3 + loss4

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss1 += loss1.item() * inputs.size(0)
                running_loss2 += loss2.item() * inputs.size(0)
                running_loss3 += loss3.item() * inputs.size(0)
                running_loss4 += loss4.item() * inputs.size(0)
                running_corrects1 += torch.sum(preds1 == label1.data)
                running_corrects2 += torch.sum(preds2 == label2.data)
                running_corrects3 += torch.sum(preds3 == label3.data)
                running_corrects4 += torch.sum(preds4 == label4.data)

            epoch_loss_avg = (running_loss1 + running_loss2 + running_loss3 + running_loss4)
            epoch_acc_avg = (running_corrects1.double() + running_corrects2.double() + running_corrects3.double() + running_corrects4.double())
            epoch_loss = epoch_loss_avg / dataset_sizes[phase]
            epoch_acc = epoch_acc_avg / dataset_sizes[phase]
            losses_avg.append(epoch_loss)
            losses1.append(running_loss1)
            losses2.append(running_loss2)
            losses3.append(running_loss3)
            losses4.append(running_loss4)
            print('{} Loss Avg: {:.4f} Acc Avg: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print("[each accs] 1:", running_corrects1.double() / dataset_sizes[phase],
                  "/ 2:", running_corrects2.double() / dataset_sizes[phase],
                  "/ 3:", running_corrects3.double() / dataset_sizes[phase],
                  "/ 4:", running_corrects4.double() / dataset_sizes[phase])

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

        model.load_state_dict(best_model_wts)
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            str_epoch = str(epoch)
            torch.save(model.state_dict(), DATA_PATH + "model-" + str_epoch + ".pth")  # Save checkpoint

            input_tensor = torch.rand(1, 3, 224, 224)  # an example input
            script_module = torch.jit.trace(model, input_tensor)
            script_module.save(DATA_PATH + "class4.pt")

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))
    view_losses()

    return model


model = Model()

optim = Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()
exp_lr_scheduler = lr_scheduler.StepLR(optim, step_size=5, gamma=0.5)

num_epoch = 20

model = train_model(model, criterion, exp_lr_scheduler, optim, num_epochs=num_epoch)


# Testing samples
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Model()
model.load_state_dict(torch.load(DATA_PATH+"model-" + str(num_epoch-1) + ".pth"))
# model.load_state_dict(torch.load(DATA_PATH+"model-19.pth"))

def get_random_images(num):
    data = dset_test
    indices = list(range(len(data)))
    np.random.shuffle(indices)
    idx = indices[:num]
    from torch.utils.data.sampler import SubsetRandomSampler
    sampler = SubsetRandomSampler(idx)
    loader = torch.utils.data.DataLoader(data, sampler=sampler, batch_size=num)
    dataiter = iter(loader)
    images, label1, label2, label3, label4 = dataiter.next()

    return images, label1, label2, label3, label4

def predict_image(image):
    image_tensor = data_transforms['test'](image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output1, output2, output3, output4 = model(input)
    index1 = output1.data.cpu().numpy().argmax()
    index2 = output2.data.cpu().numpy().argmax()
    index3 = output3.data.cpu().numpy().argmax()
    index4 = output4.data.cpu().numpy().argmax()
    return index1, index2, index3, index4

# Get the random image sample, predict them and display the results
to_pil = transforms.ToPILImage()
images, label1, label2, label3, label4 = get_random_images(5)
fig=plt.figure(figsize=(20,20))
for ii in range(len(images)):
    image = to_pil(images[ii])
    index1, index2, index3, index4 = predict_image(image)
    sub = fig.add_subplot(1, len(images), ii+1)
    print("test line %d:" % (ii),"(",index1, index2, index3, index4,")",class_names[index1], class_names[index2+17], class_names[index3+28], class_names[index4+32])
    plt.axis('off')

    plt.imshow(image)
plt.show()

