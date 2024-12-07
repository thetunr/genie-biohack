from __future__ import print_function, division
import torch
from torchvision import transforms, models
import pandas as pd
import numpy as np
import os
import skimage
from skimage import io
import warnings
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import WeightedRandomSampler
from torch.optim import lr_scheduler
import time
import copy
import sys
from sklearn.model_selection import train_test_split, KFold

warnings.filterwarnings("ignore")

def flatten(list_of_lists):
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])


class VGG16WithMetadata(nn.Module):
    def __init__(self, vgg_model, num_classes, num_metadata_features=1):
        super(VGG16WithMetadata, self).__init__()
        # Image feature extraction (VGG16 without classifier)
        self.features = vgg_model.features
        # Separate paths for image and metadata
        self.image_classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 256)
        )
        # Metadata processing path
        self.metadata_processor = nn.Sequential(
            nn.Linear(num_metadata_features, 16),
            nn.ReLU(),
            nn.Linear(16, 32)
        )
        # Combined classifier
        self.final_classifier = nn.Sequential(
            nn.Linear(256 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x, metadata):
        # Process image through VGG features
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.image_classifier(x)
        # Process metadata
        metadata = self.metadata_processor(metadata)
        # Combine features
        combined = torch.cat((x, metadata), dim=1)
        # Final classification
        return self.final_classifier(combined)



def train_model(label, dataloaders, device, dataset_sizes, model,
                criterion, optimizer, scheduler, num_epochs=2):
    since = time.time()
    training_results = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 0.0
    for epoch in range(num_epochs):
        print('-' * 20)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                scheduler.step()
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            # running_total = 0
            print(phase)
            # Iterate over data.
            for batch in dataloaders[phase]:
                inputs = batch["image"].to(device)
                # MARK: metadata
                metadata = batch["fitzpatrick_scale"].float().unsqueeze(1).to(device)
                labels = batch[label]
                labels = torch.from_numpy(np.asarray(labels)).to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward

                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    inputs = inputs.float()  # ADDED AS A FIX
                    # MARK: metadata
                    outputs = model.module(inputs, metadata)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            # print("Loss: {}/{}".format(running_loss, dataset_sizes[phase]))
            print("Accuracy: {}/{}".format(running_corrects,
                                           dataset_sizes[phase]))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            training_results.append([phase, epoch, epoch_loss, epoch_acc])
            if epoch > 10:
                if phase == 'val' and epoch_loss < best_loss:
                    print("New leading accuracy: {}".format(epoch_acc))
                    best_acc = epoch_acc
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
            elif phase == 'val':
                best_loss = epoch_loss
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    training_results = pd.DataFrame(training_results)
    training_results.columns = ["phase", "epoch", "loss", "accuracy"]
    return model, training_results


class SkinDataset():
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir, self.df.loc[self.df.index[idx], 'hasher'] + '.jpg')

        # MARK: Image Reading
        image = io.imread(img_name)

        if(len(image.shape) < 3):
            image = skimage.color.gray2rgb(image)

        hasher = self.df.loc[self.df.index[idx], 'hasher']
        high = self.df.loc[self.df.index[idx], 'high']
        mid = self.df.loc[self.df.index[idx], 'mid']
        low = self.df.loc[self.df.index[idx], 'low']
        fitzpatrick_scale = self.df.loc[self.df.index[idx], 'fitzpatrick_scale']
        if self.transform:
            image = self.transform(image)
        sample = {
                    'image': image,
                    'high': high,
                    'mid': mid,
                    'low': low,
                    'hasher': hasher,
                    'fitzpatrick_scale': fitzpatrick_scale
                }
        return sample


def custom_load(
        batch_size=256,
        num_workers=20,
        train_dir='',
        val_dir='',
        # MARK: image_dir
        image_dir='/teamspace/studios/this_studio/images'):
    val = pd.read_csv(val_dir)
    train = pd.read_csv(train_dir)
    class_sample_count = np.array(train[label].value_counts().sort_index())
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in train[label]])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(
        samples_weight.type('torch.DoubleTensor'),
        len(samples_weight),
        replacement=True)
    dataset_sizes = {"train": train.shape[0], "val": val.shape[0]}
    transformed_train = SkinDataset(
        csv_file=train_dir,
        root_dir=image_dir,
        transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=224),  # Image net standards
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
            ])
        )
    transformed_test = SkinDataset(
        csv_file=val_dir,
        root_dir=image_dir,
        transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        )
    dataloaders = {
        "train": torch.utils.data.DataLoader(
            transformed_train,
            batch_size=batch_size,
            sampler=sampler,
            # shuffle=True,
            num_workers=num_workers),
        "val": torch.utils.data.DataLoader(
            transformed_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers)
        }
    return dataloaders, dataset_sizes



# MARK: Main Function
if __name__ == '__main__':
    print(os.getcwd())
    # In the custom_load() function, make sure to specify the path to the images
    print("\nPlease specify number of epochs and 'dev' mode or not... e.g. python train.py 10 full \n")
    n_epochs = int(sys.argv[1])
    dev_mode = sys.argv[2]
    print("CUDA is available: {} \n".format(torch.cuda.is_available()))
    print("Starting... \n")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # MARK: csv file
    # csv_file = "fitzpatrick17k.csv"
    # csv_file = "cleaned_file.csv"
    # csv_file = "fitzpatrick17k_n12631.csv"
    # csv_file = "fitzpatrick17k_trunc.csv"
    csv_file = '/teamspace/studios/image-scraping/fitzpatrick17k_trunc.csv'

    if dev_mode == "dev":
        df = pd.read_csv(csv_file).sample(1000)
    else:
        df = pd.read_csv(csv_file)
    print(df['fitzpatrick_scale'].value_counts())
    print("Rows: {}".format(df.shape[0]))
    df["low"] = df['label'].astype('category').cat.codes
    df["mid"] = df['nine_partition_label'].astype('category').cat.codes
    df["high"] = df['three_partition_label'].astype('category').cat.codes
    df["hasher"] = df["md5hash"]

    # for holdout_set in ["expert_select","random_holdout", "a12", "a34","a56", "dermaamin","br"]:
    for holdout_set in ["expert_select", "random_holdout"]:
        print(holdout_set)
        if holdout_set == "expert_select":
            df2 = df
            train = df2[df2.qc.isnull()]
            test = df2[df2.qc=="1 Diagnostic"]
        elif holdout_set == "random_holdout":
            # MARK: df.low vs df.high
            train, test, y_train, y_test = train_test_split(
                                                df,
                                                df.high,
                                                test_size=0.2,
                                                random_state=4242,
                                                stratify=df.high)
        elif holdout_set == "dermaamin":
            combo = set(df[df.image_path.str.contains("dermaamin")==True].label.unique()) & set(df[df.image_path.str.contains("dermaamin")==False].label.unique())
            df = df[df.label.isin(combo)]
            df["high"] = df['label'].astype('category').cat.codes
            train = df[df.image_path.str.contains("dermaamin") == False]
            test = df[df.image_path.str.contains("dermaamin")]
        elif holdout_set == "br":
            combo = set(df[df.image_path.str.contains("dermaamin")==True].label.unique()) & set(df[df.image_path.str.contains("dermaamin")==False].label.unique())
            df = df[df.label.isin(combo)]
            df["high"] = df['label'].astype('category').cat.codes
            train = df[df.image_path.str.contains("dermaamin")]
            test = df[df.image_path.str.contains("dermaamin") == False]
            print(train.label.nunique())
            print(test.label.nunique())
        elif holdout_set == "a12":
            train = df[(df.fitzpatrick_scale==1)|(df.fitzpatrick_scale==2)]
            test = df[(df.fitzpatrick_scale!=1)&(df.fitzpatrick_scale!=2)]
            combo = set(train.label.unique()) & set(test.label.unique())
            print(combo)
            train = train[train.label.isin(combo)].reset_index()
            test = test[test.label.isin(combo)].reset_index()
            train["high"] = train['label'].astype('category').cat.codes
            test["high"] = test['label'].astype('category').cat.codes
        elif holdout_set == "a34":
            train = df[(df.fitzpatrick_scale==3)|(df.fitzpatrick_scale==4)]
            test = df[(df.fitzpatrick_scale!=3)&(df.fitzpatrick_scale!=4)]
            combo = set(train.label.unique()) & set(test.label.unique())
            train = train[train.label.isin(combo)].reset_index()
            test = test[test.label.isin(combo)].reset_index()
            train["high"] = train['label'].astype('category').cat.codes
            test["high"] = test['label'].astype('category').cat.codes
        elif holdout_set == "a56":
            train = df[(df.fitzpatrick_scale==5)|(df.fitzpatrick_scale==6)]
            test = df[(df.fitzpatrick_scale!=5)&(df.fitzpatrick_scale!=6)]
            combo = set(train.label.unique()) & set(test.label.unique())
            train = train[train.label.isin(combo)].reset_index()
            test = test[test.label.isin(combo)].reset_index()
            train["high"] = train['label'].astype('category').cat.codes
            test["high"] = test['label'].astype('category').cat.codes
        print(test.shape)
        print(test.shape)
        train_path = "train-test/temp_train.csv"
        test_path = "train-test/temp_test.csv"
        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)
        print("Training Shape: {}, Test Shape: {} \n".format(
        train.shape,
        test.shape)
        )
        for indexer, label in enumerate(["high"]):
            print(label)
            weights = np.array(max(train[label].value_counts())/train[label].value_counts().sort_index())
            label_codes = sorted(list(train[label].unique()))
            dataloaders, dataset_sizes = custom_load(
                256,
                20,
                "{}".format(train_path),
                "{}".format(test_path))
            

            # MARK: start of metadata change

            base_model = models.vgg16(pretrained=True)

            for param in base_model.parameters():
                param.requires_grad = False
            
            # model_ft.classifier[6] = nn.Sequential(
            #             nn.Linear(4096, 256), 
            #             nn.ReLU(), 
            #             nn.Dropout(0.4),
            #             nn.Linear(256, len(label_codes)),                   
            #             nn.LogSoftmax(dim=1))

            model_ft = VGG16WithMetadata(
                vgg_model=base_model,
                num_classes=len(label_codes),
                num_metadata_features=1  # Fitzpatrick scale
            )

            total_params = sum(p.numel() for p in model_ft.parameters())

            print('{} total parameters'.format(total_params))
            total_trainable_params = sum(
                p.numel() for p in model_ft.parameters() if p.requires_grad)
            print('{} total trainable parameters'.format(total_trainable_params))

            model_ft = model_ft.to(device)
            model_ft = nn.DataParallel(model_ft)

            class_weights = torch.FloatTensor(weights)

            criterion = nn.NLLLoss()
            optimizer_ft = optim.Adam(model_ft.parameters())
            exp_lr_scheduler = lr_scheduler.StepLR(
                optimizer_ft,
                step_size=7,
                gamma=0.1)

            # MARK: end of metadata change

            
            print("\nTraining classifier for {}........ \n".format(label))
            print("....... processing ........ \n")
            model_ft, training_results = train_model(
                label,
                dataloaders, device,
                dataset_sizes, model_ft,
                criterion, optimizer_ft,
                exp_lr_scheduler, n_epochs)
            print("Training Complete")
            torch.save(model_ft.state_dict(), "pth/model_path_{}_{}_{}.pth".format(n_epochs, label, holdout_set))
            print("gold")
            training_results.to_csv("training/training_{}_{}_{}.csv".format(n_epochs, label, holdout_set))
            model = model_ft.eval()
            loader = dataloaders["val"]
            prediction_list = []
            fitzpatrick_scale_list = []
            hasher_list = []
            labels_list = []
            p_list = []
            topk_p = []
            topk_n = []
            d1 = []
            d2 = []
            d3 = []
            p1 = []
            p2 = []
            p3 = []
            with torch.no_grad():
                running_corrects = 0
                for i, batch in enumerate(dataloaders['val']):
                    inputs = batch["image"].to(device)
                    metadata = batch["fitzpatrick_scale"].float().unsqueeze(1).to(device)
                    classes = batch[label].to(device)
                    fitzpatrick_scale = batch["fitzpatrick_scale"]
                    hasher = batch["hasher"]
                    outputs = model.module(inputs.float(), metadata)
                    probability = outputs
                    ppp, preds = torch.topk(probability, 1)
                    if label == "high":
                        _, preds5 = torch.topk(probability, 3)
                        topk_p.append(np.exp(_.cpu()).tolist())
                        topk_n.append(preds5.cpu().tolist())
                    running_corrects += torch.sum(preds == classes.data)
                    p_list.append(ppp.cpu().tolist())
                    prediction_list.append(preds.cpu().tolist())
                    labels_list.append(classes.tolist())
                    fitzpatrick_scale_list.append(fitzpatrick_scale.tolist())
                    hasher_list.append(hasher)
                acc = float(running_corrects)/float(dataset_sizes['val'])
            if label == "high":
                for j in topk_n:
                    for i in j:
                        d1.append(i[0])
                        d2.append(i[1])
                        d3.append(i[2])
                for j in topk_p:
                    for i in j:
                        print(i)
                        p1.append(i[0])
                        p2.append(i[1])
                        p3.append(i[2])
                df_x=pd.DataFrame({
                                    "hasher": flatten(hasher_list),
                                    "label": flatten(labels_list),
                                    "fitzpatrick_scale": flatten(fitzpatrick_scale_list),
                                    "prediction_probability": flatten(p_list),
                                    "prediction": flatten(prediction_list),
                                    "d1": d1,
                                    "d2": d2,
                                    "d3": d3,
                                    "p1": p1,
                                    "p2": p2,
                                    "p3": p3})
            else:
                # print(len(flatten(hasher_list)))
                # print(len(flatten(labels_list)))
                # print(len(flatten(fitzpatrick_scale_list)))
                # print(len(flatten(p_list)))
                # print(len(flatten(prediction_list)))
                df_x=pd.DataFrame({
                                    "hasher": flatten(hasher_list),
                                    "label": flatten(labels_list),
                                    "fitzpatrick_scale": flatten(fitzpatrick_scale_list),
                                    "prediction_probability": flatten(p_list),
                                    "prediction": flatten(prediction_list)})
            df_x.to_csv("results/results_{}_{}_{}.csv".format(n_epochs, label, holdout_set),
                            index=False)
            # print("\n Accuracy: {} \n".format(acc))
        print("done")
