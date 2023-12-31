import argparse
import os
import shutil
from os.path import join
import cv2
import dataset.dataset as dtset
import torch
import numpy as np
import random
from metrics.metric_tool import ConfuseMatrixMeter
from models.change_classifier import ChangeClassifier as Model
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from pynvml import *
import time
import torch.nn as nn
import torch.nn.functional as F
'''
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        targets = targets.type(torch.LongTensor)
        loss = (self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss).mean()
        return loss'''
        
        
ALPHA = 0.9
GAMMA = 2

class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
    
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss
#import neptune
#from neptune.types import File
#from neptune.utils import stringify_unsupported
#from neptune_pytorch import NeptuneLogger
#run = neptune.init_run(
#    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyMmI5NDFmNy03NzkzLTQ0ZmMtYjE0OC04MzJkOTQ3ZDc3OGYifQ==",
#    project="dataeazedl/deeplearning",  # replace with your own
#)
metric_arr = np.zeros((4,100))
def print_gpu_utilization(enm):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"{enm} GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    #print_gpu_utilization()


def parse_arguments():
    # Argument Parser creation
    parser = argparse.ArgumentParser(
        description="Parameter for data analysis, data cleaning and model training."
    )
    parser.add_argument(
        "--datapath",
        type=str,
        help="data path",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        help="log path",
    )

    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed training)')

    parsed_arguments = parser.parse_args()

    # create log dir if it doesn't exists
    if not os.path.exists(parsed_arguments.log_path):
        os.mkdir(parsed_arguments.log_path)

    dir_run = sorted(
        [
            filename
            for filename in os.listdir(parsed_arguments.log_path)
            if filename.startswith("run_")
        ]
    )

    if len(dir_run) > 0:
        num_run = int(dir_run[-1].split("_")[-1]) + 1
    else:
        num_run = 0
    parsed_arguments.log_path = os.path.join(
        parsed_arguments.log_path, "run_%04d" % num_run + "/"
    )

    return parsed_arguments


def train(
    dataset_train,
    dataset_val,
    model,
    criterion,
    optimizer,
    scheduler,
    logpath,
    writer,
    epochs,
    save_after,
    device
):

    model = model.to(device)
    # Hyperparams for training
    parameters = {"lr": 0.002596776436816101,"bs": 12,"input_sz": 256 * 256 * 3,"n_classes": 2,"model_filename": "basemodel","device": torch.device("cuda" if torch.cuda.is_available() else        "cpu"),"epochs": 100,}

    #npt_logger = NeptuneLogger(run,model=model,log_model_diagram=True,log_gradients=True,log_parameters=True,log_freq=1)

    tool4metric = ConfuseMatrixMeter(n_class=2)

    def evaluate(reference, testimg, mask):
        # All the tensors on the device:
        reference = reference.to(device).float()
        testimg = testimg.to(device).float()
        mask = mask.to(device).float()

        # Evaluating the model:
        generated_mask = model(reference, testimg).squeeze(1)

        # Loss gradient descend step:
        it_loss = criterion(generated_mask, mask)

        # Feeding the comparison metric tool:
        bin_genmask = (generated_mask.to("cpu") >  
                       0.5).detach().numpy().astype(int)   # Changes done (orignal = 0.5)
        mask = mask.to("cpu").numpy().astype(int)    
        tool4metric.update_cm(pr=bin_genmask, gt=mask)

        return it_loss

    def training_phase(epc):
        tool4metric.clear()
        print("Epoch {}".format(epc))
        model.train()
        epoch_loss = 0.0
        counter1 = 0
        t0 = time.time()
        #cls0,cls1 = 0,0
        for (reference, testimg), mask in dataset_train:
            #print(".........",counter1)
            '''for bt in range(8):
                cls0 +=  np.count_nonzero(mask[bt]==0)
                cls1 += np.count_nonzero(mask[bt]>0)'''
            '''
            #verfier 
            for bt in range(8):
                print((np.count_nonzero(mask[bt]==0)!=65536),end=" ")
            print(".")'''
            #print_gpu_utilization(counter1)
            counter1 += 1
            # Reset the gradients:
            optimizer.zero_grad()
            # Loss gradient descend step:
            it_loss = evaluate(reference, testimg, mask)
            it_loss.backward()
            optimizer.step()
            # Track metrics:
            epoch_loss += it_loss.to("cpu").detach().numpy()  #changes from ".to("cpu")" to ".to(device)"
            ### end of iteration for epoch ###

        epoch_loss /= len(dataset_train)
        #print((cls1+cls0)/cls0)
        #print((cls1+cls0)/cls1)
        #print(cls0,cls1)

        #########
        t1 = time.time()
        print(f"Training phase summary(Time taken {(t1-t0)/60} minutes ")
        print("Loss for epoch {} is {}".format(epc, epoch_loss))
        writer.add_scalar("Loss/epoch", epoch_loss, epc)
        scores_dictionary = tool4metric.get_scores()
        #run[npt_logger.base_namespace]["batch/loss"].append(epoch_loss)
        #run[npt_logger.base_namespace]["batch/F1_1"].append(scores_dictionary["F1_1"])
        #run[npt_logger.base_namespace]["batch/iou_1"].append(scores_dictionary["iou_1"])
        metric_arr[0][epc] = epoch_loss
        metric_arr[1][epc] = scores_dictionary["F1_1"]
        writer.add_scalar("IoU class change/epoch",
                          scores_dictionary["iou_1"], epc)
        writer.add_scalar("F1 class change/epoch",
                          scores_dictionary["F1_1"], epc)
        print(
            "IoU class change for epoch {} is {}".format(
                epc, scores_dictionary["iou_1"]
            )
        )
        print(
            "F1 class change for epoch {} is {}".format(
                epc, scores_dictionary["F1_1"])
        )
        print()
        writer.flush()

        ### Save the model ###
        if epc % save_after == 0:
            torch.save(
                model.state_dict(), os.path.join(logpath, "model_{}.pth".format(epc))
            )

    def validation_phase(epc):
        model.eval()
        epoch_loss_eval = 0.0
        tool4metric.clear()
        with torch.no_grad():
            for (reference, testimg), mask in dataset_val:
                '''for bt in range(8):
                    print((np.count_nonzero(mask[bt]==0)!=65536),end=" ")
                print(".")'''
                epoch_loss_eval += evaluate(reference,
                                            testimg, mask).to("cpu").numpy()  

        epoch_loss_eval /= len(dataset_val)
        print("Validation phase summary")
        print("Loss for epoch {} is {}".format(epc, epoch_loss_eval))
        writer.add_scalar("Loss_val/epoch", epoch_loss_eval, epc)
        scores_dictionary = tool4metric.get_scores()
        #run[npt_logger.base_namespace]["vbatch/loss"].append(epoch_loss_eval)
        #run[npt_logger.base_namespace]["vbatch/F1_1"].append(scores_dictionary["F1_1"])
        #run[npt_logger.base_namespace]["vbatch/iou_1"].append(scores_dictionary["iou_1"])
        metric_arr[2][epc] = epoch_loss_eval
        metric_arr[3][epc] = scores_dictionary["F1_1"]
        writer.add_scalar("IoU_val class change/epoch",
                          scores_dictionary["iou_1"], epc)
        writer.add_scalar("F1_val class change/epoch",
                          scores_dictionary["F1_1"], epc)
        print(
            "IoU class change for epoch {} is {}".format(
                epc, scores_dictionary["iou_1"]
            )
        )
        print(
            "F1 class change for epoch {} is {}".format(
                epc, scores_dictionary["F1_1"])
        )
        return epoch_loss_eval

    for epc in range(epochs):
        training_phase(epc)
        print_gpu_utilization(epc)
        #print(logpath)
        valloss1 = validation_phase(epc)
        np.save(f"{logpath}graph.npy",metric_arr)
        # scheduler step
        scheduler.step(valloss1)


def run():

    # set the random seed
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # Parse arguments:
    args = parse_arguments()

    # Initialize tensorboard:
    writer = SummaryWriter(log_dir=args.log_path)
    print(f"LOGS are saved in  {args.log_path} directory")

    # Inizialitazion of dataset and dataloader:
    trainingdata = dtset.MyDataset(args.datapath, "train")
    images_list_file = join(args.datapath,'list', "train" + ".txt")
    with open(images_list_file, "r") as f:
        imglst =  f.readlines()
    #imgname = imglst.strip('\n')
    print(len(imglst))
    countp, countn, ptr1 = 0, 0, 0
    wtarr = [0]*len(imglst)
    print(args.datapath)
    for idx, mask in enumerate(trainingdata):
        _, imgtmp = mask
        if np.count_nonzero(imgtmp > 0):
            countp += 1
            wtarr[ptr1] = 1
        else:
            countn += 1
        ptr1 += 1
        
    for i in range(len(imglst)):
        if wtarr[i] == 0:
            wtarr[i] = 1000/countn
            #print(wtarr[i])
        else:
            wtarr[i] = 1000/countp
    sampler1 = WeightedRandomSampler(wtarr, len(wtarr), replacement=True)
    validationdata = dtset.MyDataset(args.datapath, "val")
    
    images_list_file1 = join(args.datapath,'list', "val" + ".txt")
    with open(images_list_file1, "r") as f:
        imglst1 =  f.readlines()
    #imgname = imglst.strip('\n')
    print(len(imglst1))
    countp1, countn1, ptr11 = 0, 0, 0
    wtarr1 = [0]*len(imglst1)
    print(args.datapath)
    for idx, mask in enumerate(validationdata):
        _, imgtmp = mask
        if np.count_nonzero(imgtmp > 0):
            countp1 += 1
            wtarr1[ptr11] = 1
        else:
            countn1 += 1
        ptr11 += 1
        
    for i in range(len(imglst1)):
        if wtarr1[i] == 0:
            wtarr1[i] = 1000/countn1
            #print(wtarr[i])
        else:
            wtarr1[i] = 1000/countp1
    sampler2 = WeightedRandomSampler(wtarr1, len(wtarr1), replacement=True)
    
    data_loader_training = DataLoader(trainingdata, batch_size=12,sampler=sampler1,num_workers = 8, pin_memory=True)
    data_loader_val = DataLoader(validationdata, batch_size=12,sampler=sampler2,num_workers = 8, pin_memory=True)
    # device setting for training
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
    else:
        device = torch.device('cpu')

    print(f'Current Device: {device}\n')

    # Initialize the model
    model = Model().to(device)    #changed by "" to ".to(device)"
    restart_from_checkpoint = True
    model_path = "/model"
    #model_path = "/home/user/Desktop/ind/m4.pth"
    if restart_from_checkpoint:
        model.load_state_dict(torch.load(model_path,map_location=device))
        print("Checkpoint succesfully loaded")

    # print number of parameters
    parameters_tot = 0
    for nom, param in model.named_parameters():
        # print (nom, param.data.shape)
        parameters_tot += torch.prod(torch.tensor(param.data.shape))
    print("Number of model parameters {}\n".format(parameters_tot))

    # define the loss function for the model training.
    #criterion = torch.nn.BCELoss()
    #class_weights = [1.0220641944459663,46.32229818989909]
    #class_weights = torch.FloatTensor(class_weights)
    #criterion = FocalLoss(alpha=class_weights, gamma=2)
    criterion = FocalLoss()

    # choose the optimizer in view of the used dataset
    # Optimizer with tuned parameters for LEVIR-CD
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00256799066427741,
                                  weight_decay=0.009449677083344786, amsgrad=False)
    # scheduler for the lr of the optimizer
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.9,patience=5)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #    optimizer, T_max=100)

    # copy the configurations
    _ = shutil.copytree(
        "./models",
        os.path.join(args.log_path, "models"),
    )

    train(
        data_loader_training,
        data_loader_val,
        model,
        criterion,
        optimizer,
        scheduler,
        args.log_path,
        writer,
        epochs=100,
        save_after=1,
        device=device
    )
    writer.close()


if __name__ == "__main__":
    run()
