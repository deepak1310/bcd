import torch
from dataset.dataset import MyDataset
import tqdm
from torch.utils.data import DataLoader
from metrics.metric_tool import ConfuseMatrixMeter
from models.change_classifier import ChangeClassifier
import argparse
import cv2

def parse_arguments():
    # Argument Parser creation
    parser = argparse.ArgumentParser(
        description="Parameter for data analysis, data cleaning and model training."
    )
    parser.add_argument(
        "--datapath",
        type=str,
        help="data path",
        default="/home/codegoni/aerial/WHU-CD-256/WHU-CD-256",
    )
    parser.add_argument(
        "--modelpath",
        type=str,
        help="model path",
    )

    parsed_arguments = parser.parse_args()
    
    return parsed_arguments

if __name__ == "__main__":

    # Parse arguments:
    args = parse_arguments()

    # tool for metrics
    tool_metric = ConfuseMatrixMeter(n_class=2)

    # Initialisation of the dataset
    data_path = args.datapath 
    dataset = MyDataset(data_path, "test")
    test_loader = DataLoader(dataset, batch_size=1)

    # Initialisation of the model and print model stat
    model = ChangeClassifier()
    modelpath = args.modelpath
    model.load_state_dict(torch.load(modelpath,map_location=torch.device('cpu')))

    # Print the number of model parameters 
    param_tot = sum(p.numel() for p in model.parameters())
    print()
    print("Number of model parameters {}".format(param_tot))
    print()

    # Set evaluation mode and cast the model to the desidered device
    model.eval()
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    model.to(device)

    # loop to evaluate the model and print the metrics
    bce_loss = 0.0
    criterion = torch.nn.BCELoss()
    count = 3000
   	
    with torch.no_grad():
        for (reference, testimg), mask in tqdm.tqdm(test_loader):
            reference = reference.to(device).float()
            testimg = testimg.to(device).float()
            mask = mask.float()

            # pass refence and test in the model
            generated_mask = model(reference, testimg).squeeze(1)
            
            #if count>2000:
            #	break
            
            # compute the loss for the batch and backpropagate
            generated_mask = generated_mask.to("cpu")
            #print(mask.shape,generated_mask.shape)

            bce_loss += criterion(generated_mask, mask)

            ### Update the metric tool
            bin_genmask = (generated_mask > 0.3).numpy()  #changes (orignal = 0.5)
            bin_genmask = bin_genmask.astype(int)
            mask = mask.numpy()
            mask = mask.astype(int)
            #cv2.imwrite(f"/home/user/temporal_datasets/whubcd/splitedgray/sample_results2/{count}act.png",mask[0]*255)
            cv2.imwrite(f"/home/user/Desktop/ind/resulttest1/{count}act.png",mask[0]*255) 
            cv2.imwrite(f"/home/user/Desktop/ind/resulttest1/{count}per.png",bin_genmask[0]*255) 
            tool_metric.update_cm(pr=bin_genmask, gt=mask)
            count += 1

        bce_loss /= len(test_loader)
        print("Test summary")
        print("Loss is {}".format(bce_loss))
        print()

        scores_dictionary = tool_metric.get_scores()
        print(scores_dictionary) 
