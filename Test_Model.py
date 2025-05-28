import os
import cv2
import torch
import config_test
from utils.loader import CustomDataset
from monai.data import ( DataLoader, CacheDataset, load_decathlon_datalist, decollate_batch,)
from tqdm import tqdm
from model.Network import UIUNET
from model.Network import U_Net, R2U_Net, AttU_Net, R2AttU_Net
from utils.misc import overlay, save_mat

test_dataset = CustomDataset(config_test.TEST_FILENAME)
test_ds = CacheDataset(test_dataset, num_workers=0, cache_rate=0.5)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epoch_iterator = tqdm(test_loader, desc="Testing (X / X Steps) (loss=X.X)", dynamic_ncols=True)
if config_test.TRAIN_MODEL=="UIUNET":
    model = UIUNET(3, 1).to(device)
    print(f"loading UIUNET")
elif config_test.TRAIN_MODEL=="U_Net":
    model = U_Net(3, 1).to(device)
    print(f"loading U_Net")
elif config_test.TRAIN_MODEL=="R2U_Net":
    model = R2U_Net(3, 1).to(device)
    print(f"loading R2U_Net")
elif config_test.TRAIN_MODEL == "AttU_Net":
    model = AttU_Net(3, 1).to(device)
    print(f"loading AttU_Net")
elif config_test.TRAIN_MODEL == "R2AttU_Net":
    model = R2AttU_Net(3, 1).to(device)
    print(f"loading R2AttU_Net")
else:
    print(f"No models were matched that were available for training")
# model = UIUNET(3, 1).to(device)
# model = U_Net(3, 1).to(device)
torch.backends.cudnn.benchmark = True
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn
def clr_2_bw(image, threshold_value):
    (_, blackAndWhiteImage) = cv2.threshold(image, threshold_value, 1, cv2.THRESH_BINARY)
    return blackAndWhiteImage

model.load_state_dict(torch.load(config_test.BEST_MODEL, map_location=device),strict=False)
model.eval()
i = 0
with torch.no_grad():
    for batch in epoch_iterator:
        i +=1
        img = batch["image"].type(torch.FloatTensor)
        label = batch["label"].type(torch.FloatTensor)
        if config_test.TRAIN_MODEL == "UIUNET":
             d0, d1, d2, d3, d4, d5, d6 = model(img.cuda())
        else:
             d1 = model(img.cuda())
        #if config_test.TRAIN_MODEL == "UIUNET":
        #    d0, d1, d2, d3, d4, d5, d6 = model(img.cuda())
        #else:
        #    d0 = model(img.cuda())
        pred = d1[:, 0, :, :]
        pred = normPRED(pred)
        pred = clr_2_bw(pred.permute(1, 2, 0).cpu().numpy(), threshold_value=0.8)
        img = img.squeeze(0).permute(1,2,0).cpu().numpy().squeeze()
        label = label.squeeze(0).permute(1,2,0).cpu().numpy().squeeze()

        overlay_prdctd = overlay(img,pred)
        overlay_lbld = overlay(img, label)
        save_mat(file=overlay_prdctd , i = i, dir = config_test.TEST_DIR, folder_name = "overlay_predictions")
        save_mat(file=overlay_lbld, i=i, dir=config_test.TEST_DIR, folder_name="overlay_labels")
        save_mat(file=img, i=i, dir=config_test.TEST_DIR, folder_name="images")
        save_mat(file=label, i=i, dir=config_test.TEST_DIR, folder_name="labels")
        save_mat(file=pred, i=i, dir=config_test.TEST_DIR, folder_name="predictions")
    print("Testing completed")


