import model
import dataloader
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import random
import cv2
import wandb
import warnings
import matplotlib.pyplot as plt
from torchvision import transforms
warnings.filterwarnings("ignore")

class SegmentationMetrics:
    def __init__(self, n_classes, device, ignore=None):
        self.n_classes = n_classes
        self.device = device
        self.ignore = torch.tensor(ignore).long()
        self.include = torch.tensor(
            [n for n in range(self.n_classes) if n not in self.ignore]
        ).long()

    def reset(self):
        self.confusion_matrix = torch.zeros(
            (self.n_classes, self.n_classes), device=self.device
        ).long()
        self.ones = None
        self.last_scan_size = None

    def addbatch(self, preds, targets):
        preds_row = preds.reshape(-1)
        targets_row = targets.reshape(-1)
        indices = torch.stack([preds_row, targets_row], dim=0)
        if self.ones is None or self.last_scan_size != indices.shape[-1]:
            self.ones = torch.ones((indices.shape[-1]), device=self.device).long()
            self.last_scan_size = indices.shape[-1]
        self.confusion_matrix = self.confusion_matrix.index_put_(
            tuple(indices), self.ones, accumulate=True
        )

    def getstats(self):
        confusion_matrix = self.confusion_matrix.clone()
        confusion_matrix[self.ignore] = 0
        confusion_matrix[:, self.ignore] = 0
        true_pos = confusion_matrix.diag()
        false_pos = confusion_matrix.sum(dim=1) - true_pos
        false_neg = confusion_matrix.sum(dim=0) - true_pos
        return true_pos, false_pos, false_neg

    def getiou(self):
        true_pos, false_pos, false_neg = self.getstats()
        intersection = true_pos
        union = true_pos + false_pos + false_neg + 1e-15
        iou = intersection / union
        iou_mean = (intersection[self.include] / union[self.include]).mean()
        return iou_mean, iou

    def getacc(self):
        true_pos, false_pos, false_neg = self.getstats()
        total_truepos = true_pos.sum()
        total = true_pos[self.include].sum() + false_pos[self.include].sum() + 1e-15
        accuracy_mean = total_truepos / total
        return accuracy_mean

class DiceLoss(nn.Module):
    def __init__(self, weights=None, size_average=True):
        super(DiceLoss,self).__init__()

    def forward(self, inputs, targets, smooth=1e-5):
        batch_size = inputs.size(0)
        targets = torch.eye(5)[targets]
        targets = targets.permute(0,3,1,2)
        targets = targets.cuda()

        # defect_pixels = torch.tensor([295944.,63358.,8862815.,1830843.]) #calculated the number of defect pixels in target before hand

        class_weights = torch.tensor([0.,1.,1.,1.,1.])
        class_weights = class_weights.cuda()

        intersection = (torch.sum(inputs*targets, (0,2,3)))*class_weights
        denominator = (torch.sum(inputs+targets, (0,2,3)))*class_weights
        dice = (2*intersection/(denominator + 1e-5)).mean()

        return (1 - dice)

model = model.Model(3,5).cuda()

train_dataset = dataloader.Defect_Dataset("severstal-steel-defect-detection", "train", 64, 400)
val_dataset = dataloader.Defect_Dataset("severstal-steel-defect-detection", "val", 64, 400)
trainloader = DataLoader(train_dataset, 8, shuffle=True, num_workers=6)
valloader = DataLoader(val_dataset, 8, shuffle=False, num_workers=6)

nll_loss = nn.NLLLoss()
dice_loss = DiceLoss()
optimizer = optim.SGD(model.parameters(),lr=0.01,momentum=0.9,weight_decay=0.0001)
evaluator = SegmentationMetrics(5,"cuda",0)

wandb.init(project="Steel_Defect_Detection")

train_step = 0
best = 0
for epoch in range(50):
    loss_cntr = []
    evaluator.reset()
    for (img,mask) in tqdm(trainloader):
        img,mask = img.cuda(),mask.cuda()
        output = model(img)
        loss = nll_loss(torch.log(output.clamp(min=1e-8)), mask.long()) + dice_loss(output, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred_label = output.argmax(dim=1)
        evaluator.addbatch(pred_label,mask.long())
        loss_cntr.append(loss.item())

        wandb.log({"loss": loss.item(), "train step": train_step})
        train_step += 1

    acc = evaluator.getacc()
    iou, c_iou = evaluator.getiou()
    wandb.log({"train_acc": acc.item(), "train_iou": iou.item(), "training_epoch": epoch})
    print(f"train epoch: {epoch+1} loss: {round(np.mean(loss_cntr),3)}, iou: {round(iou.item(),3)}, class iou: {[round(c,3) for c in c_iou.detach().cpu().numpy()[1:]]}\n")     

    if epoch % 2 == 0:
        n_vis = 0
        evaluator.reset()
        for (img,mask) in tqdm(valloader):
            img,mask = img.cuda(),mask.cuda()
            with torch.no_grad():
                out = model(img)
            pred_label = out.argmax(dim=1)
            evaluator.addbatch(pred_label,mask.long())
            if epoch % 10 == 0 and n_vis < 5 and random.random() < 0.1:
                ind = random.randint(0, img.shape[0]-1)

                invTrans = transforms.Compose([ transforms.Normalize(mean = [ -2.259, -2.259, -2.259 ],
                                                        std = [ 6.6401, 6.6401, 6.6401 ])])

                im = invTrans(img[ind])
                im = im.detach().cpu().numpy().transpose(1,2,0)
                pred = pred_label[ind].detach().cpu().numpy().astype(np.float32)
                one_hot = torch.eye(5)[pred]
                one_hot = one_hot.permute(2,0,1)
                one_hot = one_hot.detach().cpu().numpy()
                total_mask = np.zeros([64,400,3],dtype=int)

                total_mask[:,:,0] += (one_hot[1]*255).astype(int) # Defect1: Red
                total_mask[:,:,1] += (one_hot[2]*255).astype(int) # Defect2: Green
                total_mask[:,:,2] += (one_hot[3]*255).astype(int) # Defect3: Blue
                total_mask[:,:,0] += (one_hot[4]*252).astype(int) # Defect4: Orange
                total_mask[:,:,1] += (one_hot[4]*157).astype(int)
                total_mask[:,:,2] += (one_hot[4]*3).astype(int)
                total_mask[:,:,0] += (one_hot[0]*255).astype(int) # No Defect: Yellow
                total_mask[:,:,1] += (one_hot[0]*255).astype(int)

                one_hot_mask = torch.eye(5)[mask[ind]]
                one_hot_mask = one_hot_mask.permute(2,0,1)
                one_hot_mask = one_hot_mask.detach().cpu().numpy()
                total_gt = np.zeros([64,400,3],dtype=int)

                total_gt[:,:,0] += (one_hot_mask[1]*255).astype(int) # Defect1: Red
                total_gt[:,:,1] += (one_hot_mask[2]*255).astype(int) # Defect2: Green
                total_gt[:,:,2] += (one_hot_mask[3]*255).astype(int) # Defect3: Blue
                total_gt[:,:,0] += (one_hot_mask[4]*252).astype(int) # Defect4: Orange
                total_gt[:,:,1] += (one_hot_mask[4]*157).astype(int)
                total_gt[:,:,2] += (one_hot_mask[4]*3).astype(int)
                total_gt[:,:,0] += (one_hot_mask[0]*255).astype(int) # No Defect: Yellow
                total_gt[:,:,1] += (one_hot_mask[0]*255).astype(int)

                wandb.log({f"{n_vis}": [wandb.Image(np.hstack((total_gt,im*255,total_mask)), caption=f"{epoch}")]})
                n_vis += 1

        acc = evaluator.getacc()
        iou, c_iou = evaluator.getiou()
        wandb.log({"val_acc": acc.item(), "val_iou": iou.item(), "val_epoch": epoch})
        print(f"val epoch: {epoch+1}, iou: {round(iou.item(),3)}, class iou: {[round(c,3) for c in c_iou.detach().cpu().numpy()[1:]]}\n")
        if iou.item() > best:
            torch.save(model.state_dict(), "best.ckpt")
            best = iou.item()

    torch.save(model.state_dict(), "last.ckpt")
