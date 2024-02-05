import sys

import torch
from torch import nn
from data_openml import data_prep_openml,task_dset_ids,DataSetCatCon
from torch.utils.data import DataLoader
import torch.optim as optim
from augmentations import embed_data_mask
from augmentations import add_noise

import os
import numpy as np

def SAINT_pretrain(model,cat_idxs,X_train,y_train,continuous_mean_std,opt,device,raha_detect_loader):
    train_ds = DataSetCatCon(X_train, y_train, cat_idxs,opt.dtask, continuous_mean_std)
    trainloader = DataLoader(train_ds, batch_size=opt.batchsize, shuffle=False,num_workers=4)
    vision_dset = opt.vision_dset
    optimizer = optim.AdamW(model.parameters(),lr=0.0001)
    pt_aug_dict = {
        'noise_type' : opt.pt_aug,
        'lambda' : opt.pt_aug_lam
    }
    criterion1 = nn.CrossEntropyLoss()          ## 交叉熵
    criterion2 = nn.MSELoss()               ## 均方误差
    print("Pretraining begins!")
    for epoch in range(opt.pretrain_epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):           ## LQ: get batch
            optimizer.zero_grad()
            x_categ, x_cont, _ ,cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device),data[4].to(device)
            rahaDetect = raha_detect_loader[i]
            batchSize = len(x_categ)

            # embed_data_mask function is used to embed both categorical and continuous data.

            ## LQ: change X a little bit (modify to do both processing)
            if 'cutmix' in opt.pt_aug:
                from augmentations import add_noise
                from augmentations import mixup_data
                ## LQ: modify to record mapping!!!!!!!!!!!!!!!
                x_categ_corr, x_cont_corr, cutmix_mapping = add_noise(x_categ,x_cont, noise_params = pt_aug_dict)
                cutmix_mapping = cutmix_mapping.long()

                ## LQ: Embedding
                _ , x_categ_enc_2, x_cont_enc_2 = embed_data_mask(x_categ_corr, x_cont_corr, cat_mask, con_mask,model,vision_dset)
                ## LQ: modify to record mapping!!!!!!!!!!!!!!!
                x_categ_enc_2, x_cont_enc_2, mixup_mapping = mixup_data(x_categ_enc_2, x_cont_enc_2, lam=opt.mixup_lam)
                mixup_mapping = mixup_mapping.long()

            else:
                _ , x_categ_enc_2, x_cont_enc_2 = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model,vision_dset)
            _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model,vision_dset)

            ## LQ: change embedding a little bit
            if 'mixup' in opt.pt_aug:
                from augmentations import mixup_data
                x_categ_enc_2, x_cont_enc_2 = mixup_data(x_categ_enc_2, x_cont_enc_2 , lam=opt.mixup_lam)

            loss = 0
            if 'contrastive' in opt.pt_tasks:
                aug_features_1  = model.transformer(x_categ_enc, x_cont_enc)            ## LQ: no mix up
                aug_features_2 = model.transformer(x_categ_enc_2, x_cont_enc_2)         ## LQ: mix up
                aug_features_1 = (aug_features_1 / aug_features_1.norm(dim=-1, keepdim=True)).flatten(1,2)
                aug_features_2 = (aug_features_2 / aug_features_2.norm(dim=-1, keepdim=True)).flatten(1,2)

                ## LQ: go through projection heads MLP
                if opt.pt_projhead_style == 'diff':
                    aug_features_1 = model.pt_mlp(aug_features_1)
                    aug_features_2 = model.pt_mlp2(aug_features_2)
                elif opt.pt_projhead_style == 'same':
                    aug_features_1 = model.pt_mlp(aug_features_1)
                    aug_features_2 = model.pt_mlp(aug_features_2)
                else:
                    print('Not using projection head')

                ## LQ: predict each representation as from which X, output is (256 * 256), which means for each representation, the probability of from each X
                logits_per_aug1 = aug_features_1 @ aug_features_2.t()/opt.nce_temp
                logits_per_aug2 =  aug_features_2 @ aug_features_1.t()/opt.nce_temp
                targets = torch.arange(logits_per_aug1.size(0)).to(logits_per_aug1.device)          ## LQ: target is just 0-0, 1-1, 2-2, 3-3...

                ## LQ: target is modified to get close to clean

                ## LQ: New Model
                for row in range(len(targets)):
                    if rahaDetect[row] == 1:            ## LQ: original X is dirty
                        if rahaDetect[cutmix_mapping[row]] == 0:            ## LQ: original X is mixed with clean X'
                            targets[row] = cutmix_mapping[row]
                        else:
                            if rahaDetect[mixup_mapping[row]] == 0:             ## LQ: embedding E is mixed with clean E'
                                targets[row] = mixup_mapping[row]

                ## LQ: weights are added to change loss calculation here!!!!!!!!!!!!!!!!!!
                weight = torch.ones(len(x_categ), dtype=torch.float)

                ## LQ: New Model
                for row in range(len(logits_per_aug1)):
                    if rahaDetect[row] == 1 and rahaDetect[cutmix_mapping[row]] == 1 and rahaDetect[mixup_mapping[row]] == 1:                ## LQ: all is dirty
                        weight[row] = 0.5
                    elif rahaDetect[row] == 0 and rahaDetect[cutmix_mapping[row]] == 0 and rahaDetect[mixup_mapping[row]] == 0:                ## LQ: all is clean
                        weight[row] = 1
                    else:
                        if rahaDetect[row] == 1:            ## LQ: original X is dirty
                            if rahaDetect[cutmix_mapping[row]] == 0:            ## LQ: original X is mixed with a clean X'
                                weight[row] = weight[row] * 2
                            if rahaDetect[mixup_mapping[row]] == 0:             ## LQ: embedding E is mixed with a clean embedding E'
                                weight[row] = weight[row] * 2
                        else:               ## LQ: original X is clean
                            if rahaDetect[cutmix_mapping[row]] == 1:            ## LQ: original X is mixed with a dirty X'
                                weight[row] = weight[row] * 2
                            if rahaDetect[mixup_mapping[row]] == 1:             ## LQ: embedding E is mixed with a dirty embedding E'
                                weight[row] = weight[row] * 2
                weight = weight / torch.sum(weight) * batchSize

                weight.to(logits_per_aug1.device)

                criterion11 = nn.CrossEntropyLoss(weight=weight)
                criterion11.to(logits_per_aug1.device)
                loss_1 = criterion11(logits_per_aug1, targets)
                loss_2 = criterion11(logits_per_aug2, targets)
                loss   = opt.lam0*(loss_1 + loss_2)/2           ## LQ: normalize
                # print("contrastive loss: ", loss)

            elif 'contrastive_sim' in opt.pt_tasks:
                aug_features_1  = model.transformer(x_categ_enc, x_cont_enc)            ## LQ: embedding no mix up, size(256 * 39 * 16)
                aug_features_2 = model.transformer(x_categ_enc_2, x_cont_enc_2)         ## LQ: embedding mixed up(256 * 39 * 16)
                aug_features_1 = (aug_features_1 / aug_features_1.norm(dim=-1, keepdim=True)).flatten(1,2)      ## LQ: flatten to size(256 * 624), every row data is a 1d vector
                aug_features_2 = (aug_features_2 / aug_features_2.norm(dim=-1, keepdim=True)).flatten(1,2)
                aug_features_1 = model.pt_mlp(aug_features_1)           ## LQ: pass through MLP
                aug_features_2 = model.pt_mlp2(aug_features_2)
                c1 = aug_features_1 @ aug_features_2.t()                ## LQ: predict each representation as from which X, output is (256 * 256), which means for each representation, the probability of from each X
                loss+= opt.lam1*torch.diagonal(-1*c1).add_(1).pow_(2).sum()
            if 'denoising' in opt.pt_tasks:
                cat_outs, con_outs = model(x_categ_enc_2, x_cont_enc_2)         ## LQ: model is full SAINT, including downstream MLPs, output has dimension conti / categ cols, each dim 256 channels
                # if con_outs.shape(-1) != 0:
                # import ipdb; ipdb.set_trace()
                if len(con_outs) > 0:               ## LQ: at least one col is continuous
                    con_outs =  torch.cat(con_outs,dim=1)           ## LQ: shape (256 * 2)

                    ## LQ: New Model
                    colNum = con_outs.size()[1]
                    toDelete = []
                    for i in range(batchSize):
                        toDelete.append(0)
                    for row in range(batchSize):
                        if rahaDetect[row] == 1 and rahaDetect[cutmix_mapping[row]] == 1 and rahaDetect[mixup_mapping[row]] == 1:  ## LQ: all is dirty
                            continue
                        elif rahaDetect[row] == 0 and rahaDetect[cutmix_mapping[row]] == 0 and rahaDetect[mixup_mapping[row]] == 0:  ## LQ: all is clean
                            continue
                        else:
                            if rahaDetect[row] == 1:  ## LQ: original X is dirty, mixed with clean
                                toDelete[row] = 1
                            else:  ## LQ: original X is clean
                                if rahaDetect[cutmix_mapping[row]] == 1:  ## LQ: original X is mixed with a dirty X'
                                    con_outs = torch.cat((con_outs, torch.reshape(con_outs[row], (1, colNum))), 0)
                                    x_cont = torch.cat((x_cont, torch.reshape(x_cont[row], (1, colNum))), 0)
                                if rahaDetect[mixup_mapping[row]] == 1:  ## LQ: embedding E is mixed with a dirty embedding E'
                                    con_outs = torch.cat((con_outs, torch.reshape(con_outs[row], (1, colNum))), 0)
                                    x_cont = torch.cat((x_cont, torch.reshape(x_cont[row], (1, colNum))), 0)
                    rowNum = batchSize
                    row = 0
                    while row < rowNum:
                        if toDelete[row] == 1:
                            con_outs = torch.cat((torch.reshape(con_outs[0:row], (row, colNum)), torch.reshape(con_outs[row+1:], (len(con_outs[row + 1:]), colNum))), 0)
                            x_cont = torch.cat((torch.reshape(x_cont[0:row], (row, colNum)), torch.reshape(x_cont[row+1:], (len(x_cont[row+1:]), colNum))), 0)
                            row = row - 1
                            rowNum = rowNum - 1
                        row = row + 1

                    l2 = criterion2(con_outs, x_cont)               ## LQ: loss of continuous compared with original X
                else:
                    l2 = 0
                # print("denoise conti: ", l2)
                l1 = 0
                # import ipdb; ipdb.set_trace()
                n_cat = x_categ.shape[-1]               ## LQ: n_cat is number of categ cols, x_categ is (256 * n_cat), which represents for each row, each categ block is predicted

                ## LQ: dirty row are repeated to highlight mix of dirty and clean !!!!!!!!!!!!!!!
                ## LQ: New Model
                toDelete = []
                for i in range(batchSize):
                    toDelete.append(0)

                for row in range(batchSize):
                    if rahaDetect[row] == 1 and rahaDetect[cutmix_mapping[row]] == 1 and rahaDetect[mixup_mapping[row]] == 1:  ## LQ: all is dirty
                        continue
                    elif rahaDetect[row] == 0 and rahaDetect[cutmix_mapping[row]] == 0 and rahaDetect[mixup_mapping[row]] == 0:  ## LQ: all is clean
                        continue
                    else:
                        if rahaDetect[row] == 1:  ## LQ: original X is dirty
                            toDelete[row] = 1
                        else:  ## LQ: original X is clean
                            if rahaDetect[cutmix_mapping[row]] == 1:  ## LQ: original X is mixed with a dirty X'
                                for j in range(1, n_cat):
                                    cat_outs[j] = torch.cat((cat_outs[j], torch.reshape(cat_outs[j][row], (1, cat_outs[j].size()[1]))), 0)
                                x_categ = torch.cat((x_categ, torch.reshape(x_categ[row], (1, n_cat))), 0)
                            if rahaDetect[mixup_mapping[row]] == 1:  ## LQ: embedding E is mixed with a dirty embedding E'
                                for j in range(1, n_cat):
                                    cat_outs[j] = torch.cat((cat_outs[j], torch.reshape(cat_outs[j][row], (1, cat_outs[j].size()[1]))), 0)
                                x_categ = torch.cat((x_categ, torch.reshape(x_categ[row], (1, n_cat))), 0)

                rowNum = batchSize
                row = 0
                while row < rowNum:
                    if toDelete[row] == 1:
                        for j in range(1, n_cat):
                            cat_outs[j] = torch.cat((torch.reshape(cat_outs[j][0:row], (row, cat_outs[j].size()[1])), torch.reshape(cat_outs[j][row+1:], (len(cat_outs[j][row+1:]), cat_outs[j].size()[1]))), 0)
                        x_categ = torch.cat((torch.reshape(x_categ[0:row], (row, n_cat)), torch.reshape(x_categ[row + 1:], (len(x_categ[row + 1:]), n_cat))), 0)
                        row = row - 1
                        rowNum = rowNum - 1
                    row = row + 1

                for j in range(1, n_cat):               ## LQ: loss for each categ col is calculated
                    l1 += criterion1(cat_outs[j], x_categ[:, j])
                # print("denoise categ: ", l1)

                loss += opt.lam2*l1 + opt.lam3*l2       ## LQ: balance between categ and conti
                # print("total loss", loss)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f'Epoch: {epoch}, Running Loss: {running_loss}')

    print('END OF PRETRAINING!')
    return model
        # if opt.active_log:
        #     wandb.log({'pt_epoch': epoch ,'pretrain_epoch_loss': running_loss
        #     })
