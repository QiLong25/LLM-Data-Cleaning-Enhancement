import sys

import torch
import numpy as np


def embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model,vision_dset=False):
    device = x_cont.device
    x_categ = x_categ + model.categories_offset.type_as(x_categ)
    x_categ_enc = model.embeds(x_categ)
    n1,n2 = x_cont.shape
    _, n3 = x_categ.shape
    if model.cont_embeddings == 'MLP':
        x_cont_enc = torch.empty(n1,n2, model.dim)
        for i in range(model.num_continuous):
            x_cont_enc[:,i,:] = model.simple_MLP[i](x_cont[:,i])
    else:
        raise Exception('This case should not work!')    


    x_cont_enc = x_cont_enc.to(device)
    cat_mask_temp = cat_mask + model.cat_mask_offset.type_as(cat_mask)
    con_mask_temp = con_mask + model.con_mask_offset.type_as(con_mask)


    cat_mask_temp = model.mask_embeds_cat(cat_mask_temp)
    con_mask_temp = model.mask_embeds_cont(con_mask_temp)
    x_categ_enc[cat_mask == 0] = cat_mask_temp[cat_mask == 0]
    x_cont_enc[con_mask == 0] = con_mask_temp[con_mask == 0]

    if vision_dset:
        
        pos = np.tile(np.arange(x_categ.shape[-1]),(x_categ.shape[0],1))
        pos =  torch.from_numpy(pos).to(device)
        pos_enc =model.pos_encodings(pos)
        x_categ_enc+=pos_enc

    return x_categ, x_categ_enc, x_cont_enc



## Modify to return mix items !!!!!!!!!!!!!!!!
def mixup_data(x1, x2 , lam=1.0, y= None, use_cuda=True):
    '''Returns mixed inputs, pairs of targets'''

    ## LQ: each of row data is represented as (33 * 16), a batch is 256, so x1, x2 is (256 * 33 * 16), which means 256 row data in total, each is (33 * 16)

    batch_size = x1.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()           ## LQ: shuffle 256, so each row data is mapped to another row, possible to be itself
    else:
        index = torch.randperm(batch_size)

    mixed_x1 = lam * x1 + (1 - lam) * x1[index, :]           ## LQ: for each row data in 256 rows, itself * lam + mapping * (1-lam) and add in each dimension (33*16)
    mixed_x2 = lam * x2 + (1 - lam) * x2[index, :]
    if y is not None:
        y_a, y_b = y, y[index]
        return mixed_x1, mixed_x2, y_a, y_b
    
    return mixed_x1, mixed_x2, index


def add_noise(x_categ,x_cont, noise_params = {'noise_type' : ['cutmix'],'lambda' : 0.1}):

    ## LQ: each of row data is represented as (33 * 16), a batch is 256, so x1, x2 is (256 * 33 * 16), which means 256 row data in total, each is (33 * 16)

    lam = noise_params['lambda']        ## LQ: default to 0.1
    device = x_categ.device
    batch_size = x_categ.size()[0]

    if 'cutmix' in noise_params['noise_type']:
        index = torch.randperm(batch_size)              ## LQ: shuffle 256, so each row data is mapped to another row, possible to be itself

        cat_corr = torch.from_numpy(np.random.choice(2,(x_categ.shape),p=[lam,1-lam])).to(device)               ## LQ: pick some locations to shift, x_categ is (256 * 33), p is probability for 0
        con_corr = torch.from_numpy(np.random.choice(2,(x_cont.shape),p=[lam,1-lam])).to(device)
        x1, x2 =  x_categ[index,:], x_cont[index,:]                     ## LQ: x1, x2 copy x_categ, x_cont from mapping index
        x_categ_corr, x_cont_corr = x_categ.clone().detach() ,x_cont.clone().detach()           ## LQ: x_categ_corr, x_cont_corr shallow copy x_categ, x_cont, will change if x_categ or x_cont change
        x_categ_corr[cat_corr==0] = x1[cat_corr==0]                     ## LQ: shift at locations with 0
        x_cont_corr[con_corr==0] = x2[con_corr==0]
        return x_categ_corr, x_cont_corr, index
    elif noise_params['noise_type'] == 'missing':
        x_categ_mask = np.random.choice(2,(x_categ.shape),p=[lam,1-lam])
        x_cont_mask = np.random.choice(2,(x_cont.shape),p=[lam,1-lam])
        x_categ_mask = torch.from_numpy(x_categ_mask).to(device)
        x_cont_mask = torch.from_numpy(x_cont_mask).to(device)
        return torch.mul(x_categ,x_categ_mask), torch.mul(x_cont,x_cont_mask)
        
    else:
        print("yet to write this")
