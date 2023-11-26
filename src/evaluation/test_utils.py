import torch
from torch.utils.data import Dataset
from torch import nn
import logging
from torch.utils.data import DataLoader
import numpy as np
import faiss

def test(opt, eval_set, model, encoder_dim, device, cuda, writer, epoch=0, write_tboard=False):
    # TODO what if features dont fit in memory? 
    test_data_loader = DataLoader(dataset=eval_set, 
                num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False, 
                pin_memory=cuda)

    model.eval()
    with torch.no_grad():
        print('====> Extracting Features')
        pool_size = encoder_dim
        if opt.pooling.lower() == 'netvlad': pool_size *= opt.num_clusters
        dbFeat = np.empty((len(eval_set), pool_size))

        for iteration, (input, indices) in enumerate(test_data_loader, 1):
            input = input.to(device)
            image_encoding = model.encoder(input)
            vlad_encoding = model.pool(image_encoding) 

            dbFeat[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()
            if iteration % 50 == 0 or len(test_data_loader) <= 10:
                print("==> Batch ({}/{})".format(iteration, 
                    len(test_data_loader)), flush=True)

            del input, image_encoding, vlad_encoding
    del test_data_loader

    # extracted for both db and query, now split in own sets
    qFeat = dbFeat[eval_set.dbStruct.numDb:].astype('float32')
    dbFeat = dbFeat[:eval_set.dbStruct.numDb].astype('float32')
    
    print('====> Building faiss index')
    faiss_index = faiss.IndexFlatL2(pool_size)
    faiss_index.add(dbFeat)

    print('====> Calculating recall @ N')
    n_values = [1,5,10,20]

    _, predictions = faiss_index.search(qFeat, max(n_values)) 

    # for each query get those within threshold distance
    gt = eval_set.getPositives() 

    correct_at_n = np.zeros(len(n_values))
    #TODO can we do this on the matrix in one go?
    for qIx, pred in enumerate(predictions):
        for i,n in enumerate(n_values):
            # if in top N then also in top NN, where NN > N
            if np.any(np.in1d(pred[:n], gt[qIx])):
                correct_at_n[i:] += 1
                break
    recall_at_n = correct_at_n / eval_set.dbStruct.numQ

    recalls = {} #make dict for output
    for i,n in enumerate(n_values):
        recalls[n] = recall_at_n[i]
        print("====> Recall@{}: {:.4f}".format(n, recall_at_n[i]))
        if write_tboard: writer.add_scalar('Val/Recall@' + str(n), recall_at_n[i], epoch)

    return recalls
