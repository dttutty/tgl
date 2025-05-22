import os



import torch
import threading
from sklearn.metrics import average_precision_score, roc_auc_score
from modules import GeneralModel
from utils import   mfgs_to_cuda, prepare_input, get_ids
import torch.distributed as dist

from modules import  MemoryMailbox
local_rank = dist.get_rank()
world_size = dist.get_world_size()
host_rank = world_size - 1
# print('cuda_device:',torch.cuda.current_device())
# print('cuda_visible_devices:',torch.cuda.device_count())
# if local_rank != world_size -1:
#     torch.cuda.set_device(local_rank)
        
class DataPipelineThread(threading.Thread):
    
    def __init__(self, my_lists , stream, pinned_nfeat_buffs, pinned_efeat_buffs, node_feats, edge_feats, mailbox: MemoryMailbox, memory_param):
        super(DataPipelineThread, self).__init__()
        keys = ['mfgs', 'root', 'ts', 'eid', 'block']
        self.my_mfgs, self.my_root, self.my_ts, self.my_eid, self.my_block  = [my_lists[k] for k in keys]
        self.stream = stream
        self.mfgs = None
        self.root = None
        self.ts = None
        self.eid = None
        self.block = None
        self.pinned_nfeat_buffs = pinned_nfeat_buffs
        self.pinned_efeat_buffs = pinned_efeat_buffs
        self.node_feats = node_feats
        self.edge_feats = edge_feats
        self.mailbox = mailbox
        self.memory_param = memory_param

    def run(self):
        with torch.cuda.stream(self.stream):
            # print(local_rank, 'start thread')
            nids, eids = get_ids(self.my_mfgs[0], self.node_feats, self.edge_feats)
            mfgs = mfgs_to_cuda(self.my_mfgs[0])
            prepare_input(mfgs, self.node_feats, self.edge_feats, pinned=True, nfeat_buffs=self.pinned_nfeat_buffs, efeat_buffs=self.pinned_efeat_buffs, nids=nids, eids=eids)
            if self.mailbox is not None:
                self.mailbox.prep_input_mails(mfgs[0], use_pinned_buffers=True)
                if self.memory_param['deliver_to'] == 'neighbors':
                    self.block = self.my_block[0]
            self.mfgs = mfgs
            self.root = self.my_root[0]
            self.ts = self.my_ts[0]
            self.eid = self.my_eid[0]
            # print(local_rank, 'finished')

    def get_stream(self):
        return self.stream

    def get_mfgs(self):
        return self.mfgs
    
    def get_root(self):
        return self.root
    
    def get_ts(self):
        return self.ts

    def get_eid(self):
        return self.eid

    def get_block(self):
        return self.block


from typing import Any, List, Tuple


def scatter(my_lists, multi_lists, name, src):
    dist.scatter_object_list(
        my_lists[name],
        multi_lists[name],
        src=src
    )
    
def run_training_step(prev_thread, mailbox: MemoryMailbox,  memory_param, model: GeneralModel, optimizer, creterion, node_feats, edge_feats, pinned_nfeat_buffs, pinned_efeat_buffs, nccl_group, tot_loss):
    
    names = ['mfgs', 'root', 'ts', 'eid', 'block']
    my_lists = {name: [None] for name in names}
    multi_lists = {name: [None] * world_size for name in names}
    
    scatter(my_lists, multi_lists, 'mfgs', host_rank)
    
    if mailbox is not None:
    # scatter root, ts, eid
        for name in ('root', 'ts', 'eid'):
            scatter(my_lists, multi_lists, name, host_rank)
        if memory_param['deliver_to'] == 'neighbors':
            scatter(my_lists, multi_lists, 'block', host_rank)
    stream = torch.cuda.Stream()
    


    if prev_thread is not None:
        curr_thread = DataPipelineThread(my_lists, stream, pinned_nfeat_buffs, pinned_efeat_buffs, node_feats, edge_feats, mailbox, memory_param)
        curr_thread.start()
        prev_thread, tot_loss = train_from_prefetch(prev_thread, mailbox, memory_param, model, optimizer, creterion, edge_feats, nccl_group, tot_loss, curr_thread)
    else:
        prev_thread = DataPipelineThread(my_lists, stream, pinned_nfeat_buffs, pinned_efeat_buffs, node_feats, edge_feats, mailbox, memory_param)
        prev_thread.start()
    return prev_thread, tot_loss
        
        


def train_from_prefetch( prev_thread, mailbox: MemoryMailbox, memory_param, model: GeneralModel, optimizer, creterion, edge_feats, nccl_group, tot_loss, curr_thread=None):
    prev_thread.join()
    # with torch.cuda.stream(prev_thread.get_stream()):
    mfgs = prev_thread.get_mfgs()
    model.train()
    optimizer.zero_grad()
    pred_pos, pred_neg = model(mfgs)
    loss = creterion(pred_pos, torch.ones_like(pred_pos))
    loss += creterion(pred_neg, torch.zeros_like(pred_neg))
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        tot_loss += float(loss)
    if mailbox is not None:
        with torch.no_grad():
            eid = prev_thread.get_eid()
            mem_edge_feats = edge_feats[eid] if edge_feats is not None else None
            root_nodes = prev_thread.get_root()
            ts = prev_thread.get_ts()
            block = prev_thread.get_block()
            mailbox.update_mailbox(model.module.memory_updater.last_updated_nid, model.module.memory_updater.last_updated_memory, root_nodes, ts, mem_edge_feats, block)
            mailbox.update_memory(model.module.memory_updater.last_updated_nid, model.module.memory_updater.last_updated_memory, root_nodes, model.module.memory_updater.last_updated_ts)
            if memory_param['deliver_to'] == 'neighbors':
                dist.barrier(group=nccl_group)
                if local_rank == 0:
                    mailbox.update_next_mail_pos()
    prev_thread = curr_thread
    return prev_thread, tot_loss
     
def run_evaluation_step(prev_thread, mailbox,  memory_param, model: GeneralModel, optimizer, creterion, edge_feats, pinned_nfeat_buffs, pinned_efeat_buffs, nccl_group, node_feats, tot_loss):
    if prev_thread is not None:
        # finish last training mini-batch
        prev_thread, tot_loss = train_from_prefetch( prev_thread, mailbox,  memory_param, model, optimizer, creterion, edge_feats, nccl_group, tot_loss)
        
    my_mfgs = [None]
    multi_mfgs = [None] * world_size
    dist.scatter_object_list(my_mfgs, multi_mfgs, src=host_rank)
    
    mfgs = mfgs_to_cuda(my_mfgs[0])
    prepare_input(mfgs, node_feats, edge_feats, pinned=True, nfeat_buffs=pinned_nfeat_buffs, efeat_buffs=pinned_efeat_buffs)
    
    model.eval()
    with torch.no_grad():
        if mailbox is not None:
            mailbox.prep_input_mails(mfgs[0])
        pred_pos, pred_neg = model(mfgs)
        if mailbox is not None:
            names = ['root', 'ts', 'eid']
            my_root,   my_ts,   my_eid   = [[None] for _ in names]
            multi_root, multi_ts, multi_eid = [[None] * world_size for _ in names]
            dist.scatter_object_list(my_root, multi_root, src=host_rank)
            dist.scatter_object_list(my_ts, multi_ts, src=host_rank)
            dist.scatter_object_list(my_eid, multi_eid, src=host_rank)
            eid = my_eid[0]
            mem_edge_feats = edge_feats[eid] if edge_feats is not None else None
            root_nodes = my_root[0]
            ts = my_ts[0]
            block = None
            
            if memory_param['deliver_to'] == 'neighbors':
                my_block = [None]
                multi_block = [None] * world_size
                dist.scatter_object_list(my_block, multi_block, src=host_rank)
                block = my_block[0]
            mailbox.update_mailbox(model.module.memory_updater.last_updated_nid, model.module.memory_updater.last_updated_memory, root_nodes, ts, mem_edge_feats, block)
            mailbox.update_memory(model.module.memory_updater.last_updated_nid, model.module.memory_updater.last_updated_memory, root_nodes, model.module.memory_updater.last_updated_ts)
            
            if memory_param['deliver_to'] == 'neighbors':
                dist.barrier(group=nccl_group)
                if local_rank == 0:
                    mailbox.update_next_mail_pos()
                    
        y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu()
        y_true = torch.cat([torch.ones(pred_pos.size(0)), torch.zeros(pred_neg.size(0))], dim=0)
        ap = average_precision_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        dist.gather_object(float(ap), None, dst=host_rank)
        dist.gather_object(float(auc), None, dst=host_rank)
    return prev_thread, tot_loss

def get_pinned_buffers(sample_param, batch_size, node_feats, edge_feats):
    pinned_nfeat_buffs = list()
    pinned_efeat_buffs = list()
    limit = int(batch_size * 3.3)
    if 'neighbor' in sample_param:
        for i in sample_param['neighbor']:
            limit *= i + 1
            if edge_feats is not None:
                for _ in range(sample_param['history']):
                    pinned_efeat_buffs.insert(0, torch.zeros((limit, edge_feats.shape[1]), pin_memory=True))
    if node_feats is not None:
        for _ in range(sample_param['history']):
            pinned_nfeat_buffs.insert(0, torch.zeros((limit, node_feats.shape[1]), pin_memory=True))
    return pinned_nfeat_buffs, pinned_efeat_buffs

# GPU worker process
def run_worker(sample_param, memory_param, gnn_param, train_param, nccl_group, mailbox, dim_feats, path_saver, node_feats, edge_feats):
    model = GeneralModel(dim_feats[1], dim_feats[4], sample_param, memory_param, gnn_param, train_param).cuda()
    find_unused_parameters = True if sample_param['history'] > 1 else False
    
    # ddp model
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], process_group=nccl_group, output_device=local_rank, find_unused_parameters=find_unused_parameters)
    creterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_param['lr'])
    
    pinned_nfeat_buffs, pinned_efeat_buffs = get_pinned_buffers(sample_param, train_param['batch_size'], node_feats, edge_feats)
    if mailbox is not None:
        mailbox.allocate_pinned_memory_buffers(sample_param, train_param['batch_size'])
    
    tot_loss = 0
    prev_thread = None
    while True:
        state_buf = torch.empty(1, dtype=torch.int32, device='cpu')
        dist.broadcast(state_buf, src=host_rank)
        match int(state_buf.item()):
            case -1:
                # exit
                break
            case 0:
                prev_thread, tot_loss = run_training_step(prev_thread, mailbox,  memory_param, model, optimizer, creterion, node_feats, edge_feats, pinned_nfeat_buffs, pinned_efeat_buffs, nccl_group, tot_loss)
            case 1:
                prev_thread, tot_loss = run_evaluation_step(prev_thread, mailbox, memory_param, model, optimizer, creterion, edge_feats, pinned_nfeat_buffs, pinned_efeat_buffs, nccl_group, node_feats, tot_loss)
            case 2:
                # save model
                if local_rank == 0:
                    torch.save(model.state_dict(), path_saver)
            case 3:
                # load model
                model.load_state_dict(torch.load(path_saver, map_location=torch.device('cuda:0')))
            case 4:
                pass
            case 5:
                # reduce loss
                dist.gather_object(float(tot_loss), None, dst=host_rank)
                tot_loss = 0
            case _:
                raise ValueError(f"Unhandled state: {int(state_buf.item())}")
