import torch
import os
import yaml
import dgl
import time
import pandas as pd
import numpy as np

def get_ids(mfgs, node_feats, edge_feats):
    nids = list()
    eids = list()
    if node_feats is not None:
        for b in mfgs[0]:
            nids.append(b.srcdata['ID'].long())
    if 'ID' in mfgs[0][0].edata:
        if edge_feats is not None:
            for mfg in mfgs:
                for b in mfg:
                    eids.append(b.edata['ID'].long())
    else:
        eids = None
    return nids, eids


def load_feat(d, rand_de=0, rand_dn=0):
    node_feats = None
    if os.path.exists('DATA/{}/node_features.pt'.format(d)):
        node_feats = torch.load('DATA/{}/node_features.pt'.format(d))
        if node_feats.dtype == torch.bool:
            node_feats = node_feats.type(torch.float32)
    edge_feats = None
    if os.path.exists('DATA/{}/edge_features.pt'.format(d)):
        edge_feats = torch.load('DATA/{}/edge_features.pt'.format(d))
        if edge_feats.dtype == torch.bool:
            edge_feats = edge_feats.type(torch.float32)
    if rand_de > 0:
        if d == 'LASTFM':
            edge_feats = torch.randn(1293103, rand_de)
        elif d == 'MOOC':
            edge_feats = torch.randn(411749, rand_de)
    if rand_dn > 0:
        if d == 'LASTFM':
            node_feats = torch.randn(1980, rand_dn)
        elif d == 'MOOC':
            edge_feats = torch.randn(7144, rand_dn)
    return node_feats, edge_feats

def load_graph(d):
    df = pd.read_csv('DATA/{}/edges.csv'.format(d))
    g = np.load('DATA/{}/ext_full.npz'.format(d))
    return g, df

from sampler import SampleParams
from memorys import MemoryParams
from modules import GNNParams, TrainParams
def parse_config(f, fashion='new'):
    conf = yaml.safe_load(open(f, 'r'))
    if fashion == 'new':
        sample_param = SampleParams(**conf['sampling'][0])
        memory_param = MemoryParams(**conf['memory'][0])
        gnn_param = GNNParams(**conf['gnn'][0])
        train_param = TrainParams(**conf['train'][0])
    else:
        sample_param = conf['sampling'][0]
        memory_param = conf['memory'][0]
        gnn_param = conf['gnn'][0]
        train_param = conf['train'][0]
    return sample_param, memory_param, gnn_param, train_param

def to_dgl_blocks(ret, history, reverse=False, cuda=True):
    mfgs = list()
    # 6 tgbs -> mfgs length 6 [1,2,3,4,5,6]
    # if history = 2, grouped_mfgs = [[1,2],[3,4],[5,6]]
    for tgb in ret:
        tgb_nodes = torch.from_numpy(tgb.nodes())
        tgb_dts = torch.from_numpy(tgb.dts())
        tgb_ts = torch.from_numpy(tgb.ts())
        tgb_eid = torch.from_numpy(tgb.eid())
        if not reverse:
            edges = (tgb.col(), tgb.row())
            num_src, num_dst = tgb.dim_in(), tgb.dim_out()
            dt_slice_start = num_dst
            id_ts_target = 'srcdata'
        else:
            edges = (tgb.row(), tgb.col())
            num_src, num_dst = tgb.dim_out(), tgb.dim_in()
            dt_slice_start = num_src
            id_ts_target = 'dstdata'
        block = dgl.create_block(edges, num_src_nodes=num_src, num_dst_nodes=num_dst)
        getattr(block, id_ts_target)['ID'] = tgb_nodes
        getattr(block, id_ts_target)['ts'] = tgb_ts
        
        # Assign edge data
        block.edata['dt'] = tgb_dts[dt_slice_start:]
        block.edata['ID'] = tgb_eid
        
        mfgs.append(block.to('cuda:0') if cuda else block)
    grouped_mfgs = list(map(list, zip(*[iter(mfgs)] * history)))
    grouped_mfgs.reverse()
    return grouped_mfgs

def node_to_dgl_blocks(root_nodes, ts, cuda=True):
    mfgs = list()
    block = dgl.create_block(([],[]), num_src_nodes=root_nodes.shape[0], num_dst_nodes=root_nodes.shape[0])
    block.srcdata['ID'] = torch.from_numpy(root_nodes)
    block.srcdata['ts'] = torch.from_numpy(ts)
    if cuda:
        mfgs.insert(0, [block.to('cuda:0')])
    else:
        mfgs.insert(0, [block])
    return mfgs

def mfgs_to_cuda(mfgs):
    for mfg in mfgs:
        for i in range(len(mfg)):
            mfg[i] = mfg[i].to('cuda:0')
    return mfgs

def prepare_input(mfgs, node_feats, edge_feats, combine_first=False, pinned=False, nfeat_buffs=None, efeat_buffs=None, nids=None, eids=None):
    if combine_first:
        for i in range(len(mfgs[0])):
            if mfgs[0][i].num_src_nodes() > mfgs[0][i].num_dst_nodes():
                num_dst = mfgs[0][i].num_dst_nodes()
                ts = mfgs[0][i].srcdata['ts'][num_dst:]
                nid = mfgs[0][i].srcdata['ID'][num_dst:].float()
                nts = torch.stack([ts, nid], dim=1)
                unts, idx = torch.unique(nts, dim=0, return_inverse=True)
                uts = unts[:, 0]
                unid = unts[:, 1]
                # import pdb; pdb.set_trace()
                b = dgl.create_block((idx + num_dst, mfgs[0][i].edges()[1]), num_src_nodes=unts.shape[0] + num_dst, num_dst_nodes=num_dst, device=torch.device('cuda:0'))
                b.srcdata['ts'] = torch.cat([mfgs[0][i].srcdata['ts'][:num_dst], uts], dim=0)
                b.srcdata['ID'] = torch.cat([mfgs[0][i].srcdata['ID'][:num_dst], unid], dim=0)
                b.edata['dt'] = mfgs[0][i].edata['dt']
                b.edata['ID'] = mfgs[0][i].edata['ID']
                mfgs[0][i] = b
                
    t_idx = 0
    t_cuda = 0
    i = 0
    if node_feats is not None:
        for b in mfgs[0]:
            if pinned:
                if nids is not None:
                    idx = nids[i]
                else:
                    idx = b.srcdata['ID'].cpu().long()
                torch.index_select(node_feats, 0, idx, out=nfeat_buffs[i][:idx.shape[0]])
                b.srcdata['h'] = nfeat_buffs[i][:idx.shape[0]].cuda(non_blocking=True)
                i += 1
            else:
                srch = node_feats[b.srcdata['ID'].long()].float()
                b.srcdata['h'] = srch.cuda()
    i = 0
    if edge_feats is not None:
        for mfg in mfgs:
            for b in mfg:
                if b.num_src_nodes() > b.num_dst_nodes():
                    if pinned:
                        if eids is not None:
                            idx = eids[i]
                        else:
                            idx = b.edata['ID'].cpu().long()
                        torch.index_select(edge_feats, 0, idx, out=efeat_buffs[i][:idx.shape[0]])
                        b.edata['f'] = efeat_buffs[i][:idx.shape[0]].cuda(non_blocking=True)
                        i += 1
                    else:
                        srch = edge_feats[b.edata['ID'].long()].float()
                        b.edata['f'] = srch.cuda()
    return mfgs



