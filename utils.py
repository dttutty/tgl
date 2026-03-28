import torch
import os
import random
import yaml
import dgl
import time
import pandas as pd
import numpy as np

def _yaml_safe_value(value):
    if isinstance(value, dict):
        return {k: _yaml_safe_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_yaml_safe_value(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)

def _yaml_block(data):
    dumped = yaml.safe_dump(
        _yaml_safe_value(data),
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    )
    return dumped.rstrip() if dumped else ''

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _normalize_lr_scheduler_config(train_param):
    if train_param is None:
        return None
    scheduler_cfg = train_param.get('lr_scheduler')
    if scheduler_cfg is None or scheduler_cfg is False:
        return None
    if isinstance(scheduler_cfg, str):
        scheduler_cfg = {'type': scheduler_cfg}
    elif not isinstance(scheduler_cfg, dict):
        raise TypeError('train.lr_scheduler must be a string or a mapping')

    scheduler_cfg = dict(scheduler_cfg)
    scheduler_type = scheduler_cfg.get('type', 'reduce_on_plateau')
    if scheduler_type is None or scheduler_type is False:
        return None

    scheduler_type = str(scheduler_type).lower()
    if scheduler_type == 'none':
        return None
    if scheduler_type in {'plateau', 'reduce_on_plateau', 'reducelronplateau'}:
        scheduler_cfg['type'] = 'reduce_on_plateau'
        return scheduler_cfg
    raise ValueError('Unsupported train.lr_scheduler type: {}'.format(scheduler_cfg['type']))

def has_train_lr_scheduler(train_param):
    return _normalize_lr_scheduler_config(train_param) is not None

def create_train_lr_scheduler(optimizer, train_param, default_monitor='val_ap'):
    scheduler_cfg = _normalize_lr_scheduler_config(train_param)
    if scheduler_cfg is None:
        return None

    monitor = scheduler_cfg.pop('monitor', None) or default_monitor
    mode = scheduler_cfg.pop('mode', None)
    if mode is None:
        mode = 'min' if 'loss' in monitor.lower() else 'max'

    scheduler_type = scheduler_cfg.pop('type')
    if scheduler_type != 'reduce_on_plateau':
        raise ValueError('Unsupported normalized scheduler type: {}'.format(scheduler_type))

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=mode,
        factor=scheduler_cfg.pop('factor', 0.5),
        patience=scheduler_cfg.pop('patience', 5),
        threshold=scheduler_cfg.pop('threshold', 1e-4),
        threshold_mode=scheduler_cfg.pop('threshold_mode', 'rel'),
        cooldown=scheduler_cfg.pop('cooldown', 0),
        min_lr=scheduler_cfg.pop('min_lr', 0.0),
        eps=scheduler_cfg.pop('eps', 1e-8),
    )
    if scheduler_cfg:
        raise ValueError(
            'Unsupported train.lr_scheduler options: {}'.format(', '.join(sorted(scheduler_cfg.keys())))
        )

    return {
        'type': scheduler_type,
        'monitor': monitor,
        'mode': mode,
        'scheduler': scheduler,
    }

def step_train_lr_scheduler(lr_scheduler, metrics):
    if lr_scheduler is None:
        return None

    monitor = lr_scheduler['monitor']
    if monitor not in metrics:
        raise KeyError(
            "Configured lr scheduler monitor '{}' is unavailable. Available metrics: {}".format(
                monitor, ', '.join(sorted(metrics.keys()))
            )
        )

    metric = metrics[monitor]
    if metric is None:
        raise ValueError("Configured lr scheduler monitor '{}' has value None".format(monitor))

    scheduler = lr_scheduler['scheduler']
    old_lrs = [float(param_group['lr']) for param_group in scheduler.optimizer.param_groups]
    scheduler.step(float(metric))
    new_lrs = [float(param_group['lr']) for param_group in scheduler.optimizer.param_groups]

    return {
        'type': lr_scheduler['type'],
        'monitor': monitor,
        'metric': float(metric),
        'old_lrs': old_lrs,
        'new_lrs': new_lrs,
    }

def format_lr_scheduler_step(step_result):
    if step_result is None:
        return ''

    def _format_lrs(lrs):
        return ', '.join('{:.6g}'.format(lr) for lr in lrs)

    return 'lr scheduler {} monitor {}={:.6f} lr {} -> {}'.format(
        step_result['type'],
        step_result['monitor'],
        step_result['metric'],
        _format_lrs(step_result['old_lrs']),
        _format_lrs(step_result['new_lrs']),
    )

def load_feat(d, rand_de, rand_dn):
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
            node_feats = torch.randn(7144, rand_dn)
    # print("Node features shape:", None if node_feats is None else node_feats.shape)
    # print("Edge features shape:", None if edge_feats is None else edge_feats.shape)
    return node_feats, edge_feats

def load_graph(d):
    df = pd.read_csv('DATA/{}/edges.csv'.format(d))
    g = np.load('DATA/{}/full_graph_with_reverse_edges.npz'.format(d))
    return g, df

def parse_config(f):
    conf = yaml.safe_load(open(f, 'r'))
    sample_param = conf['sampling'][0]
    memory_param = conf['memory'][0]
    gnn_param = conf['gnn'][0]
    train_param = conf['train'][0]
    return sample_param, memory_param, gnn_param, train_param

def print_run_configuration(args, sample_param=None, memory_param=None, gnn_param=None, train_param=None, config_path=None, rank=None):
    if rank is not None and rank != 0:
        return

    header = 'Run configuration'
    if rank is not None:
        header = '{} (rank {})'.format(header, rank)

    print('=' * 80)
    print(header)
    if config_path:
        print('config_path: {}'.format(config_path))
    print('args:')
    print(_yaml_block(vars(args)) or '{}')
    print('config:')
    if all(section is not None for section in [sample_param, memory_param, gnn_param, train_param]):
        print(_yaml_block({
            'sampling': [sample_param],
            'memory': [memory_param],
            'gnn': [gnn_param],
            'train': [train_param],
        }) or '{}')
    elif config_path:
        print('<config not loaded>')
    else:
        print('<config not provided>')
    print('=' * 80)

def to_dgl_blocks(ret, hist, reverse=False, cuda=True):
    mfgs = list()
    for r in ret:
        if not reverse:
            b = dgl.create_block((r.col(), r.row()), num_src_nodes=r.dim_in(), num_dst_nodes=r.dim_out())
            b.srcdata['ID'] = torch.from_numpy(r.nodes())
            b.edata['dt'] = torch.from_numpy(r.dts())[b.num_dst_nodes():]
            b.srcdata['ts'] = torch.from_numpy(r.ts())
        else:
            b = dgl.create_block((r.row(), r.col()), num_src_nodes=r.dim_out(), num_dst_nodes=r.dim_in())
            b.dstdata['ID'] = torch.from_numpy(r.nodes())
            b.edata['dt'] = torch.from_numpy(r.dts())[b.num_src_nodes():]
            b.dstdata['ts'] = torch.from_numpy(r.ts())
        b.edata['ID'] = torch.from_numpy(r.eid())
        if cuda:
            mfgs.append(b.to('cuda:0'))
        else:
            mfgs.append(b)
    mfgs = list(map(list, zip(*[iter(mfgs)] * hist)))
    mfgs.reverse()
    return mfgs

def node_to_dgl_blocks(root_nodes, ts, cuda=True):
    mfgs = list()
    b = dgl.create_block(([],[]), num_src_nodes=root_nodes.shape[0], num_dst_nodes=root_nodes.shape[0])
    b.srcdata['ID'] = torch.from_numpy(root_nodes)
    b.srcdata['ts'] = torch.from_numpy(ts)
    if cuda:
        mfgs.insert(0, [b.to('cuda:0')])
    else:
        mfgs.insert(0, [b])
    return mfgs

def mfgs_to_cuda(mfgs):
    for mfg in mfgs:
        for i in range(len(mfg)):
            mfg[i] = mfg[i].to('cuda:0')
    return mfgs

def gather_feature_rows(feats, idx, target_device=None):
    if not torch.is_tensor(idx):
        idx = torch.as_tensor(idx, device=feats.device, dtype=torch.long)
    else:
        idx = idx.to(feats.device, dtype=torch.long)
    gathered = torch.index_select(feats, 0, idx).float()
    if target_device is not None and gathered.device != target_device:
        gathered = gathered.to(target_device, non_blocking=True)
    return gathered

def prepare_input(mfgs, node_feats, edge_feats, combine_first=False, pinned=False, nfeat_buffs=None, efeat_buffs=None, nids=None, eids=None):
    if combine_first:
        for i in range(len(mfgs[0])):
            if mfgs[0][i].num_src_nodes() > mfgs[0][i].num_dst_nodes():
                num_dst = mfgs[0][i].num_dst_nodes()
                ts = mfgs[0][i].srcdata['ts'][num_dst:]
                nid = mfgs[0][i].srcdata['ID'][num_dst:].to(ts.dtype)
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
                b.srcdata['h'] = gather_feature_rows(node_feats, b.srcdata['ID'], b.device)
    # i = 0
    # if edge_feats is not None:
    #     for mfg in mfgs:
    #         for b in mfg:
    #             if b.num_src_nodes() > b.num_dst_nodes():
    #                 if pinned:
    #                     if eids is not None:
    #                         idx = eids[i]
    #                     else:
    #                         idx = b.edata['ID'].cpu().long()
    #                     torch.index_select(edge_feats, 0, idx, out=efeat_buffs[i][:idx.shape[0]])
    #                     b.edata['f'] = efeat_buffs[i][:idx.shape[0]].cuda(non_blocking=True)
    #                     i += 1
    #                 else:
    #                     srch = edge_feats[b.edata['ID'].long()].float()
    #                     b.edata['f'] = srch.cuda()
    
    i = 0
    if edge_feats is not None:
        for mfg in mfgs:
            for b in mfg:
                if b.num_edges() == 0 or 'ID' not in b.edata:
                    continue
                if pinned:
                    if eids is not None:
                        idx = eids[i]
                    else:
                        idx = b.edata['ID'].cpu().long()
                    torch.index_select(edge_feats, 0, idx, out=efeat_buffs[i][:idx.shape[0]])
                    b.edata['f'] = efeat_buffs[i][:idx.shape[0]].cuda(non_blocking=True)
                    i += 1
                else:
                    b.edata['f'] = gather_feature_rows(edge_feats, b.edata['ID'], b.device)
            
    return mfgs

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
