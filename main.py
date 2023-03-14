import torch
import argparse

from nerf.provider import NeRFDataset
from nerf.utils_neurallift import *


from optimizer import Shampoo

import pdb
import os
import yaml, json, types

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/cabin.yaml', help='load config')
    args = parser.parse_args()
    with open(args.config, "r") as stream:
        try:
            opt = (yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)

    def load_object(dct):
        return types.SimpleNamespace(**dct)
    opt = json.loads(json.dumps(opt), object_hook=load_object)
    print(opt)
    # from IPython import embed
    # embed()

    from datetime import datetime
    opt.workspace = os.path.basename(args.config).replace('.yaml', '')
    opt.workspace = os.path.join('logs', str(datetime.today().strftime('%Y-%m-%d')), opt.workspace + '_' + datetime.today().strftime('%H:%M:%S'))
    import os, shutil
    os.makedirs(opt.workspace, exist_ok=True)
    shutil.copy(args.config, os.path.join(opt.workspace, os.path.basename(args.config)))

    print('Double Check data path:')
    print(opt.mask_path)
    print(opt.rgb_path)
    print(opt.depth_path)
    print('====================')

    if opt.backbone == 'vanilla':
        from nerf.network import NeRFNetwork
    elif opt.backbone == 'grid_finite':    
        from nerf.network_grid_finite import NeRFNetwork
    else:
        raise NotImplementedError(f'--backbone {opt.backbone} is not implemented!')

    print(opt)

    
    import time
    # seed_everything(np.random.randint(10))
    seed_everything(opt.seed)

    model = NeRFNetwork(opt)

    print(model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if opt.test:
        guidance = None

        trainer = Trainer('lift', opt, model, guidance, device=device, workspace=opt.workspace, fp16=opt.fp16, use_checkpoint=opt.ckpt)
        test_loader = NeRFDataset(opt, device=device, type='test', H=opt.H, W=opt.W, size=100, shading=opt.test_shading).dataloader()
        trainer.test(test_loader)
        
        if opt.save_mesh:
            trainer.save_mesh(resolution=256)
    else:
        if opt.guidance == 'sd_clipguide':
            from nerf.sd_clipguide import StableDiffusion
            guidance = StableDiffusion(opt, device, sd_name=opt.sd_name)
        else:
            raise NotImplementedError(f'--guidance {opt.guidance} is not implemented.')

        optimizer = lambda model: torch.optim.AdamW(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)
    

        train_loader = NeRFDataset(opt, device=device, type='train', H=opt.h, W=opt.w, size=100).dataloader()

        opt.max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)

        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 1) # fixed
        # scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))

        trainer = Trainer('lift', opt, model, guidance, device=device, workspace=opt.workspace, optimizer=optimizer, ema_decay=None, fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint=opt.ckpt, eval_interval=opt.eval_interval, scheduler_update_every_step=True)
        valid_loader = NeRFDataset(opt, device=device, type='val', H=opt.H, W=opt.W, size=5).dataloader()

        opt.max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
    
        if True:
            trainer.train(train_loader, valid_loader, opt.max_epoch)

        test_loader = NeRFDataset(opt, device=device, type='test', H=opt.H, W=opt.W, size=100).dataloader()
        trainer.test(test_loader)

        if opt.save_mesh:
            trainer.save_mesh(resolution=256)
