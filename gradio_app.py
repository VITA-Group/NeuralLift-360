import torch
import argparse

from nerf.provider import NeRFDataset
from nerf.utils_neurallift import *
import gradio as gr
import gc


from optimizer import Shampoo

import pdb
import os
import yaml, json, types

css="""
.gradio-container {
    max-width: 512px; margin: auto;
} 
"""

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/cabin.yaml', help='load config')
    parser.add_argument('--share', action='store_true', help="do you want to share gradio app to external network?")
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

    trainer = None
    model = None

    # define UI

    with gr.Blocks(css=css) as demo:

        # title
        gr.Markdown('[NeuralLift-360](https://github.com/VITA-Group/NeuralLift-360) Image-to-3D Example')

        # inputs
        with gr.Row().style(equal_height=True):
            ref_im = gr.Image(label="reference_image", elem_id="ref_im", value=opt.rgb_path)
            mask = gr.Image(label="reference_mask", elem_id="ref_mask", value=opt.mask_path)
            with gr.Column(scale=1, min_width=600):
                prompt = gr.Textbox(label="Prompt", max_lines=1, value=opt.text)
                iters = gr.Slider(label="Iters", minimum=1000, maximum=20000, value=opt.iters, step=100)
                seed = gr.Slider(label="Seed", minimum=0, maximum=2147483647, step=1, randomize=True)
        button = gr.Button('Generate')

        # outputs
        image = gr.Image(label="image", visible=True)
        video = gr.Video(label="video", visible=False)
        logs = gr.Textbox(label="logging")

        def submit(text, iters, seed):
            global trainer, model
            opt.seed = seed
            opt.text = text
            opt.iters = iters
    
            seed_everything(opt.seed)

            # clean up
            if trainer is not None:
                del model
                del trainer
                gc.collect()
                torch.cuda.empty_cache()
                print('[INFO] clean up!')


            model = NeRFNetwork(opt)

            print(model)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if opt.guidance == 'sd_clipguide':
                from nerf.sd_clipguide import StableDiffusion
                guidance = StableDiffusion(opt, device, sd_name=opt.sd_name)
            else:
                raise NotImplementedError(f'--guidance {opt.guidance} is not implemented.')

            optimizer = lambda model: torch.optim.AdamW(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)


            train_loader = NeRFDataset(opt, device=device, type='train', H=opt.h, W=opt.w, size=100).dataloader()
            test_loader = NeRFDataset(opt, device=device, type='test', H=opt.H, W=opt.W, size=100).dataloader()

            opt.max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)

            scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 1) # fixed
            


            trainer = Trainer('lift', opt, model, guidance, device=device, workspace=opt.workspace, optimizer=optimizer, ema_decay=None, fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint=opt.ckpt, eval_interval=opt.eval_interval, scheduler_update_every_step=True)

            trainer.writer = tensorboardX.SummaryWriter(os.path.join(opt.workspace, "run", 'lift'))

            valid_loader = NeRFDataset(opt, device=device, type='val', H=opt.H, W=opt.W, size=5).dataloader()

            opt.max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)


            # we have to get the explicit training loop out here to yield progressive results...
            loader = iter(valid_loader)

            start_t = time.time()

            for epoch in tqdm.tqdm(range(opt.max_epoch)):
                STEPS = 100
                
                trainer.train_gui(train_loader,
                epoch=epoch, step=STEPS)
                
                # manual test and get intermediate results
                try:
                    data = next(loader)
                except StopIteration:
                    loader = iter(valid_loader)
                    data = next(loader)

                trainer.model.eval()

                if trainer.ema is not None:
                    trainer.ema.store()
                    trainer.ema.copy_to()

                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=trainer.fp16):
                        preds, preds_depth, pred_mask = trainer.test_step(data, perturb=False)

                if trainer.ema is not None:
                    trainer.ema.restore()

                pred = preds[0].detach().cpu().numpy()
                # pred_depth = preds_depth[0].detach().cpu().numpy()

                pred = (pred * 255).astype(np.uint8)

                yield {
                    image: gr.update(value=pred, visible=True),
                    video: gr.update(visible=False),
                    logs: f"training iters: {epoch * STEPS} / {iters}, lr: {trainer.optimizer.param_groups[0]['lr']:.6f}",
                }
            

            # test
            trainer.test(test_loader)

            results = glob.glob(os.path.join(opt.workspace, 'results', '*rgb*.mp4'))
            assert results is not None, "cannot retrieve results!"
            results.sort(key=lambda x: os.path.getmtime(x)) # sort by mtime
            
            end_t = time.time()
            
            yield {
                image: gr.update(visible=False),
                video: gr.update(value=results[-1], visible=True),
                logs: f"Generation Finished in {(end_t - start_t)/ 60:.4f} minutes!",
            }


        button.click(
            submit, 
            [prompt, iters, seed],
            [image, video, logs]
        )

    # concurrency_count: only allow ONE running progress, else GPU will OOM.
    demo.queue(concurrency_count=1)

    demo.launch(share=args.share, debug=True)