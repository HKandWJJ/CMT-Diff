import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import time
from model.ddpm_trans_modules import retinex as rt
from model_UWnet.model import UWnet
import torchvision
from thop import profile

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/underwater.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['val'], help='val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_infer', action='store_true')
    
    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # 转换为NoneDict，对于丢失的键返回None。
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        wandb_logger = WandbLogger(opt)
    else:
        wandb_logger = None

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('===> Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    UWnet = UWnet().to(device)
    logger.info('===> Initial Model Finished')

    state_dict = torch.load(opt['path']['UWnet_resume_state'])
    UWnet.load_state_dict(state_dict)

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['val'], schedule_phase='val')
    
    logger.info('===> Begin Model Inference.')
    #params
    params = sum(p.numel() for p in diffusion.netG.parameters())
    print(f"total_params:{params}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input = torch.randn(1, 6, 256, 256).to(device)
    timestep = torch.tensor([1]).to(device)
    flops, param = profile(diffusion.netG.denoise_fn, inputs=(input, timestep))
    print(f"flops:{flops}, param:{param}")


    current_step = 0
    current_epoch = 0
    idx = 0

    result_path = '{}'.format(opt['path']['results'])
    UWnet_result_path = '{}'.format(opt['path']['UWnet_results'])
    os.makedirs(result_path, exist_ok=True)
    for _,  val_data in enumerate(val_loader):
        idx += 1
        diffusion.feed_data(val_data)
        start = time.time()
        diffusion.test(continous=True)
        UWnet.eval()
        with torch.no_grad():
            img = val_data['SR'].to(device)
            generate_img = UWnet(img)

        end = time.time()
        print('Execution time:', (end - start), 'seconds')
        visuals = diffusion.get_current_visuals(need_LR=False)

        hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
        fake_img = Metrics.tensor2img(visuals['INF'])  # uint8

        sr_img_mode = 'grid'
        if sr_img_mode == 'single':
            # single img series
            sr_img = visuals['SR']  # uint8
            sample_num = sr_img.shape[0]
            for iter in range(0, sample_num):
                Metrics.save_img(
                    Metrics.tensor2img(sr_img[iter]), '{}/{}_{}_sr_{}.png'.format(result_path, current_step, idx, iter))
        else:
            # grid img
            sr_img = Metrics.tensor2img(visuals['SR'])  
            Metrics.save_img(
                Metrics.tensor2img(visuals['SR'][-1]), '{}/{}.jpg'.format(result_path, idx))
                # Metrics.tensor2img(visuals['HR']), '{}/{}.jpg'.format(result_path, idx))
            Metrics.save_img(
                Metrics.tensor2img(generate_img), '{}/{}.jpg'.format(UWnet_result_path, idx))






        if wandb_logger and opt['log_infer']:
            wandb_logger.log_eval_data(fake_img, Metrics.tensor2img(visuals['SR'][-1]), hr_img)

    if wandb_logger and opt['log_infer']:
        wandb_logger.log_eval_table(commit=True)
