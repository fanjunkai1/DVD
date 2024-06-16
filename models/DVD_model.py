from utils import get_root_logger
from utils.registry import MODEL_REGISTRY
from torch import distributed as dist
import torch
import importlib
from archs import build_network
from losses import build_loss
from tqdm import tqdm
from collections import Counter
from os import path as osp
from copy import deepcopy
from metrics import calculate_metric
from utils import get_root_logger, imwrite, tensor2img, img2tensor
from utils.dist_util import get_dist_info
from collections import OrderedDict
from .base_model import BaseModel
from archs.DVD_arch import SpyNet
from archs.arch_util import flow_warp
import torch.nn.functional as F
from utils import check_flow_occlusion, flow_to_image


@MODEL_REGISTRY.register()
class NSDNGANModel(BaseModel):
    def __init__(self, opt):
        super(NSDNGANModel, self).__init__(opt)
        # if self.is_train:
        #     self.train_tsa_iter = opt['train'].get('tsa_iter')
            # self.output_pool= ImagePool(opt['network_g']['num_frame']-1)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        # self.print_network(self.net_g)

        # load pretrained models
        load_g_path = self.opt['path'].get('pretrain_network_g', None)
        if load_g_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_g_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.net_d_iters = opt['train'].get('net_d_iters', 1)
            self.net_d_init_iters = opt['train'].get('net_d_init_iters', 0)

        if self.is_train:
            self.init_training_settings()
        
        spynet_path='pretrained/spynet_sintel_final-3d2a1287.pth'
        with torch.no_grad():
            self.spynet = SpyNet(spynet_path)   
        self.spynet = self.model_to_device(self.spynet)
        

    def init_training_settings(self):

        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0.999)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define network net_d
        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        # self.print_network(self.net_d)
        
        load_d_path = self.opt['path'].get('pretrain_network_d', None)
        if load_d_path is not None:
            param_key = self.opt['path'].get('param_key_d', 'params')
            self.load_network(self.net_d, load_d_path, self.opt['path'].get('strict_load_d', True), param_key)
        
        self.net_g.train()
        self.net_d.train()

        # define losses
        if train_opt.get('align_opt'):
            self.cri_align = build_loss(train_opt['align_opt']).to(self.device)
        else:
            self.cri_align = None

        if train_opt.get('warp_opt'):
            self.cri_warp = build_loss(train_opt['warp_opt']).to(self.device)
        else:
            self.cri_warp = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual= build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('contextual_opt'):
            self.cri_contextual = build_loss(train_opt['contextual_opt']).to(self.device)
        else:
            self.cri_contextual = None

        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)

        if self.cri_gan is None and self.cri_contextual is None:
            raise ValueError('GAN and contextual losses are None.')

        # set up optimizers and schedulers
        self.setup_schedulers()
        self.setup_optimizers()
        
    
    def feed_data(self, data):
        self.hfs = data['hfs'].to(self.device)
        self.predehazing_hfs = data['predehazing_hfs'].to(self.device)
        if 'cf_ref_curr' in data:
            self.cf_ref_curr = data['cf_ref_curr'].to(self.device)
        if 'cf_ref_next' in data:
            self.cf_ref_next = data['cf_ref_next'].to(self.device)
        # if 'optical_flow' in data:
        #     self.optical_flow = data['optical_flow'].to(self.device)

        self.curr_frame_path = data['curr_frame_path']

    def setup_optimizers(self):
        train_opt = self.opt['train']
        dcn_lr_mul = train_opt.get('dcn_lr_mul', 1)
        logger = get_root_logger()
        logger.info(f'Multiple the learning rate for dcn with {dcn_lr_mul}.')
        if dcn_lr_mul == 1:
            optim_params = self.net_g.parameters()
        else:  # separate dcn params and normal params for different lr
            normal_params = []
            dcn_params = []
            for name, param in self.net_g.named_parameters():
                if 'dcn' in name:
                    dcn_params.append(param)
                else:
                    normal_params.append(param)
            optim_params = [
                {  # add normal params first
                    'params': normal_params,
                    'lr': train_opt['optim_g']['lr']
                },
                {
                    'params': dcn_params,
                    'lr': train_opt['optim_g']['lr'] * dcn_lr_mul
                },
            ]
        # optimizer g
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)
        # optimizer d
        optim_type = train_opt['optim_d'].pop('type')
        self.optimizer_d = self.get_optimizer(optim_type, self.net_d.parameters(), **train_opt['optim_d'])
        self.optimizers.append(self.optimizer_d)


    def optimize_parameters(self, current_iter, previous_result):
        # if self.train_tsa_iter:
        #     if current_iter == 1:
        #         logger = get_root_logger()
        #         logger.info(f'Only train TSA module for {self.train_tsa_iter} iters.')
        #         for name, param in self.net_g.named_parameters():
        #             if 'fusion' not in name:
        #                 param.requires_grad = False
        #     elif current_iter == self.train_tsa_iter:
        # logger = get_root_logger()
        # logger.warning('Train all the parameters.')

        # self.output_previous = self.output_pool.query(self.output)

        for param in self.net_g.parameters():
            param.requires_grad = True

        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False


        self.optimizer_g.zero_grad()
        self.output, self.aligned_frames, self.flow_vis, self.nbr_fea, self.nbr_ref_fea = self.net_g(self.predehazing_hfs)

        l_g_total = 0
        loss_dict = OrderedDict()

        if self.cri_contextual:
            l_contextual_1 = self.cri_contextual(self.output, self.cf_ref_curr)[0]
            l_contextual_2 = self.cri_contextual(self.output, self.cf_ref_next)[0]
            l_contextual = (l_contextual_1 + l_contextual_2)
            l_g_total += l_contextual

            loss_dict['l_contextual'] = l_contextual


        if self.cri_align:
            l_align_sum = 0
            b, t, c, h, w = self.aligned_frames.shape
            for i in range(0, t):
                l_align = self.cri_align(self.predehazing_hfs[:,-1, :, : , :], self.aligned_frames[:, i, :, :, :])
                # l_perceptual = self.cri_perceptual(self.predehazing_hfs[:,-1, :, : , :], self.aligned_frames[:, i, :, :, :])[0]
                l_align_sum += l_align
            l_g_total += l_align_sum

            loss_dict['l_align'] = l_align_sum
        
        if self.cri_warp:
            flow_b = self.spynet(self.output, previous_result).detach()
            flow_f = self.spynet(previous_result, self.output).detach()
            mask_f, mask_b = check_flow_occlusion(flow_b[0,:,:,:], flow_f[0,:,:,:])
            warped_to_curr = flow_warp(self.output, flow_b.permute(0, 2, 3, 1), 'bilinear')
            l_warp = self.cri_warp(self.output*mask_b, warped_to_curr*mask_b)
            l_g_total += l_warp

            loss_dict['l_warp'] = l_warp

        # gan loss
        fake_g_pred = self.net_d(self.output)
        l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
        l_g_total += l_g_gan
        loss_dict['l_g_gan'] = l_g_gan

        l_g_total.backward()
        self.optimizer_g.step()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True
        
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            
            # for name, param in self.net_d.named_parameters():
            #     if param.grad is None:
            #         print(name)
            
            self.optimizer_d.zero_grad()
            # print("############")
            # real
            real_d_pred = (self.net_d(self.cf_ref_curr) + self.net_d(self.cf_ref_next)) / 2
            # print("hahhhhhhhhhh")
            l_d_real = self.cri_gan(real_d_pred, True, is_disc=True) 
            loss_dict['l_d_real'] = l_d_real
            loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
            
            l_d_real.backward()
            # fake
            fake_d_pred = self.net_d(self.output.detach().clone())  # clone for pt1.9
            l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True) 
            loss_dict['l_d_fake'] = l_d_fake
            loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
            l_d_fake.backward()
            self.optimizer_d.step()

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
            
        self.log_dict = self.reduce_loss_dict(loss_dict)

        return self.output


    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output, self.aligned_frames, self.flow_vis, self.nbr_fea, self.nbr_ref_fea = self.net_g_ema(self.predehazing_hfs)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output, self.aligned_frames, self.flow_vis, self.nbr_fea, self.nbr_ref_fea = self.net_g(self.predehazing_hfs)
            self.net_g.train()
    
    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        logger = get_root_logger()
        logger.info('Only support single GPU validation.')
        self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        pbar = tqdm(total=len(dataloader), unit='image')

        metric_data = dict()


        for idx, val_data in enumerate(dataloader):
            self.save_predehazing_img_list = []
            self.save_hfs_img_list = []
            self.save_aligned_frames_img_list = []
            self.save_flow_img_list = []
            self.save_aligned_fea_list = []
            video_index = int(val_data['curr_frame_path'][0].split('/')[-3].split('_')[0])

            img_name = osp.splitext(osp.basename(val_data['curr_frame_path'][0]))[0]
            self.feed_data(val_data)
            self.test()
            visuals = self.get_current_visuals(current_iter)

            save_file_name = str(video_index) + '_' + "video_results"

            for hfs_img_name in self.save_hfs_img_list:
                hfs_img = tensor2img(visuals[hfs_img_name])
                save_hfs_img_path = osp.join(self.opt['path']['visualization'], 
                                            save_file_name, img_name, 'input_hazy_frame',
                                            f'{hfs_img_name}.png')
                
                imwrite(hfs_img, save_hfs_img_path)

            for predehazing_img_name in self.save_predehazing_img_list:
                predehazing_img = tensor2img(visuals[predehazing_img_name])
                save_predehazing_img_path = osp.join(self.opt['path']['visualization'], 
                                                    save_file_name, img_name, 'predehazing_results',
                                                    f'{predehazing_img_name}.png')
                
                imwrite(predehazing_img, save_predehazing_img_path)
            
            for aligned_frame_img_name in self.save_aligned_frames_img_list:
                aligned_frame_img = tensor2img(visuals[aligned_frame_img_name]) 
                save_aligned_frames_img_path = osp.join(self.opt['path']['visualization'], 
                                                    save_file_name, img_name, 'aligned_frames_vis',
                                                    f'{aligned_frame_img_name}.png')
                imwrite(aligned_frame_img, save_aligned_frames_img_path)
            
            for flow_img_name in self.save_flow_img_list:
                # print(visuals[flow_img_name].shape)
                flow_img = flow_to_image(visuals[flow_img_name]) 
                save_flow_img_path = osp.join(self.opt['path']['visualization'], 
                                                    save_file_name, img_name, 'flow_vis',
                                                    f'{flow_img_name}.png')
                imwrite(flow_img, save_flow_img_path)
                
            curr_frame_index = int(self.curr_frame_path[0].split('/')[-1].split('_')[1])
            cf_ref_curr_img = tensor2img(visuals['cf_ref_curr'])
            imwrite(cf_ref_curr_img,osp.join(self.opt['path']['visualization'], 
                                                    save_file_name, img_name, 'reference_frame',
                                                    f'frame_{curr_frame_index}_clear.png'))
            cf_ref_next_img = tensor2img(visuals['cf_ref_next'])
            imwrite(cf_ref_next_img,osp.join(self.opt['path']['visualization'], 
                                        save_file_name, img_name, 'reference_frame',
                                        f'frame_{curr_frame_index+1}_clear.png'))

            
            sr_img = tensor2img(visuals['result'])

            nbr_fea_img = tensor2img(visuals['nbr_fea'])
            imwrite(nbr_fea_img, osp.join(self.opt['path']['visualization'], 
                            save_file_name, img_name, 'feature_map_vis',
                            f'nbr_before_fea_vis_16_{current_iter}.png'))
            
            ref_fea_img = tensor2img(visuals['ref_fea'])
            imwrite(ref_fea_img, osp.join(self.opt['path']['visualization'], 
                            save_file_name, img_name, 'feature_map_vis',
                            f'ref_fea_vis_16_{current_iter}.png'))
            

            for aligned_fea_name in self.save_aligned_fea_list:
                aligned_fea_img = tensor2img(visuals[aligned_fea_name]) 
                save_aligned_fea_path = osp.join(self.opt['path']['visualization'], 
                                                    save_file_name, img_name, 'feature_map_vis',
                                                    f'{aligned_fea_name}.png')
                imwrite(aligned_fea_img, save_aligned_fea_path)
            # metric_data = [img2tensor(sr_img).unsqueeze(0) / 255, self.cf_ref_curr]
            # metric_data = [img2tensor(sr_img).unsqueeze(0) / 255]
            metric_data['img'] = sr_img
            
            # if 'cf_ref_curr' in visuals:
            #     cf_ref_curr_img = tensor2img([visuals['cf_ref_curr']])
            #     metric_data['img2'] = cf_ref_curr_img
            #     del self.cf_ref_curr

            # tentative for out of GPU memory
            del self.predehazing_hfs
            del self.output
            torch.cuda.empty_cache()


            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             save_file_name, img_name, 'augmentation_results',
                                             f'{img_name}_{current_iter}.png')
                    imwrite(sr_img, save_img_path)
                    
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], save_file_name,
                            f'{img_name}_{self.opt["val"]["suffix"]}.png')
                        imwrite(sr_img, save_img_path)
                    else:
                        save_img_path_1 = osp.join(
                            self.opt['path']['visualization'], save_file_name,
                            img_name, 'augmentation_results', f'{img_name}_results.png')
                        
                        save_img_path_2 = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{video_index}_{img_name}_results.png')
                        imwrite(sr_img, save_img_path_1)
                        imwrite(sr_img, save_img_path_2)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)


            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self, current_iter):
        out_dict = OrderedDict()
        
        out_dict['result'] = self.output.detach().cpu()

        if hasattr(self, 'cf_ref_curr'):
            out_dict['cf_ref_curr'] = self.cf_ref_curr.detach().cpu()
        if hasattr(self, 'cf_ref_next'):
            out_dict['cf_ref_next'] = self.cf_ref_next.detach().cpu()

        if hasattr(self, 'predehazing_hfs'):
           
           b, t, c, h, w = self.predehazing_hfs.shape
           for i in range(0, t):
               filename = 'frame_{}_predehazing'.format(int(self.curr_frame_path[0].split('/')[-1].split('_')[1]) - i)
            #    print(filename)
               out_dict[filename] = self.predehazing_hfs[:, t-i-1, :, :, :].detach().cpu()
               self.save_predehazing_img_list.append(filename)

        if hasattr(self, 'hfs'):
           
           b, t, c, h, w = self.hfs.shape
           for i in range(0, t):
               filename = 'frame_{}_hazy'.format(int(self.curr_frame_path[0].split('/')[-1].split('_')[1]) - i)
            #    print(filename)
               out_dict[filename] = self.hfs[:, t-i-1, :, :, :].detach().cpu()
               self.save_hfs_img_list.append(filename)
        
        if hasattr(self, 'aligned_frames'):
            b, t, c, h, w = self.aligned_frames.shape
            for i in range(0, t):
                filename = '{}_frame_aligned_to_{}_frame_{}'.format(int(self.curr_frame_path[0].split('/')[-1].split('_')[1]) - i - 1,
                                                                 int(self.curr_frame_path[0].split('/')[-1].split('_')[1]),current_iter)
                out_dict[filename] = self.aligned_frames[:, t-i-1, :, :, :].detach().cpu()
                self.save_aligned_frames_img_list.append(filename)
        
        if hasattr(self, 'flow_vis'):
            t, h, w, c = self.flow_vis.shape
            for i in range(0, t):
                filename = '{}_to_{}_frame_flow_img_{}'.format(int(self.curr_frame_path[0].split('/')[-1].split('_')[1]) - i - 1,
                                                                 int(self.curr_frame_path[0].split('/')[-1].split('_')[1]),current_iter)
                out_dict[filename] = self.flow_vis[t-i-1, :, :, :].detach().cpu().numpy()
                self.save_flow_img_list.append(filename)

        if hasattr(self, 'nbr_fea'):
            channel_index = 16
            # self.nbr_fea.shape = [1, 64 ,256, 256]
            out_dict['nbr_fea'] = self.nbr_fea[:, channel_index, :, :].detach().cpu() # [0:64] select a channel for visualize

        if hasattr(self, 'nbr_ref_fea'):
            channel_index = 16
            # self.nbr_ref_fea.shape = [1, 2, 64 ,256, 256]
            b, t, c, h, w = self.nbr_ref_fea.shape 
            out_dict['ref_fea'] = self.nbr_ref_fea[:, -1, channel_index, :, :].detach().cpu()
            for i in range(0, t-1):
                filename = '{}_fea_aligned_to_{}_fea_{}_{}'.format(int(self.curr_frame_path[0].split('/')[-1].split('_')[1]) - i - 1,
                                                                    int(self.curr_frame_path[0].split('/')[-1].split('_')[1]), channel_index,current_iter)
                out_dict[filename] = self.nbr_ref_fea[:, t-i-1, channel_index, :, :].detach().cpu()
                self.save_aligned_fea_list.append(filename)
        
        return out_dict

    def save(self, epoch, current_iter):

        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)

