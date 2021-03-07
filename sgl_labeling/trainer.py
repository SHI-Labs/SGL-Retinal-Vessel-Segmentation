import os
import math
from decimal import Decimal

import data
import utility
import numpy as np
import torch
import torch.nn.utils as utils
import torch.nn as nn
import torch.nn.functional as F
from  loss.bceloss import dice_bce_loss as DICE
from  loss.bceloss import penalty_bce_loss as PBCE
from loss.tv import TVLoss
from tqdm import tqdm
import time
torch.autograd.set_detect_anomaly(True)
class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.ploss = nn.L1Loss().cuda()
        self.bce_loss = nn.BCELoss().cuda()
        self.dice_bce_loss = DICE().cuda()
        self.pbce_loss = PBCE().cuda()
        self.tv_loss = TVLoss().cuda()
        self.optimizer = utility.make_optimizer(args, self.model)

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (data_pack, _) in enumerate(self.loader_train):
            data_pack = self.prepare(data_pack)
            timer_data.hold()
            timer_model.tic()
            hr, ve, ma, pm, _, _ = data_pack
            self.optimizer.zero_grad()
            enh, estimation= self.model(hr, 1)
            #loss = self.bce_loss(estimation, ve)  #l1 supervisions
            #loss = self.ploss(medmap, med) + 
            loss = self.loss(estimation, ve, pm*ma)# + 0.01*self.tv_loss(enh)# + self.pbce_loss(estimation, ve, pm*ma)  #l1 supervisions
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()
        Background_IOU = []
        Vessel_IOU = []
        ACC = []
        SE = []
        SP = []
        AUC = []
        BIOU = []
       
        thin_Background_IOU = []
        thin_Vessel_IOU = []
        thin_ACC = []
        thin_SE = []
        thin_SP = []
        thin_DICE = []
       
        thick_Background_IOU = []
        thick_Vessel_IOU = []
        thick_ACC = []
        thick_SE = []
        thick_SP = []
        thick_DICE = []
        
        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                for data_pack, filename in tqdm(d, ncols=80):
                    data_pack = self.prepare(data_pack)
                    hr, ve, ma, pm, ve_thin, ve_thick = data_pack
                    enhance, est_o = self.model(hr, idx_scale) 
                    #est_o = self.model(hr, idx_scale)
                    #enhance = est_o
                    est_o = est_o * ma
                    ve = ve * ma
                    dmap = utility.visualize_dmap(est_o, ve) * 255
                    #enhance = 255 * ((enhance - torch.min(enhance)) / (torch.max(enhance) - torch.min(enhance)))
                    enhance = enhance * ma * 255
                    #enhance = 255 * enhance
                    est = est_o * 255.
                    pm = pm[:, :, 0:584, 0:565]
                    pm[pm>=0.99] = 1
                    pm[pm<0.99] = 0  #define the regions
                       
                    #pm = pm * 255.

                    #med = med #* 255.
                    hr = hr #* 255.
                    ve = ve * ma * 255.
                    ve_thin = ve_thin * ma * 255.
                    ve_thick = ve_thick * ma * 255.
                    est[est>100] = 255
                    est[est<=100] = 0
                    est = est[:, :, 0:584, 0:565]
                    #med = med[:, :, 0:584, 0:565]
                    #medmap = medmap[:,:,0:584, 0:565]
                    dmap = dmap[:, :, 0:584, 0:565]
                    est_o = est_o[:, :, 0:584, 0:565]
                    enhance = enhance[:, :, 0:584, 0:565]
                    ve = ve[:, :, 0:584, 0:565]
                    ve_thin = ve_thin[:, :, 0:584, 0:565]
                    ve_thick = ve_thick[:, :, 0:584, 0:565]
                    hr = hr[:, :, 0:584, 0:565]
                    est = utility.quantize(est, self.args.rgb_range)
                    estnp = np.transpose(est[0].cpu().numpy(), (1,2,0)) / 255.
                    est_thin, est_thick = data.common.ttsep(estnp)
                    vis_vessel = utility.visualize(est, ve, False)
                    vis_thin = utility.visualize(est_thin*255., ve_thin, True)                    
                    vis_thick = utility.visualize(est_thick*255., ve_thick, True)                    
                    #med = utility.quantize(med, self.args.rgb_range)
                    save_list = [est_o*255]
                   
                    self.ckp.log[-1, idx_data, idx_scale] += utility.calc_dice(est, ve, False)
                    #self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(est, med, 1, 255)
                    ve = utility.quantize(ve, self.args.rgb_range)
                    Acc,Se,Sp,Auc,IU0, IU1 = utility.calc_metrics(est_o, est, ve, False)
                    BIOU.append(utility.calc_boundiou(est_o, ve/255., pm))
                    AUC.append(Auc)
                    Background_IOU.append(IU0)
                    Vessel_IOU.append(IU1)
                    ACC.append(Acc)
                    SE.append(Se)
                    SP.append(Sp)
                    #Thin vessel
                    Acc,Se,Sp,_,IU0, IU1 = utility.calc_metrics(est_o, est_thin, ve_thin, True)
                    thin_Background_IOU.append(IU0)
                    thin_Vessel_IOU.append(IU1)
                    thin_ACC.append(Acc)
                    thin_SE.append(Se)
                    thin_SP.append(Sp)
                    DICE = utility.calc_dice(est_thin, ve_thin, True)
                    thin_DICE.append(DICE)
                    #Thick vessel
                    Acc,Se,Sp,_,IU0, IU1 = utility.calc_metrics(est_o, est_thick, ve_thick, True)
                    thick_Background_IOU.append(IU0)
                    thick_Vessel_IOU.append(IU1)
                    thick_ACC.append(Acc)
                    thick_SE.append(Se)
                    thick_SP.append(Sp)
                    DICE = utility.calc_dice(est_thick, ve_thick, True)
                    thick_DICE.append(DICE)

                    if self.args.save_gt:
                        save_list.extend([hr, pm*255])

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)
                print(np.mean(np.stack(BIOU)))
                print('Acc: %s  |  Se: %s |  Sp: %s |  Auc: %s |  Background_IOU: %s |  vessel_IOU: %s '%(str(np.mean(np.stack(ACC))),str(np.mean(np.stack(SE))), str(np.mean(np.stack(SP))),str(np.mean(np.stack(AUC))),str(np.mean(np.stack(Background_IOU))),str(np.mean(np.stack(Vessel_IOU)))))
                #Thin Vessel
                print('Thin Vessel: Acc: %s  |  Se: %s |  Sp: %s | DICE: %s | Background_IOU: %s |  vessel_IOU: %s '%(str(np.mean(np.stack(thin_ACC))),str(np.mean(np.stack(thin_SE))), str(np.mean(np.stack(thin_SP))), str(np.mean(np.stack(thin_DICE))), str(np.mean(np.stack(thin_Background_IOU))),str(np.mean(np.stack(thin_Vessel_IOU)))))
                #Thick Vessel
                print('Thick Vessel: Acc: %s  |  Se: %s |  Sp: %s |  DICE: %s |  Background_IOU: %s |  vessel_IOU: %s '%(str(np.mean(np.stack(thick_ACC))),str(np.mean(np.stack(thick_SE))), str(np.mean(np.stack(thick_SP))),str(np.mean(np.stack(thick_DICE))),str(np.mean(np.stack(thick_Background_IOU))),str(np.mean(np.stack(thick_Vessel_IOU)))))
                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR Score: {:.6f} (Best: {:.6f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1
                    )
                )
                #print(np.mean(ssim))

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            #self.evaluation_model()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs
