import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("..")

import hyperparams as hyp
import archs.v2v2d 
import utils_geom
import utils_vox
import utils_misc
import utils_basic

EPS = 1e-4
class LinClassNet(nn.Module):
    def __init__(self, in_dim=64):
        super(LinClassNet, self).__init__()

        # let's classify obj, non-obj, just on the surface, very much like we're doing in the test

        self.net = nn.Sequential(
            nn.Linear(in_dim, 2, bias=True),
        ).cuda()
        print(self.net)

        # self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.crit = nn.BCEWithLogitsLoss()

    def forward(self, codes, obj_inds, bkg_inds, summ_writer=None):
        total_loss = torch.tensor(0.0).cuda()

        obj_codes = codes[obj_inds]
        bkg_codes = codes[bkg_inds]

        obj_logits = self.net(obj_codes)
        obj_targets = torch.ones_like(obj_logits)
        obj_loss = self.crit(obj_logits, obj_targets)
        
        bkg_logits = self.net(bkg_codes)
        bkg_targets = torch.zeros_like(bkg_logits)
        bkg_loss = self.crit(bkg_logits, bkg_targets)

        bal_loss = obj_loss + bkg_loss
        total_loss = utils_misc.add_loss('linclass/bce_loss', total_loss,
                                         bal_loss, hyp.linclass_coeff, summ_writer)
        
        obj_sig = F.sigmoid(obj_logits)
        obj_bin = torch.round(obj_sig)
        
        bkg_sig = F.sigmoid(bkg_logits)
        bkg_bin = torch.round(bkg_sig)

        # collect some accuracy stats 
        obj_match = torch.eq(obj_bin, 1).float()
        bkg_match = torch.eq(bkg_bin, 0).float()
        
        obj_acc = torch.mean(obj_match)
        bkg_acc = torch.mean(bkg_match)
        bal_acc = (obj_acc + bkg_acc)*0.5
        summ_writer.summ_scalar('unscaled_linclass/acc_obj', obj_acc.cpu().item())
        summ_writer.summ_scalar('unscaled_linclass/acc_bkg', bkg_acc.cpu().item())
        summ_writer.summ_scalar('unscaled_linclass/acc_bal', bal_acc.cpu().item())
        
        return total_loss

