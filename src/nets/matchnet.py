import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

import sys
sys.path.append("..")

import hyperparams as hyp
import utils.basic
import utils.geom
import utils.vox
import utils.misc
import utils.track
import utils.samp

EPS = 1e-4
class MatchNet(nn.Module):
    def __init__(self, radlist):
        super(MatchNet, self).__init__()
        self.smoothl1 = torch.nn.SmoothL1Loss(reduction='mean')
        self.criterion = nn.SoftMarginLoss(reduction='none').cuda()
        self.near_radius = 3.0
        self.neg_proportion = 0.75 # how much to weight neg vs pos in the heatmap

        self.prep_times = []
        self.corr_times = []
        self.argmax_times = []

        radcat = torch.from_numpy(np.array(radlist).astype(np.float32)).cuda()
        self.radcat = radcat.reshape(-1)

        self.grid = None

    def forward(self, templates, search_region, xyz_g=None, rad_g=None, use_window=False, summ_writer=None):
        # templates is a list of rotated templates stacked on dim1, each is shaped B x C x ZZ x ZY x ZX
        # search_region is the featuremap where we are searching; it is B x C x Z x Y x X
        # xyz_g is the location of the answer in the search region; it is B x 3

        f_start = torch.cuda.Event(enable_timing=True)
        f_end = torch.cuda.Event(enable_timing=True)

        e_start = torch.cuda.Event(enable_timing=True)
        e_end = torch.cuda.Event(enable_timing=True)

        

        

        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # # start_time = time.time()
        # start.record()
        

        # e_start_time = time.time()

        
        B, N, C, ZZ, ZY, ZX = list(templates.shape)
        _, C2, Z, Y, X = list(search_region.shape)
        if xyz_g is not None:
            _, D = list(xyz_g.shape)
            assert(D==3)
        assert(C==C2)

        # N = len(templatelist)
        # print('N', N)
        # print('search_region', search_region.shape)
        
        # templates = torch.stack(templatelist, dim=1)
        
        # this is B x N x C x ZZ x ZY x ZX
        search_regions = search_region.unsqueeze(1).repeat(1, N, 1, 1, 1, 1)
        
        # # this is B x N x C x Z x Y x X
        # templates_ = templates.reshape(B*N, C, ZZ, ZY, ZX)
        # search_regions_ = search_regions.reshape(B*N, C, Z, Y, X)

        # print('search_regions_', search_regions_.shape)
        # print('templates_', templates_.shape)

        # end.record()
        # torch.cuda.synchronize()
        # self.prep_times.append(start.elapsed_time(end)/1000.0) # time.time() - start_time)

        
        # e_time = time.time() - e_start_time
        # f_start_time = time.time()

        # e_start.record()
        # e_end.record()
        # torch.cuda.synchronize()
        # e_time = e_start.elapsed_time(e_end)/1000.0
        
        
        # start_time = time.time()
        # start.record()
        # corrs, xyz_offset = utils.track.cross_corr_with_template(search_regions_, templates_)
        corrs, xyz_offset = utils.track.cross_corr_with_templates(search_region, templates)
        # corrs is B x N x Z2 x Z2 x Z2

        
        # f_start.record()
        
        
        # self.corr_times.append(time.time() - start_time)
        # end.record()
        # torch.cuda.synchronize()
        # self.corr_times.append(start.elapsed_time(end)/1000.0) # time.time() - start_time)

        # f_time = time.time() - f_start_time
        
        
        # start_time = time.time()
        # start.record()
        # corrs is B*N x whatever
        _, _, Z2, Y2, X2 = corrs.shape
        # corrs = corrs.reshape(B, N, Z2, Y2, X2)
        corrlist = torch.unbind(corrs, dim=1)
        
        # corr = corrlist[0].unsqueeze(1)
        # xyz_e = utils.track.convert_corr_to_xyz(corr, xyz_offset, hard=False)
        # match_loss = self.smoothl1(xyz_e, xyz_g)
        # total_loss = utils.misc.add_loss('match/match_loss', total_loss, match_loss, hyp.match_coeff, summ_writer)
        # rad_e = torch.zeros_like(xyz_e[:,0])

        # if self.grid is None:
        #     # avoid recomputing this grid on every iter
        #     self.grid = utils.basic.meshgrid3Dr(B, self.radcat, Z2, Y2, X2, stack=False, norm=False, cuda=True)

        # rad_e, xyz_e = utils.track.convert_corrlist_to_xyzr(corrlist, radlist, xyz_offset, hard=False)
        rad_e, xyz_e = utils.track.convert_corrs_to_xyzr(corrs, self.radcat, xyz_offset, hard=False)#, grid=self.grid)
        # print('rad_e', rad_e.shape, rad_e.detach().cpu().numpy())
        # print('rad_g', rad_g.shape, rad_g.detach().cpu().numpy())


        # corrs_ = torch.nn.functional.softmax(corrs.reshape(B, -1)).reshape(B, N, Z2, Y2, X2)
        # conf = utils.samp.bilinear_sample3D(corrs_, (xyz_e - xyz_offset).unsqueeze(1))
        # max_conf = torch.max(conf, dim=1)[0]


        if (xyz_g is not None) and (rad_g is not None):
            
            deg_e = utils.geom.rad2deg(rad_e)
            deg_g = utils.geom.rad2deg(rad_g)

            # print('deg_e', deg_e.shape, deg_e.detach().cpu().numpy())
            # print('deg_g', deg_g.shape, deg_g.detach().cpu().numpy())
            
            total_loss = torch.tensor(0.0).cuda()
            match_loss = self.smoothl1(xyz_e, xyz_g)
            match_loss_r = self.smoothl1(deg_e, deg_g)
            total_loss = utils.misc.add_loss('match/match_loss', total_loss, match_loss, hyp.match_coeff, summ_writer)
            total_loss = utils.misc.add_loss('match/match_loss_r', total_loss, match_loss_r, hyp.match_r_coeff, summ_writer)
            # total_loss = utils.misc.add_loss('match/match_loss_r', total_loss, match_loss_r, 0.0, summ_writer)
        else:
            total_loss = 0.0

        # end.record()
        # torch.cuda.synchronize()
        # self.argmax_times.append(start.elapsed_time(end)/1000.0) # time.time() - start_time)
        # self.argmax_times.append(time.time() - start_time)
        
        # # no rot really
        # assert(N==1)
        # template = templatelist[0]
        # corr, xyz_offset = utils.track.cross_corr_with_template(search_region, template)
        # xyz_e = utils.track.convert_corr_to_xyz(corr, xyz_offset, hard=False)
        # match_loss = self.smoothl1(xyz_e, xyz_g)
        # total_loss = utils.misc.add_loss('match/match_loss', total_loss, match_loss, hyp.match_coeff, summ_writer)
        # rad_e = torch.zeros_like(xyz_e[:,0])
        
        # print('xyz_e:', xyz_e.detach().cpu().numpy())
        
        # print('corrlist[0]', corrlist[0].shape)
        if summ_writer is not None:
            summ_writer.summ_feat('match/input_search_region', search_region, pca=True)
            
            if N > 1:
                for n in list(range(N)):
                    summ_writer.summ_feat('match/input_template_%d' % n, templates[:,n], pca=True)
                    # summ_writer.summ_oned('match/corr_%d', torch.mean(corr[:,n:n+1], dim=3)) # reduce the vertical dim
                    summ_writer.summ_oned('match/corr_%d' % n, torch.mean(corrlist[n].unsqueeze(1), dim=3)) # reduce the vertical dim
            else:
                template = templates[:,0]
                summ_writer.summ_feat('match/input_template', template, pca=True)
                summ_writer.summ_oned('match/corr', torch.mean(corr, dim=3)) # reduce the vertical dim

            # corrlist_ = [torch.mean(corr, dim=3) for corr in corrlist] # reduce the vertical dim
            # summ_writer.summ_oneds('match/corrlist', corrlist_)
            corrlist_ = [corr.unsqueeze(1) for corr in corrlist]
            summ_writer.summ_oneds('match/corrlist', corrlist_, bev=True)

                    

        # if use_window:
        #     z_window = np.reshape(np.hanning(corr.shape[2]), [1, 1, corr.shape[2], 1, 1])
        #     y_window = np.reshape(np.hanning(corr.shape[3]), [1, 1, 1, corr.shape[3], 1])
        #     x_window = np.reshape(np.hanning(corr.shape[4]), [1, 1, 1, 1, corr.shape[4]])
        #     z_window = torch.from_numpy(z_window).float().cuda()
        #     y_window = torch.from_numpy(y_window).float().cuda()
        #     x_window = torch.from_numpy(x_window).float().cuda()
        #     window_weight = 0.25
        #     corr = corr*(1.0-window_weight) + corr*z_window*window_weight
        #     corr = corr*(1.0-window_weight) + corr*y_window*window_weight
        #     corr = corr*(1.0-window_weight) + corr*x_window*window_weight
        # # normalize each corr map, mostly for vis purposes
        # corr = utils.basic.normalize(corr)
            
        # if summ_writer is not None:
        #     summ_writer.summ_oned('match/corr_windowed', torch.mean(corr, dim=3)) # reduce the vertical dim

        
        # print('prep_time', np.mean(self.prep_times))
        # print('corr_time', np.mean(self.corr_times))
        # print('argmax_time', np.mean(self.argmax_times))


        # f_end.record()
        # torch.cuda.synchronize()
        # f_time = f_start.elapsed_time(f_end)/1000.0
        
        
        # return corrs, rad_e, xyz_e, max_conf, total_loss, np.mean(self.corr_times)
        # return corrs, rad_e, xyz_e, max_conf, total_loss, f_time
        return corrs, rad_e, xyz_e, total_loss

