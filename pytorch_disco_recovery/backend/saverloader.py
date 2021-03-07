import torch
import os,pathlib
import hyperparams as hyp
import numpy as np
import ipdb 
st = ipdb.set_trace

def load_total(model, optimizer):
    start_iter = 0
    if hyp.total_init:
        print("TOTAL INIT")
        print(hyp.total_init)
        start_iter = load(hyp.total_init, model, optimizer)
        if start_iter:
            print("loaded full model. resuming from iter %08d" % start_iter)
        else:
            print("could not find a full model. starting from scratch")
    return start_iter

def load_weights(model, optimizer):
    if hyp.total_init:
        print("TOTAL INIT")
        print(hyp.total_init)
        start_iter = load(hyp.total_init, model, optimizer)
        if start_iter:
            print("loaded full model. resuming from iter %08d" % start_iter)
        else:
            print("could not find a full model. starting from scratch")
    else:
        # if (1):
        start_iter = 0
        inits = {"featnet2D": hyp.feat2D_init,
                 "feat3dnet": hyp.feat3d_init,
                 "feat3dnet_occ": hyp.feat3docc_init,
                 # "upnet3d": hyp.up3d_init,
                 "segnet": hyp.seg_init,
                 "viewnet": hyp.view_init,
                 "segnet": hyp.seg_init,
                 "matchnet": hyp.match_init,
                 "motnet": hyp.mot_init,
                 "confnet": hyp.conf_init,
                 "rigidnet": hyp.rigid_init,
                 "locnet": hyp.loc_init,
                 "vq2dnet": hyp.vq2d_init,
                 "vq3dnet": hyp.vq3d_init,
                 "sigen3dnet": hyp.sigen3d_init,
                 "motionregnet": hyp.motionreg_init,
                 "gen3dnet": hyp.gen3d_init,
                 "forecastnet": hyp.forecast_init,
                 "detnet": hyp.det_init,
                 "visnet": hyp.vis_init,
                 "flownet": hyp.flow_init,
                 "prinet2D": hyp.pri2D_init,
                 "embnet2D": hyp.emb2D_init,
                 # "embnet3d": hyp.emb3d_init, # no params here really
                 "inpnet": hyp.inp_init,
                 "egonet": hyp.ego_init,
                 "feat3dnet": hyp.feat3d_init,
                 "occnet": hyp.occ_init,
                 "occrelnet": hyp.occrel_init,
                 "centernet": hyp.center_init,
                 "preoccnet": hyp.preocc_init,
                 "optim": hyp.optim_init,
                 # "obj": hyp.obj_init,
                 # "bkg": hyp.bkg_init,
                 'localdecodernet_render_occ': hyp.localdecoder_render_init,
                 "localdecodernet": hyp.localdecoder_init,
        }

        for part, init in list(inits.items()):
            # print('loading part', part)
            if init:
                if part == 'featnet2D':
                    model_part = model.featnet2D
                elif part == 'feat3dnet':
                    model_part = model.feat3dnet
                # elif part == 'upnet3d':
                #     model_part = model.upnet3d
                elif part == 'viewnet':
                    model_part = model.viewnet
                elif part == 'segnet':
                    model_part = model.segnet
                elif part == 'matchnet':
                    model_part = model.matchnet
                elif part == 'motnet':
                    model_part = model.motnet
                elif part == 'confnet':
                    model_part = model.confnet
                elif part == 'rigidnet':
                    model_part = model.rigidnet
                elif part == 'locnet':
                    model_part = model.locnet
                elif part == 'vq2dnet':
                    model_part = model.vq2dnet
                elif part == 'vq3dnet':
                    model_part = model.vq3dnet
                elif part == 'sigen3dnet':
                    model_part = model.sigen3dnet
                elif part == 'motionregnet':
                    model_part = model.motionregnet
                elif part == 'gen3dnet':
                    model_part = model.gen3dnet
                elif part == 'forecastnet':
                    model_part = model.forecastnet
                elif part == 'detnet':
                    model_part = model.detnet
                elif part == 'occnet':
                    model_part = model.occnet
                elif part == 'occrelnet':
                    model_part = model.occrelnet
                elif part == 'centernet':
                    model_part = model.centernet
                elif part == 'preoccnet':
                    model_part = model.preoccnet
                elif part == 'prinet2D':
                    model_part = model.prinet2D
                elif part == 'embnet2D':
                    model_part = model.embnet2D
                elif part == 'flownet':
                    model_part = model.flownet
                elif part == 'egonet':
                    model_part = model.egonet
                elif part == 'feat3dnet':
                    model_part = model.feat3dnet
                elif part == 'optim':
                    model_part = model.optim_dict
                elif part == 'localdecodernet':
                    model_part = model.localdecodernet
                elif part == 'localdecodernet_render_occ':
                    model_part = model.localdecodernet_render_occ
                elif part == 'feat3dnet_occ':
                    model_part = model.feat3dnet_occ
                # elif part == 'bkg':
                #     model_part = model.bkg
                # elif part == 'obj':
                #     model_part = [
                #         model.obj,
                #         model.obj_alist,
                #         model.obj_tlist,
                #         model.obj_l,
                #     ]
                else:
                    assert(False)
                if part=='optim':
                    load_optim(model_part, init)
                else:
                    if isinstance(model_part, list):
                        for mp in model_part:
                            iter = load_part([mp], part, init)
                    else:
                        iter = load_part(model_part, part, init)
                if iter:
                    print("loaded %s at iter %08d" % (init, iter))
                else:
                    print("could not find a checkpoint for %s" % init)
    if hyp.reset_iter:
        start_iter = 0
    return start_iter


def save(model, checkpoint_dir, step, optimizer, keep_latest=1):
    model_name = "model-%08d.pth"%(step)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    prev_chkpts = list(pathlib.Path(checkpoint_dir).glob('model-*'))
    prev_chkpts.sort(key=lambda p: p.stat().st_mtime,reverse=True)
    if len(prev_chkpts) > keep_latest-1:
        for f in prev_chkpts[keep_latest-1:]:
            f.unlink()
    path = os.path.join(checkpoint_dir, model_name)
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
        }, path)
    print("Saved a checkpoint: %s"%(path))

def save_optim(model, checkpoint_dir, step):
    print('SAVE_OPTIM: NOT YET SAVING ANY MOMENTUM PARAMS')
    model_name = "model-%08d-optim.npz" % (step)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    path = os.path.join(checkpoint_dir, model_name)
    save_dict = {}
    for k, v in model.optim_dict.items():
        print('saving %s' % k)
        save_dict[k] = v.detach().cpu().numpy()
    np.savez(path, save_dict=save_dict)
    print("Saved an optim checkpoint: %s" % (path))
    
def save_latents(bkg, obj, traj, checkpoint_dir, step):
    model_name = "model-%08d-latents.npz" % (step)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    path = os.path.join(checkpoint_dir, model_name)
    save_dict = {}

    # latent_list = [latent.detach().cpu().numpy()

    # save_dict['latent_list'] = latent_list
    # save_dict['latent_optim_list'] = latent_optim_list
    # save_dict['origin_T_cami_list'] = origin_T_cami_list

    save_dict['bkg'] = bkg
    save_dict['obj'] = obj
    save_dict['traj'] = traj
    # save_dict['latent_optim_list'] = latent_optim_list
    # save_dict['origin_T_cami_list'] = origin_T_cami_list
    
    
    np.savez(path, save_dict=save_dict)
    
    print("Saved an optim checkpoint: %s" % (path))
    
def load_optim(model_optim_dict, load_dir):
    checkpoint_dir = os.path.join("checkpoints/", load_dir)
    print('LOAD_OPTIM: NOT YET LOADING ANY MOMENTUM PARAMS')
    if not os.path.exists(checkpoint_dir):
        print("...ain't no full checkpoint here!")
    else:
        ckpt_names = os.listdir(checkpoint_dir)
        steps = [int((i.split('-')[1]).split('.')[0]) for i in ckpt_names]
        if len(ckpt_names) > 0:
            # ind, step = max(steps)
            # model_name = "model-%08d-optim.npz" % (step)
            ind = np.argmax(steps)
            # model_name = "model-%08d-optim.npz" % (step)
            model_name = ckpt_names[ind]
            path = os.path.join(checkpoint_dir, model_name)
            print('loading from %s' % path)
            load_dict = np.load(path, allow_pickle=True)['save_dict']
            load_dict = load_dict.item()
            # print(load_dict.files)
            # print(o)
            # o = o.item()
            # print(o)
            # load_dict = o.files['save_dict']
            # print(load_dict)
            for k, v in model_optim_dict.items():
                print('loading %s' % k)
                v.data = torch.FloatTensor(load_dict[k]).to(torch.device('cuda'))
            print('done loading')
        else:
            print("...ain't no full checkpoint here!")

def load_latents(load_dir):
    checkpoint_dir = os.path.join("saved_checkpoints/", load_dir)
    print("reading latents checkpoint from %s" % checkpoint_dir)
    if not os.path.exists(checkpoint_dir):
        print("...that checkpoint dir does not even exist!")
        return None, None, None
    ckpt_names = os.listdir(checkpoint_dir)
    ckpt_names = [ckpt for ckpt in ckpt_names if('latents' in ckpt)]
    steps = [int((i.split('-')[1]).split('.')[0]) for i in ckpt_names]
    if len(ckpt_names) > 0:
        ind = np.argmax(steps)
        # print('steps', steps)
        # print('ckpt_names', ckpt_names)
        model_name = ckpt_names[ind]
        # print('model_name', model_name)
        path = os.path.join(checkpoint_dir, model_name)
        print('loading from %s' % path)
        load_dict = np.load(path, allow_pickle=True)['save_dict']
        load_dict = load_dict.item()
        # print(load_dict.files)
        # print(o)
        # o = o.item()
        # print(o)
        # load_dict = o.files['save_dict']
        # print(load_dict)

        bkg = load_dict['bkg']
        obj = load_dict['obj']
        traj = load_dict['traj']

        print('bkg', bkg.shape)
        print('obj', obj.shape)
        print('traj', traj.shape)
        print('got the items')
        
        # print('latent_list', len(latent_list), latent_list[0].shape)
        # print('optim_list', len(latent_optim_list))
        # print('got the items')
        return bkg, obj, traj
    else:
        print("...ain't no full checkpoint here!")
        return None, None, None

def load(model_name, model, optimizer):
    print("reading full checkpoint...")
    # checkpoint_dir = os.path.join("checkpoints/", model_name)
    checkpoint_dir = os.path.join("saved_checkpoints/", model_name)
    step = 0
    if not os.path.exists(checkpoint_dir):
        print("...ain't no full checkpoint here!")
    else:
        ckpt_names = os.listdir(checkpoint_dir)
        steps = [int((i.split('-')[1]).split('.')[0]) for i in ckpt_names]
        if len(ckpt_names) > 0:
            step = max(steps)
            model_name = 'model-%08d.pth' % (step)
            path = os.path.join(checkpoint_dir, model_name)
            print("...found checkpoint %s"%(path))

            checkpoint = torch.load(path)
            
            # # Print model's state_dict
            # print("Model's state_dict:")
            # for param_tensor in model.state_dict():
            #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
            # input()

            # # Print optimizer's state_dict
            # print("Optimizer's state_dict:")
            # for var_name in optimizer.state_dict():
            #     print(var_name, "\t", optimizer.state_dict()[var_name])
            # input()
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print("...ain't no full checkpoint here!")
    return step

def load_part(model, part, init):
    print("reading %s checkpoint..." % part)
    init_dir = os.path.join("saved_checkpoints", init)
    print(init_dir)
    step = 0
    if not os.path.exists(init_dir):
        print("...ain't no %s checkpoint here!"%(part))
    else:
        ckpt_names = os.listdir(init_dir)
        steps = [int((i.split('-')[1]).split('.')[0]) for i in ckpt_names]

        print('ckpt_names', ckpt_names)
        print('steps', steps)
        
        if len(ckpt_names) > 0:
            step = max(steps)
            ind = np.argmax(steps)
            model_name = 'model-%08d.pth' % (step)

            print('setting model name %s instead of %s' % (model_name, ckpt_names[ind]))
            # model_name = ckpt_names[ind]
            path = os.path.join(init_dir, model_name)
            print("...found checkpoint %s"%(path))


            if part=='feat3dnet':
                part2 = 'featnet3D'
                part2 = part
            elif part=='feat3dnet_occ':
                part2 = 'feat3dnet'
            else:
                part2 = part
            
            checkpoint = torch.load(path)
            model_state_dict = model.state_dict()
            print(model_state_dict.keys())
            for load_para_name, para in checkpoint['model_state_dict'].items():
                model_para_name = load_para_name[len(part2)+1:]
                print('model_para_name', model_para_name, 'load_para_name', load_para_name, 'partname', part2)
                if part2+"."+model_para_name != load_para_name:
                    if "_slow" in load_para_name:
                        continue
                    else:
                        print('skipping, because %s.%s does not match %s' % (part2, model_para_name, load_para_name))
                        continue
                else:
                    print('loading this')
                    if model_para_name in model_state_dict.keys():
                        print('for real')
                        # print(model_para_name, load_para_name)
                        print('param in ckpt', para.data.shape)
                        print('param in state dict', model_state_dict[model_para_name].shape)
                        model_state_dict[model_para_name].copy_(para.data)
                    else:
                        print('warning: %s is not in the state dict of the current model' % model_para_name)
        else:
            print("...ain't no %s checkpoint here!"%(part))
    return step

