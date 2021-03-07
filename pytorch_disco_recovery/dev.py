import torch as torch
import time
import torch.nn.functional as F

B = 3
# Cs = [4, 8, 16, 24, 32, 40, 47, 48, 64, 96, 128]
Cs = [32, 33, 34, 36, 38, 40, 46, 48, 64, 96]
# C = 32
for C in Cs:

    SEARCH_SIZE = 32

    timings = []

    Z = SEARCH_SIZE
    Y = SEARCH_SIZE
    X = SEARCH_SIZE
    template_ = torch.randn([B, C, Z, Y, X]).float().cuda()
    search_region = torch.randn([B, C, Z, Y, X]).float().cuda()

    TEMPLATE_SIZE = 22
    ZZ = TEMPLATE_SIZE
    ZY = TEMPLATE_SIZE
    ZX = TEMPLATE_SIZE
    template = template_[:,:,:ZZ,:ZY,:ZX]

    # print('search_region', search_region.shape)
    # print('template', template.shape)

    # what i want to do here is:
    # stack up all the dotprods i will need,
    # and then execute them

    # i need the template at many offsets
    # or, more efficiently, i need many slices of the search region

    def manual_cc(search_region, template):
        B, C, ZZ, ZY, ZX = list(template.shape)
        B2, C2, Z, Y, X = list(search_region.shape)
        assert(B==B2)
        assert(C==C2)

        dz_max = Z-ZZ+1
        dy_max = Y-ZY+1
        dx_max = X-ZX+1

        N = dz_max*dy_max*dx_max
        # print('N', N)

        slices = torch.zeros([B, N, C, ZZ, ZY, ZX]).float().cuda()
        count = 0
        # slices = []
        for dz in list(range(dz_max)):
            for dy in list(range(dy_max)):
                for dx in list(range(dx_max)):
                    sli = search_region[:,:,dz:(dz+ZZ),dy:(dy+ZY),dx:(dx+ZX)]
                    # if dx==0 and dy==0:
                    #     # print('dz start, end:', dz, dz+ZZ+1)
                    #     print('dz start, end:', dz, dz+ZZ)
                    #     print(sli.shape)
                    slices[:, count] = sli
                    count += 1
        slices = slices.view(B, N, -1)
        # print('slices', slices.shape)
        template = template.reshape(B, -1, 1)
        # print('template', template.shape)
        dot = torch.matmul(slices, template)
        # print('dot', dot.shape)
        dot = dot.reshape(B, 1, dz_max, dy_max, dx_max)
        return dot

    # warm up
    # print('warming up...')
    for n in list(range(3)):
        # cor = F.conv3d(search_region.view(1, B*C, Z, Y, X), template, groups=B)
        cor = F.conv3d(search_region, template)
        # dot = manual_cc(search_region, template)

    # print('starting test...')

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for n in list(range(100)):
        for b in list(range(B)):
            search_region_b = search_region[b:b+1]
            template_b = template[b:b+1]
            cor = F.conv3d(search_region_b, template_b)
            # cor = cor.permute(1, 0, 2, 3, 4)
    end.record()
    torch.cuda.synchronize()
    cor_time = start.elapsed_time(end)/1000.0
    print('C', C, 'altcor_time', cor_time)

# interesting:
# there seems to be a check somewhere, relating to C>33, so torch switches convolution algorithms
# C 32 altcor_time 13.87294921875
# C 33 altcor_time 14.3093828125
# C 34 altcor_time 1.199964111328125
# C 36 altcor_time 1.2709488525390624
# C 38 altcor_time 1.3429698486328125
# C 40 altcor_time 1.41473583984375
# C 46 altcor_time 1.6318084716796875
# C 48 altcor_time 1.70513818359375
# C 64 altcor_time 2.288990234375
# C 96 altcor_time 3.452435546875


    
# start.record()
# for n in list(range(100)):
#     cor = F.conv3d(search_region.view(1, B*C, Z, Y, X), template, groups=B)
#     cor = cor.permute(1, 0, 2, 3, 4)
# end.record()
# torch.cuda.synchronize()
# cor_time = start.elapsed_time(end)/1000.0
# print('cor_time', cor_time)


# start.record()
# for n in list(range(100)):
#     dot = manual_cc(search_region, template)
# end.record()
# torch.cuda.synchronize()
# dot_time = start.elapsed_time(end)/1000.0
# print('dot_time', dot_time)


# # print('dot', dot.shape)
# # print('dot[0,0,0,0]', dot[0,0,0,0].detach().cpu().numpy())

# # cor = F.conv3d(search_region.view(1, B*C, Z, Y, X), template, groups=B)
# # print('cor[0,0,0,0]', cor[0,0,0,0].detach().cpu().numpy())

