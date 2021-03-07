import torch as torch
import time
import torch.nn.functional as F

B = 1
C = 32

# TEMPLATE_SIZE = 32
SEARCH_SIZE = 64

template_sizes = list(range(1, 23))
timings = []

Z = SEARCH_SIZE
Y = SEARCH_SIZE
X = SEARCH_SIZE
template_ = torch.randn([B, C, Z, Y, X]).float().cuda()
search_region = torch.randn([B, C, Z, Y, X]).float().cuda()

print('search_region', search_region.shape)

# conv_layer = nn.Conv3d(C, 1, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)

for TEMPLATE_SIZE in template_sizes:

    start_time = time.time()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    ZZ = TEMPLATE_SIZE
    ZY = TEMPLATE_SIZE
    ZX = TEMPLATE_SIZE

    start.record()

    # template = torch.zeros([B, C, ZZ, ZY, ZX]).float().cuda()
    template = template_[:,:,:ZZ,:ZY,:ZX]
    data_time = time.time() - start_time

    end.record()
    torch.cuda.synchronize()

    data_time = start.elapsed_time(end)
    
    start.record()

    # print('starting test...')
    start_time = time.time()
    for n in list(range(100)):
        corr = F.conv3d(search_region.view(1, B*C, Z, Y, X), template, groups=B) # fast valid conv
        corr = corr.permute(1, 0, 2, 3, 4)
    corr_ok = corr.clone()
    test_time = time.time() - start_time

    for n in list(range(100)):
        corr = F.conv3d(search_region.view(B, C, Z, Y, X), template)
        corr = corr.permute(1, 0, 2, 3, 4)
    

    end.record()
    torch.cuda.synchronize()

    test_time = start.elapsed_time(end)/1000.0
    
    # torch.cuda.synchronize()
    timings.append(test_time)
    print('template_size = %d; data_time = %.3f; test_time = %.3f' % (TEMPLATE_SIZE, data_time, test_time))

print('timings', timings)
