import sys
sys.path.append('../pymot')
import pymot
import json
import numpy as np


num_tests = 50
motas = []
motps = []

mod = 'mot03'
for test in list(range(1, num_tests+1)):
    
    groundtruth = 'outs/01_s8_m128x32x128_F3s_d64_faks16i3v_ns_%s_%06d_boxes2d_g.json' % (mod, test)
    hypothesis = 'outs/01_s8_m128x32x128_F3s_d64_faks16i3v_ns_%s_%06d_boxes2d_e.json' % (mod, test)
    # hypothesis = 'outs/01_s8_m128x32x128_F3s_d64_faks16i3v_ns_mot02_%06d_boxes2d_e.json' % test
    # hypothesis = 'outs/01_s8_m128x32x128_F3s_d64_faks16i3v_ns_mot02_000001_boxes2d_e.json'
    iou = 0.2

    gt = open(groundtruth) # gt file
    if groundtruth.endswith(".json"):
        # print('gt', groundtruth)
        # print('gt', gt)
        groundtruth = json.load(gt)[0]
    else:
        groundtruth = MOT_groundtruth_import(gt.readlines())
    gt.close()

    # Load MOT format files
    hypo = open(hypothesis) # hypo file
    if hypothesis.endswith(".json"):
        hypotheses = json.load(hypo)[0]
    else:
        hypotheses = MOT_hypo_import(hypo.readlines())
    hypo.close()


    evaluator = pymot.MOTEvaluation(groundtruth, hypotheses, iou)

    # if(check_format):
    #     formatChecker = FormatChecker(groundtruth, hypotheses)
    #     success = formatChecker.checkForExistingIDs()
    #     success |= formatChecker.checkForAmbiguousIDs()
    #     success |= formatChecker.checkForCompleteness()

    #     if not success:
    #         write_stderr_red("Error:", "Stopping. Fix ids first. Evaluating with broken data does not make sense!\n    File: %s" % groundtruth)
    #         sys.exit()

    evaluator.evaluate()

    # print "Track statistics"
    # evaluator.printTrackStatistics()
    # print 
    # print "Results"
    # evaluator.printResults()

    mota = evaluator.getMOTA()
    motp = evaluator.getMOTP()

    print('mota', mota)
    print('motp', motp)

    motas.append(mota)
    motps.append(motp)

mean_mota = np.mean(motas)
mean_motp = np.mean(motps)
print 'mean_mota', mean_mota 
print 'mean_motp', mean_motp 

# 50 iters
# mean_mota 0.955
# mean_motp 0.8164124378962593

# 388 iters
# mean_mota 0.9494201030927835
# mean_motp 0.8311651905412689
