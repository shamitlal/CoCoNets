from exp_base import *

############## choose an experiment ##############

# the idea here is to do some clean bkg explaintraction

current = 'builder'
# current = 'trainer'
current = 'tester'

mod = '"explain00"' # builder; explain start, copied from _sob
mod = '"explain01"' # no cache
mod = '"explain02"' # run explain mode; summ lrtlist
mod = '"explain03"' # summ_feats
mod = '"explain04"' # load from cache else compute
mod = '"explain05"' # summ occs; also get median
mod = '"explain06"' # do not use cache; check if every frame has good amount inbound
mod = '"explain07"' # check if start frame and end frame have 300 inbound
mod = '"explain08"' # min_pts = 1000
mod = '"explain09"' # show occXA0 and occXAE
mod = '"explain10"' # print dists; return right after occ
mod = '"explain11"' # return if dist>20
mod = '"explain12"' # return if dist>10
mod = '"explain13"' # req 2k inb
mod = '"explain14"' # 100 iters; req 1500 inb
mod = '"explain15"' # 100 iters; req 1000 inb
mod = '"explain16"' # return if dist>5
mod = '"explain17"' # show more things
mod = '"explain18"' # narrower rand range for centroid
mod = '"explain19"' # use y of an object
mod = '"explain20"' # put things in RA coords
mod = '"explain21"' # give up after 1 try with a solid centroid
mod = '"explain22"' # 100 frames
mod = '"explain23"' # allow 10m motion
mod = '"explain24"' # build cache
mod = '"explain25"' # switch back to X; build cache
mod = '"explain26"' # use cache < median did not exist
mod = '"explain27"' # use cache again; 
mod = '"explain28"' # get boxes
mod = '"explain29"' # wider bounds; no cache
mod = '"explain30"' # req 
mod = '"explain31"' # req at least 3vox per object
mod = '"explain32"' # 16-16-16
mod = '"explain33"' # req cs>0.55
mod = '"explain34"' # req cs>0.6
mod = '"explain35"' # again
mod = '"explain36"' # compute and return connlist
mod = '"explain37"' # properly req cs>0.6; get eval
mod = '"explain38"' # select a "best" box based on teh summ of the conn
mod = '"explain39"' # print more wrt best; only choose boxes with valid scores
# looks pretty good
# next i need to take that "best" object and track it forward and backward in time
mod = '"explain40"' # repeat but use_cache
mod = '"explain41"' # repeat with cache
mod = '"explain42"' # get more cache
mod = '"explain43"' # generate a mask
mod = '"explain44"' # again, but score=1
mod = '"explain45"' # run some of the test mode; 
mod = '"explain46"' # print the tracking iter
mod = '"explain47"' # proceed < OOB error
mod = '"explain48"' # print shapes and pause
mod = '"explain49"' # use occ_mask
mod = '"explain50"' # again
mod = '"explain51"' # generate traj vis
mod = '"explain52"' # again, but summ on gt occ
mod = '"explain53"' # heat_b = F.relu(heat_b * self.occ_memXAI_median[b:b+1]) < great idea
mod = '"explain54"' # heat_b = F.relu(heat_b * self.diff_memXAI[b:b+1]) < great idea
mod = '"explain55"' # use sigma window
mod = '"explain56"' # window_weight = 0.25; radius 2 instead of 8
mod = '"explain57"' # window_weight = 0.9; use s-1 as the window index
mod = '"explain58"' # use window as a prior too, so that once the object gets to the edge, maybe we just stay there
mod = '"explain59"' # radius 4
# so, i've tracked the object forward
# next i need to track it backward
# then i need to re-compose the top-down scene
mod = '"explain60"' # pack up single-step tracker
mod = '"explain61"' # track down to 0 too
mod = '"explain62"' # crop and pad
mod = '"explain63"' # summ_lrtlist_bev
mod = '"explain64"' # show every 4th box
mod = '"explain65"' # re-compose the scene
mod = '"explain66"' # show box traj one step at a time in bev gif
mod = '"explain67"' # directly generate occ_memY0, to see if it's different
mod = '"explain68"' # show full gif again
mod = '"explain69"' # when composign occ_memY0, do not norm
mod = '"explain70"' # dilate and clamp diff_memXAI during tracking
mod = '"explain71"' # dilate again!
mod = '"explain72"' # dilate just once; window soft=False; show the wnidow < looks a bit better
mod = '"explain73"' # show perspective boxes
mod = '"explain74"' # radius 8
mod = '"explain75"' # return early with inb check with pad=1
mod = '"explain76"' # show perspective vis
mod = '"explain77"' # return if sum(c_mask) < 2
mod = '"explain78"' # check inb with alrady_mem and crop_vec
mod = '"explain79"' # padding=2 
mod = '"explain80"' # alt and less-differentiable of assembling the hypothesis: create the mask and mult
mod = '"explain81"' # just compose the scene feats, and run occnet
mod = '"explain82"' # coeff 1.1 and additive 0.5
mod = '"explain83"' # actually just 1.0 and no additive, bc i am ruining the ground otherwise

# the next big step is:
# re-run the differencing with the newly assembled scenes
# do not lose the objects collected so far

# instead of doing this the full way, i can use a shortcut right now, which is just to re-run the boxing, with the locations masked out

mod = '"explain84"' # repeat
mod = '"explain85"' # pack up box proposer; get blue vis and return
mod = '"explain86"' # transform coords; continue
mod = '"explain87"' # 1 iter of super loop
mod = '"explain88"' # pack up tracker
mod = '"explain89"' # also return top occ and feat
mod = '"explain90"' # diff*1.0-mask; two super iters
mod = '"explain91"' # 4 super iters
mod = '"explain92"' # 3 super; zero out scores of early returns; print info about biggest
mod = '"explain93"' # use score in bev vis
mod = '"explain94"' # use padding=0.5 for discard
mod = '"explain95"' # fix bug in proposal vis
mod = '"explain96"' # when obj_occ is too weak, use the full mask
mod = '"explain97"' # show diff_memXAI_all no each super iter
mod = '"explain98"' # for suppressing diffs, use a wider mask
mod = '"explain99"' # again but do nto summ outide of the super loop
mod = '"explain100"' # additive 1.5 on suppress
mod = '"explain101"' # pad 1.0 for objects too small to track
# yes, with this thing turned on, we just try to assign each voxel to some part of the apparent foreground.
mod = '"explain102"' # turn off bkg supression
mod = '"explain103"' # show best box with its size
mod = '"explain104"' # pad=2 for inb
mod = '"explain105"' # pad=3 for inb
mod = '"explain106"' # half-weight new stuff
mod = '"explain107"' # score=0 for objects that move <1m
mod = '"explain108"' # do not suppress bkg
mod = '"explain109"' # hard window
mod = '"explain110"' # score=0 for objects that move <2m
mod = '"explain111"' # never use occ*mask; just use the mask directly
# hm, i've lost many objects around here
mod = '"explain112"' # scores default 1, to catch bug where the seed frame is unlablled
mod = '"explain113"' # if TOTAL TRAVEL is <2m, discard
mod = '"explain114"' # 50 iters
mod = '"explain115"' # S_test = 90
mod = '"explain116"' # allow 20m movement
mod = '"explain117"' # again, to take advantage of computed features
mod = '"explain118"' # do not zero out boxes, unless they go OOB
mod = '"explain119"' # S_test = 50; req <10m motion
mod = '"explain120"' # eliminate diffs within 2vox of border, to prevent proposals near edges
mod = '"explain121"' # proper try < ok good, the inits are farther away from edges

# next two things:
# > some basic motion smoothness/forecasting, partly to help us avoid losing the object, and partly bc it makes a lot of sense with the story
# >> note this may help at both ends of the frame20 issue, where the object gets occluded (frame0) and leaves fov (frame50)
# > cycle consistency directly in the cost volumes, before bothering ransac; zero out correspondences whose nearest neighbor is not symmetric
# > voxel-centric max-displacement bounds

mod = '"explain122"' # 20 iters
mod = '"explain123"' # show summary with all tracklets
mod = '"explain124"' # use proper colors/tids in teh summ
mod = '"explain125"' # add offset of 2 for esxtimated box clors
mod = '"explain126"' # add full eval, at least with per-frame AP
mod = '"explain127"' # add eval at proposal stage too
mod = '"explain128"' # only eval iter0 proposals
# ok, it appears we are not usually selecting the best box. AP is lower for the tracker vs the proposer
mod = '"explain129"' # add some motion forecasting; use const velocity when OOB
mod = '"explain130"' # avoid inverse of cam0_T_obj
mod = '"explain131"' # only track objects of nonzero size
mod = '"explain132"' # use the prior with padding=4 instead of 3
mod = '"explain133"' # put score=0.5 when using the prior, so that maybe it shows up
mod = '"explain134"' # allow one more lrt to exist for prior
mod = '"explain135"' # padding=6 for inb
mod = '"explain136"' # do that old inverse; print t_prev
mod = '"explain137"' # 10 iters, since i see a bug on iter9; print the info i'm using
mod = '"explain138"' # do that for the backwards one too
mod = '"explain139"' # fix const vel bug < ok finally
mod = '"explain140"' # 20 iters
mod = '"explain141"' # padding=4.0 for inb
mod = '"explain142"' # return prior if <3 inbound < this makes no sense; it's a centroid check
mod = '"explain143"' # S = 90 < mostly just pads the beginning of the video; not helpful

# now please cycle consistency
mod = '"explain144"' # S = 50;  make the windows thing per-voxel, as a step toward symmetric tracking
mod = '"explain145"' # compute that window in the cropped resolution
mod = '"explain146"' # radius 4.0 instead of 8
mod = '"explain147"' # summ every window
mod = '"explain148"' # do not reduce n
mod = '"explain149"' # permute
mod = '"explain150"' # don't bother summing the windows; they look ok
mod = '"explain151"' # radius 6
# actually this method is totally wrong; it's putting a search radius around the zeroth location of each voxel 
# what i really want is a search radius around the last known location of each voxel;
# i'm not really saving these locations right now
mod = '"explain152"' # no window < gets crazy
mod = '"explain153"' # extract correspI and show its shape
mod = '"explain154"' # get cycle dists; deleted a z*10
mod = '"explain155"' # go through;
mod = '"explain156"' # use centroid window < somehow, i am getting constant stats
mod = '"explain157"' # use reverse window centered at mean(xyzI)
mod = '"explain158"' # use crop vec when sampling
mod = '"explain159"' # print xyz0, xyzI, revI, pause
mod = '"explain160"' # print xyz0, xyzI, revI, pause < ok, the reversed_xyzI are all the same, 39.5,39.5,39.5
mod = '"explain161"' # elim the window < somehow, fixed. wider stats span now.
mod = '"explain162"' # only proceed with dists <5vox; assert(False) if that does not pass
# this does improve mAP a bit! but i worry it's fragile now
mod = '"explain163"' # print how many corresps are proceeding
mod = '"explain164"' # eliminate that max subtraction, so that i can more easily zero out bad corresps
mod = '"explain165"' # zero out, and get a second cycle dist
mod = '"explain166"' # zero out a couple more
mod = '"explain167"' # do that again, but i zero out different ones and collect more stats
mod = '"explain168"' # one more time but hard=True, so that i really set the max to zero
mod = '"explain169"' # do 4 cycles < ok, mean and max tend to decrease
mod = '"explain170"' # 5 cycles and proceed < uh, now it is not proceeding
mod = '"explain171"' # hard=False
mod = '"explain172"' # thresh 4.0 instead of 5.0
mod = '"explain173"' # 20 iters
mod = '"explain174"' # 8 cycles instead of 5
mod = '"explain175"' # take hard argmaxes during non-cycle rejection
mod = '"explain176"' # use the predicted centroid for the oob, and for the window; also radius 4.0
mod = '"explain177"' # max 16 cycles
mod = '"explain178"' # border=4 
mod = '"explain179"' # 1k iters
mod = '"explain180"' # cycle thresh 4 instead of 5
mod = '"explain181"' # cycle thresh 3 instead of 4; 10k iters
mod = '"explain182"' # border=2
mod = '"explain183"' # clear the cache; 1k iters
mod = '"explain184"' # use the cache
mod = '"explain185"' # add 2.0 instead of 1.5, to suppress more < pretty narrow, but let's say it wins
mod = '"explain186"' # add 2.0 instead of 1.5, to suppress more
mod = '"explain187"' # if total travel dist is <2m, score=0
mod = '"explain188"' # as sanity check, eval gt against gt for the track ap
mod = '"explain189"' # comment out everything els

# add summary stats
# start on the ablations
# generate results for the flow-based model

mod = '"explain190"' # tester
mod = '"explain191"' # 2 super iters instead of 3; 4 cycles instead of 16 < this is 2x faster and same accuracy. why aren't the cycles helping? but this is only one valid data point. 
mod = '"explain192"' # 20 iters
mod = '"explain193"' # collect and summ all maps
mod = '"explain194"' # hard=True for final argmax < apparently zero difference in numbers
mod = '"explain195"' # report the mean across S
mod = '"explain196"' # hard=True for the first xyzI
# mean_proposal_maps [0.36 0.3  0.22 0.19 0.14 0.06 0.  ]
# mean_track_maps [0.36 0.3  0.22 0.19 0.14 0.06 0.  ]
# explain195 has the exact same result
mod = '"explain197"' # 16 cycles
# mean_proposal_maps [0.36 0.3  0.22 0.19 0.14 0.06 0.  ]
# mean_track_maps [0.36 0.3  0.22 0.19 0.14 0.06 0.  ]
# exactly the same...
# in tb this one looks better
# i must have a bug somewhere
mod = '"explain198"' # 1 cycles; fewer prints
# mean_proposal_maps [0.36 0.3  0.22 0.19 0.14 0.06 0.  ]
# mean_track_maps [0.36 0.3  0.22 0.19 0.14 0.06 0.  ]
# ah, the bug is i am not printing track maps at all
mod = '"explain199"' # properly print track maps
# mean_proposal_maps [0.36 0.3  0.22 0.19 0.14 0.06 0.  ]
# mean_track_maps [0.41 0.35 0.3  0.22 0.07 0.01 0.  ]
mod = '"explain200"' # 4 cycles
# mean_proposal_maps [0.36 0.3  0.22 0.19 0.14 0.06 0.  ]
# mean_track_maps [0.32 0.27 0.24 0.17 0.07 0.01 0.  ]
# ok congrats, cycles are worsening perf here.
mod = '"explain201"' # 16 cycles
# mean_track_maps [0.4  0.33 0.26 0.17 0.05 0.01 0.  ]
mod = '"explain202"' # no cycles
# mean_track_maps [0.41 0.36 0.3  0.2  0.08 0.01 0.  ]

# i sure would like to have voxel-centric max displacements

mod = '"explain203"' # per-voxel window with radius 3
# mean_track_maps [0.29 0.22 0.17 0.13 0.06 0.   0.  ]
# worse. i think the objects got stuck.
mod = '"explain203"' # per-voxel window with radius 6
# mean_track_maps [0.32 0.27 0.23 0.18 0.08 0.   0.  ]
# ok, still worse than the global window
mod = '"explain204"' # do not discard stationary, so i can see what's happening
# yes, i think the radius is not happening properly 
mod = '"explain205"' # print the xyzI prior and the clist  < ah, i was not using the right memI_T_mem0
mod = '"explain206"' # use the right meMI
# ok
# mean_track_maps [0.33 0.3  0.27 0.2  0.12 0.03 0.  ]
mod = '"explain207"' # discard stationary again
# mean_track_maps [0.34 0.3  0.27 0.2  0.12 0.03 0.  ]
mod = '"explain208"' # again, since i'm not sure that happened
mod = '"explain209"' # use box cache; stationary dist 3
# mean_track_maps [0.34 0.31 0.28 0.21 0.13 0.03 0.  ]
mod = '"explain210"' # stationary dist 2 again; radius 3 per voxel
# mean_track_maps [0.41 0.36 0.32 0.22 0.12 0.03 0.  ]
mod = '"explain211"' # properly load cache
mod = '"explain212"' # include super_iter in cache fn
mod = '"explain213"' # load that cache
# ok works
mod = '"explain214"' # radius 2 for voxels
# mean_track_maps [0.39 0.35 0.28 0.2  0.12 0.03 0.  ]
mod = '"explain215"' # print more, to help me understand where these values come from, and why it disagrees with tb
mod = '"explain216"' # do not discard map=0 in the tb summ
# mean_track_maps [0.39 0.35 0.28 0.2  0.12 0.03 0.  ]
mod = '"explain217"' # print less
mod = '"explain218"' # 4 cycles 
# mean_track_maps [0.39 0.34 0.29 0.19 0.07 0.01 0.  ]
mod = '"explain219"' # 8 cycles; print about those cyc
# the cyc dists do not seem to be improving; probably the issue is the window size
# mean_track_maps [0.39 0.34 0.29 0.19 0.07 0.01 0.  ]
mod = '"explain220"' # radius 4 instead of 2; no cyc
# mean_track_maps [0.35 0.31 0.29 0.21 0.12 0.03 0.  ]
# actually better at the higher ious
mod = '"explain221"' # 2cyc and fix bug in thing
# mean_track_maps [0.43 0.39 0.36 0.23 0.11 0.01 0.  ]
# ok encouraging help!
mod = '"explain222"' # 2cyc and fix bug in thing and use xyz0 window on reverse
# mean_track_maps [0.35 0.31 0.29 0.2  0.12 0.03 0.  ]
# worse! so that reverse cycle window is not helping me. maybe that makes sense. give the reverse guy a wider fov.
mod = '"explain223"' # again 4cyc; no reverse window
# mean_track_maps [0.43 0.39 0.36 0.24 0.1  0.01 0.  ]
# this is about the same as the effect of 2cyc
mod = '"explain224"' # 50 iters
# mean_proposal_maps [0.46 0.36 0.24 0.2  0.13 0.04 0.  ]
# mean_track_maps [0.4  0.35 0.28 0.17 0.05 0.01 0.  ]
mod = '"explain225"' # if total < 2 or endpoint < 1, call it stationary
# mean_proposal_maps [0.46 0.36 0.24 0.2  0.13 0.04 0.  ]
# mean_track_maps [0.4  0.35 0.28 0.17 0.05 0.01 0.  ]
# (identical)
mod = '"explain226"' # no cyc; compute-0-16
# mean_proposal_maps [0.46 0.36 0.24 0.2  0.13 0.04 0.  ]
# mean_track_maps [0.39 0.33 0.25 0.16 0.07 0.02 0.  ]
# ok, better on the 0.5 iou but tw worse
mod = '"explain227"' # no cyc; compute-0-16; 100 iters
# mean_track_maps [0.37 0.32 0.25 0.17 0.09 0.01 0.  ]
mod = '"explain228"' # cyc 8
# mean_track_maps [0.39 0.35 0.27 0.18 0.07 0.01 0.  ]
mod = '"explain229"' # cyc 8; radius=8 instead of 4
# mean_track_maps [0.38 0.33 0.26 0.18 0.08 0.01 0.  ]
# 


mod = '"mot00"' # generate flows, mag, and propose boxes on there 
mod = '"mot01"' # again
mod = '"mot02"' # gaussian blur on flow05
mod = '"mot03"' # 100 iters
# mean_proposal_maps [0.07 0.04 0.01 0.   0.   0.   0.  ]
mod = '"mot04"' # blur the short flows THEN chain
# mean_proposal_maps [0.07 0.04 0.01 0.   0.   0.   0.  ]
# on tb it looked different, but i guess it averaged out?
mod = '"mot05"' # no blur; flow05 = flow05 * consistency_mask
mod = '"mot06"' # use flow05 * consistency_mask * occ0
mod = '"mot07"' # only discard at <0.50
mod = '"mot08"' # only discard at <0.51; use flow05 * occ0
mod = '"mot09"' # mult by gt occ instead
mod = '"mot10"' # mult occ by flow BEFORE chaining < somehow these flows are larger
mod = '"mot11"' # AND consistency mask
mod = '"mot12"' # blur
# mean_proposal_maps [0.16 0.13 0.07 0.02 0.   0.   0.  ]

mod = '"mot13"' # no mask; blur
mod = '"mot14"' # no mask; *occ0 then blur
# [0.09 0.07 0.05 0.02 0.   0.   0.  ]
mod = '"mot15"' # blur, then backwarp, then consistency mask, then occ0
# mean_proposal_maps [0.17 0.1  0.04 0.01 0.   0.   0.  ]
# close but not entirely better than mot12
mod = '"mot16"' # occ before chain, then consistency, then blur (hopefully replicating moc12) < yes, it's identical
mod = '"mot17"' # dilate occs; occ the backwards before chaining too
# bad
mod = '"mot18"' # blur, then chain, then cons, then mult by dilated occ0
# slightly worse
mod = '"mot19"' # hard thresh 0.55; 
mod = '"mot20"' # try mot16, but discard <0.55 < forgot to occ 
mod = '"mot21"' # occ both directions
# mean_proposal_maps [0.07 0.05 0.03 0.01 0.   0.   0.  ]
# pretty bad!
mod = '"mot22"' # chain, cons, blur, occ
# mean_proposal_maps [0.14 0.07 0.03 0.01 0.   0.   0.  ]
mod = '"mot23"' # print grid
mod = '"mot24"' # mult by estimated occ0 instead of gt occ0
# bad
mod = '"mot25"' # identity flows, hopefully
mod = '"mot26"' # do not /0.07
mod = '"mot27"' # regular flow again
mod = '"mot28"' # avoid the consistency check
mod = '"mot29"' # do /0.07


mod = '"explain230"' # no flow; no cyc; radius 4
# proposal maps:[0.43 0.33 0.21 0.14 0.07 0.02 0.  ]
# ok this matches explain226 for the first 50 iters, then produces more. so i think this is the model. 
mod = '"explain231"' # allow any camera motion
mod = '"explain232"' # again; only evaluate on moving scenes
mod = '"explain233"' # go through that again, using the later 232 feats to compute more results
mod = '"explain234"' # use scores*diff_per_step and also check endpoint for 2m instead of 1m < seems identical to explain233
# proposal maps: [0.58 0.36 0.24 0.16 0.09 0.01 0.  ]
mod = '"explain235"' # use scores1*diff, considering that the prior is has score=0.5
# proposal maps:[0.58 0.36 0.24 0.16 0.09 0.01 0.  ]
# interesting. this is actually better than the static-cam data
mod = '"explain236"' # measure bev iou too
# mean_proposal_maps_3d [0.58 0.36 0.24 0.16 0.09 0.01 0.  ]
# mean_proposal_maps_2d [0.65 0.61 0.56 0.5  0.38 0.26 0.13]
# mean_track_maps_3d [0.44 0.32 0.27 0.19 0.08 0.01 0.  ]
# mean_track_maps_2d [0.59 0.55 0.47 0.39 0.29 0.2  0.08]
# nice.
# this is hard to beat.

# next priorities:
# on each step, re-center teh box according to some random jitter and a CS check
# zoom in during tracking

mod = '"explain237"' # add noise and evaluate four alts
mod = '"explain238"' # print the noise
mod = '"explain239"' # use noise*0.1
# mean_proposal_maps_3d [0.58 0.36 0.24 0.16 0.09 0.01 0.  ]
# mean_proposal_maps_2d [0.65 0.61 0.56 0.5  0.38 0.26 0.13]
# mean_track_maps_3d [0.45 0.32 0.28 0.19 0.07 0.   0.  ]
# mean_track_maps_2d [0.57 0.54 0.46 0.4  0.29 0.2  0.09]
# this is slightly better at the higher ious, but not a clear win
mod = '"explain240"' # jitter forward and backward
# mean_proposal_maps_3d [0.58 0.36 0.24 0.16 0.09 0.01 0.  ]
# mean_proposal_maps_2d [0.65 0.61 0.56 0.5  0.38 0.26 0.13]
# mean_track_maps_3d [0.42 0.3  0.23 0.16 0.05 0.   0.  ]
# mean_track_maps_2d [0.53 0.5  0.41 0.35 0.24 0.16 0.05]
# worse.
mod = '"explain241"' # jitter5 the initial box
# mean_track_maps_3d [0.46 0.32 0.25 0.19 0.06 0.01 0.  ]
# mean_track_maps_2d [0.56 0.52 0.44 0.35 0.24 0.17 0.07]
mod = '"explain242"' # jitter8 the initial box
# mean_track_maps_3d [0.42 0.3  0.26 0.19 0.06 0.   0.  ]
# mean_track_maps_2d [0.53 0.49 0.42 0.36 0.25 0.18 0.06]
# ok not bad by the end. halfway through, it was not winning
mod = '"explain243"' # jitter8 the initial box, and allow everything to change
# mean_track_maps_3d [0.42 0.3  0.26 0.19 0.06 0.   0.  ]
# mean_track_maps_2d [0.53 0.49 0.42 0.36 0.25 0.18 0.06]
mod = '"explain244"' # use diff for the CS
# mean_track_maps_3d [0.39 0.28 0.24 0.16 0.07 0.01 0.  ]
# mean_track_maps_2d [0.53 0.48 0.38 0.31 0.23 0.19 0.07]
# this helps a teeny bit for the higher iou thresholds, but not for the rest. maybe that's ok!
mod = '"explain245"' # allow everything to jitter
mod = '"explain246"' # similar but 1.2 and additive coeff 1.0, instead of 1.8,0.0
# mean_track_maps_3d [0.42 0.29 0.24 0.15 0.04 0.01 0.  ]
# mean_track_maps_2d [0.54 0.5  0.42 0.36 0.24 0.18 0.07]
# ok good recovery at the end, but this is still close to 242, and a bit worse on box3d iou
mod = '"explain247"' # in forward tracking, re-center
mod = '"explain248"' # use zoom bounds in forward tracking; delta*0 on backward tracking
# not great
mod = '"explain249"' # eliminate the windowing
# made partway through;
# 0.19 0.17 0.15 0.12 0.09 0.07 0.04
# something missing here 
mod = '"explain250"' # summ the search regions; also, use zooming for backward
mod = '"explain251"' # zoom properly at the start too, so that the features are the right size
# mean_track_maps_3d [0.45 0.28 0.22 0.18 0.09 0.   0.  ]
# mean_track_maps_2d [0.51 0.48 0.39 0.35 0.26 0.18 0.07]
# ok, this helps the 3d line a bit. not the other. 
mod = '"explain252"' # put the centroid at the estimated next position, instead of last position
mod = '"explain253"' # use last position again; jitter using gt occ
# mean_track_maps_3d [0.37 0.29 0.2  0.13 0.06 0.   0.  ]
# mean_track_maps_2d [0.45 0.41 0.35 0.32 0.22 0.13 0.06]
# worse.
mod = '"explain254"' # when zooming, do not use such a wide y dim please; use the hyp.*_zoom
# worse.
mod = '"explain255"' # 8-8-8 bounds again
# mean_track_maps_3d [0.33 0.23 0.2  0.16 0.05 0.01 0.  ]
# mean_track_maps_2d [0.47 0.43 0.35 0.29 0.22 0.17 0.06]
# not great 
mod = '"explain256"' # do use the window, radius=4 < hurts
# worse.

# mod = '"explain257"' # no jitter
# mod = '"explain258"' # 12-12-12 bounds

# resolve the issue of oob wrt the scene

mod = '"explain257"' # jitter; 8-8-8; center on prev; when scene-oob, use prior (instead of when zoom-oob)
mod = '"explain258"' # radius=8 on the window
# not great
mod = '"explain259"' # no window
# mean_track_maps_3d [0.33 0.24 0.21 0.16 0.05 0.   0.  ]
# mean_track_maps_2d [0.42 0.38 0.33 0.27 0.2  0.16 0.08]
mod = '"explain260"' # padding=0 for oob
# mean_track_maps_3d [0.34 0.23 0.19 0.13 0.06 0.02 0.01]
# mean_track_maps_2d [0.4  0.36 0.32 0.28 0.22 0.15 0.08]
# not that helpful
mod = '"explain261"' # 12-12-12
# mean_track_maps_3d [0.3  0.21 0.18 0.13 0.06 0.01 0.  ]
# mean_track_maps_2d [0.4  0.38 0.31 0.26 0.21 0.16 0.06]
# pretty even
mod = '"explain262"' # window with radius 6
# mean_track_maps_3d [0.25 0.17 0.15 0.12 0.04 0.   0.  ]
# mean_track_maps_2d [0.27 0.27 0.23 0.21 0.17 0.12 0.06]
# worse!
mod = '"explain263"' # first jitter12, with occ_memXAI_all[I]
# mean_track_maps_3d [0.24 0.19 0.15 0.11 0.02 0.   0.  ]
# mean_track_maps_2d [0.28 0.26 0.23 0.2  0.16 0.12 0.05]
# similar 
mod = '"explain264"' # window with radius 3
# mean_track_maps_3d [0.2  0.14 0.11 0.08 0.02 0.   0.  ]
# mean_track_maps_2d [0.21 0.2  0.17 0.15 0.12 0.07 0.04]
# worse yet. this window is not helping at all
mod = '"explain265"' # 16-16-16 zoom
# mean_track_maps_3d [0.21 0.15 0.12 0.08 0.03 0.   0.  ]
# mean_track_maps_2d [0.23 0.22 0.17 0.15 0.13 0.09 0.03]
mod = '"explain266"' # use_window=False
# mean_track_maps_3d [0.25 0.19 0.16 0.1  0.04 0.01 0.  ]
# mean_track_maps_2d [0.3  0.28 0.24 0.21 0.16 0.09 0.03]
# yeah that window sucks
mod = '"explain267"' # 8-4-8 zoom, padding=2 for oob
mod = '"explain268"' # use occ*mask
# mean_track_maps_3d [0.36 0.28 0.24 0.18 0.05 0.   0.  ]
# mean_track_maps_2d [0.41 0.39 0.33 0.29 0.24 0.17 0.08]
mod = '"explain269"' # 8-8-8 zoom
# mean_track_maps_3d [0.36 0.27 0.22 0.17 0.05 0.01 0.  ]
# mean_track_maps_2d [0.43 0.38 0.32 0.26 0.24 0.18 0.07]
# ok, i think this is better than i've ever done in zoom
mod = '"explain270"' # 3 super iters
# mean_track_maps_3d [0.47 0.32 0.22 0.18 0.07 0.   0.  ]
# mean_track_maps_2d [0.56 0.5  0.41 0.33 0.28 0.2  0.12]
# yes thtat's better
mod = '"explain271"' # 4 super iters
# mean_track_maps_3d [0.5  0.34 0.24 0.15 0.05 0.01 0.  ]
# mean_track_maps_2d [0.57 0.54 0.49 0.4  0.29 0.21 0.11]
# this helps overall
mod = '"explain272"' # when jittering, constrain to 0.5m
# mean_track_maps_3d [0.49 0.35 0.25 0.13 0.07 0.   0.  ]
# mean_track_maps_2d [0.57 0.54 0.48 0.41 0.29 0.16 0.09]
# looks like no effect, or it hurts even
mod = '"explain273"' # return the prior if occrel is unsure within the curr mask
# mean_track_maps_3d [0.54 0.37 0.28 0.21 0.08 0.   0.  ]
# mean_track_maps_2d [0.6  0.58 0.51 0.43 0.34 0.22 0.11]
# this helps a bit
# this is basically a new winner

mod = '"explain274"' # suppress with 2.5 instead of 2.0
# mean_track_maps_3d [0.53 0.36 0.26 0.16 0.05 0.01 0.  ]
# mean_track_maps_2d [0.59 0.56 0.51 0.43 0.33 0.18 0.09]

mod = '"explain275"' # back to 2.0; edge border=3 
# mean_track_maps_3d [0.56 0.39 0.27 0.15 0.05 0.01 0.  ]
# mean_track_maps_2d [0.61 0.59 0.52 0.45 0.35 0.19 0.1 ]
# not consistnetly better



# katerina says:
# coarse-to-fine at the proposal stage
# zoom in after proposing, to get even better proposals within the region

# this makes a lot of sense
mod = '"zoom00"' # zoom median then return
mod = '"zoom01"' # return True
mod = '"zoom02"' # show zoom diffs
mod = '"zoom03"' # use dilated vis
mod = '"zoom04"' # get the zoom blue boxes
mod = '"zoom05"' # do not use box cache in the zoom boxing 
mod = '"zoom06"' # properly zoom05
mod = '"zoom07"' # track the biggest zoom object

# i need to suppress the target also in the zoom 

mod = '"zoom08"' # suppress
# qualitatively: it seems the boxes shrink a bit due to teh zoom
# mean_track_maps_3d [0.28 0.21 0.14 0.12 0.08 0.01 0.  ]
# mean_track_maps_2d [0.43 0.37 0.31 0.25 0.2  0.15 0.1 ]
# not great
mod = '"zoom09"' # dilate the vis twice more
# qualitatively: sometimes a good box gets dropped here
mod = '"zoom10"' # only use best_I +- 2, to not drop the basic anchor; be careful with inds
# mean_track_maps_3d [0.3  0.21 0.1  0.05 0.03 0.01 0.  ]
# mean_track_maps_2d [0.34 0.31 0.25 0.19 0.11 0.05 0.02]
# terrible
mod = '"zoom11"' # cleaned up some vs all notation slightly
mod = '"zoom12"' # fixed a bug w size (computed via wrong connlist)
# some big too-boxes slip in
mod = '"zoom13"' # only dilate vis once < crashed due to ind_some oob
mod = '"zoom14"' # req cs 0.65 in high res < crashed due to ind_some oob
mod = '"zoom15"' # show zoom feats and occs; use np.min/max for the some inds
mod = '"zoom16"' # do not give up as easily: req occrel <4 to return
mod = '"zoom17"' # ind_some = [best_I]
# apparent bug: we are re-discovering the same object, even on the same frame, which means it is not being suppressed
mod = '"zoom18"' # 20 iters; inspect the top3d masks
mod = '"zoom19"' # do that again but show each super iter's top3d mask
mod = '"zoom20"' # 15 iters
mod = '"zoom21"' # print size and frame for zoom and bev
mod = '"zoom22"' # do not use box cache ever < great bug
mod = '"zoom23"' # use padding=1 for centroid inbound
# redo that, since somehow only 1 object was found on iter15
mod = '"zoom24"' # more prints
# somehow, a great potential object got junked during blue boxing
mod = '"zoom25"' # only reject boxes<0.5 CS
mod = '"zoom26"' # return if global_step < 14
mod = '"zoom27"' # do not discard any boxes based on CS
# setting super_iters=1 to speed up debugging of that first box
mod = '"zoom28"' # super_iters=1
mod = '"zoom29"' # border=1 on the thing
mod = '"zoom30"' # summ diff_iter within propose
mod = '"zoom31"' # no border
# ok, i see the diff right there. 
mod = '"zoom32"' # print threshs
mod = '"zoom33"' # eliminate "if thresH > 0.5"
mod = '"zoom34"' # print about the segments
mod = '"zoom35"' # avoid the "less than huge" constraint
mod = '"zoom36"' # avoid the "cetner is occupeid" constrainte < ok! both of these were necessary apparently
mod = '"zoom37"' # discard boxes < 0.55 cs
mod = '"zoom38"' # fewer prints; 3 super iters
mod = '"zoom39"' # occrel better be 2 or less before we give up
mod = '"zoom40"' # padding=0 on inb
mod = '"zoom41"' # use fancier (smoother) prior < nan!
mod = '"zoom42"' # print 
mod = '"zoom43"' # print  more
mod = '"zoom44"' # use const vel
mod = '"zoom45"' # update clist
mod = '"zoom46"' # compute prevprev
mod = '"zoom47"' # no prevprev
mod = '"zoom48"' # vel in the other direction
# nice. very high perf on that iter
mod = '"zoom49"' # do not skip the first 14 iters
mod = '"zoom50"' # 100 iters
# mean_track_maps_3d [0.39 0.31 0.23 0.17 0.07 0.03 0.01]
# mean_track_maps_2d [0.47 0.44 0.38 0.31 0.24 0.2  0.12]
mod = '"zoom51"' # use padding=2 for inb
# mean_track_maps_3d [0.38 0.3  0.23 0.14 0.06 0.01 0.  ]
# mean_track_maps_2d [0.47 0.43 0.4  0.33 0.28 0.18 0.1 ]
mod = '"zoom52"' # super_iters = 4
# mean_track_maps_3d [0.38 0.25 0.2  0.17 0.08 0.03 0.01]
# mean_track_maps_2d [0.5  0.45 0.38 0.3  0.24 0.21 0.13]
mod = '"zoom53"' # cs thresh 0.6 for discarding objects, instead of 0.55
# mean_track_maps_3d [0.3  0.24 0.14 0.1  0.05 0.02 0.  ]
# mean_track_maps_2d [0.41 0.4  0.35 0.29 0.22 0.16 0.09]
# this is actually worse. so let's go back to 0.55
mod = '"zoom54"' # use padding=1 for inb, instead of 2
# mean_track_maps_3d [0.31 0.26 0.19 0.12 0.05 0.   0.  ]
# mean_track_maps_2d [0.43 0.41 0.37 0.32 0.23 0.16 0.07]
# ok slight bonus all around. let's keep this.
mod = '"zoom55"' # use_window=True, radius=4
mod = '"zoom56"' # super_iters = 3
# mean_track_maps_3d [0.27 0.22 0.18 0.12 0.05 0.01 0.  ]
# mean_track_maps_2d [0.33 0.31 0.28 0.24 0.19 0.14 0.08]
mod = '"zoom57"' # do not necessarily discard the coarse object
# mean_track_maps_3d [0.3  0.23 0.18 0.13 0.06 0.02 0.  ]
# mean_track_maps_2d [0.34 0.32 0.29 0.24 0.2  0.16 0.11]
# good plan. 
mod = '"zoom58"' # use_window=False
# mean_track_maps_3d [0.4  0.3  0.22 0.13 0.06 0.01 0.  ]
# mean_track_maps_2d [0.47 0.44 0.39 0.34 0.24 0.2  0.11]
mod = '"zoom59"' # only discard < 0.55
# mean_track_maps_3d [0.41 0.35 0.26 0.16 0.06 0.01 0.  ]
# mean_track_maps_2d [0.51 0.49 0.44 0.36 0.28 0.22 0.11]
mod = '"zoom60"' # 4 super iters
# mean_track_maps_3d [0.34 0.26 0.17 0.12 0.05 0.01 0.  ]
# mean_track_maps_2d [0.49 0.46 0.39 0.27 0.2  0.14 0.09]
mod = '"zoom61"' # req sum of 4 for occrel
# mean_track_maps_3d [0.37 0.32 0.23 0.16 0.08 0.02 0.01]
# mean_track_maps_2d [0.52 0.49 0.44 0.38 0.3  0.2  0.11]
mod = '"zoom62"' # 12-12-12
# mean_track_maps_3d [0.39 0.28 0.19 0.11 0.05 0.01 0.  ]
# mean_track_maps_2d [0.46 0.44 0.4  0.32 0.23 0.13 0.05]
    
mod = '"box00"' # adjust the bike boxes; return early
mod = '"box01"' # 12-12-12 full run
# mean_track_maps_3d [0.47 0.36 0.26 0.16 0.06 0.02 0.01]
# mean_track_maps_2d [0.48 0.44 0.36 0.3  0.23 0.15 0.08]
mod = '"box02"' # 8-8-8
# mean_track_maps_3d [0.49 0.44 0.31 0.18 0.08 0.02 0.01]
# mean_track_maps_2d [0.54 0.52 0.46 0.38 0.31 0.21 0.12]
mod = '"box03"' # 8-4-8; fix a bug with which vox util to use; also use zoom shapes
# mean_track_maps_3d [0.45 0.3  0.06 0.02 0.   0.   0.  ]
# mean_track_maps_2d [0.52 0.49 0.43 0.36 0.28 0.17 0.09]
# not as good
mod = '"box04"' # 8-8-8 with the bug fixed
# mean_track_maps_3d [0.47 0.4  0.31 0.21 0.06 0.02 0.  ]
# mean_track_maps_2d [0.49 0.44 0.38 0.35 0.26 0.16 0.07]
mod = '"box05"' # ly = ly + 1.0
# mean_proposal_maps_3d [0.65 0.62 0.54 0.41 0.24 0.11 0.03]
# mean_proposal_maps_2d [0.66 0.65 0.61 0.54 0.42 0.27 0.15]
# mean_track_maps_3d [0.5  0.42 0.33 0.23 0.11 0.03 0.01]
# mean_track_maps_2d [0.52 0.47 0.39 0.33 0.24 0.15 0.07]
# ok i think this guy wins, despite the bev maps being a little lower than i expected

mod = '"box06"' # same but the stationary part of the data
# mean_proposal_maps_3d [0.48 0.45 0.41 0.33 0.24 0.11 0.02]
# mean_proposal_maps_2d [0.49 0.48 0.44 0.4  0.35 0.27 0.19]
# mean_track_maps_3d [0.46 0.43 0.4  0.35 0.23 0.09 0.02]
# mean_track_maps_2d [0.47 0.46 0.43 0.41 0.36 0.27 0.16]
# shockingly high. also the proposal maps are a bit low. what happened here?


mod = '"box07"' # flow method
mod = '"box08"' # reject huge boxes
# mean_proposal_maps_3d [0.04 0.02 0.01 0.   0.   0.   0.  ]
# mean_proposal_maps_2d [0.07 0.05 0.03 0.02 0.01 0.   0.  ]
mod = '"box09"' # same but on MOVING-cam data
# mean_proposal_maps_3d [0.14 0.08 0.03 0.01 0.01 0.   0.  ]
# mean_proposal_maps_2d [0.19 0.14 0.1  0.06 0.03 0.02 0.  ]



mod = '"box10"' # back to box06 (only reject 1vox boxes); also evaluate perspective box proposals; moving cam
# mean_proposal_maps_3d [0.66 0.63 0.55 0.43 0.28 0.14 0.04]
# mean_proposal_maps_2d [0.68 0.66 0.63 0.55 0.44 0.32 0.17]
# mean_proposal_maps_pers [0.69 0.68 0.65 0.61 0.54 0.42 0.24]
# mean_track_maps_3d [0.48 0.42 0.35 0.23 0.14 0.04 0.01]
# mean_track_maps_2d [0.49 0.45 0.4  0.37 0.28 0.2  0.11]
# this did not quite get to 100 iters: it crashed at 77
mod = '"box11"' # masked mean in eval!!
mod = '"box12"' # perspective track eval too
# mean_proposal_maps_3d [0.66 0.63 0.55 0.43 0.28 0.14 0.04]
# mean_proposal_maps_2d [0.68 0.66 0.63 0.55 0.44 0.32 0.17]
# mean_proposal_maps_pers [0.69 0.68 0.65 0.61 0.54 0.42 0.24]
# mean_track_maps_3d [0.46 0.4  0.36 0.23 0.09 0.02 0.  ]
# mean_track_maps_2d [0.47 0.43 0.39 0.36 0.27 0.17 0.07]
# mean_track_maps_pers [0.53 0.51 0.48 0.43 0.36 0.21 0.08]

mod = '"box13"' # flow again; do discard huge boxes; moving cam
# mean_proposal_maps_3d [0.14 0.08 0.03 0.01 0.01 0.   0.  ]
# mean_proposal_maps_2d [0.19 0.14 0.1  0.06 0.03 0.02 0.  ]
# mean_proposal_maps_pers [0.22 0.16 0.11 0.07 0.03 0.02 0.01]

mod = '"box14"' # flow again; do discard huge boxes; stationary cam


mod = '"box15"' # stationary cam; not flow
# mean_proposal_maps_3d [0.49 0.46 0.41 0.34 0.25 0.12 0.02]
# mean_proposal_maps_2d [0.5  0.49 0.45 0.4  0.36 0.28 0.2 ]
# mean_proposal_maps_pers [0.51 0.5  0.49 0.47 0.43 0.35 0.23]
# mean_track_maps_3d [0.39 0.37 0.33 0.25 0.18 0.08 0.01]
# mean_track_maps_2d [0.39 0.38 0.37 0.33 0.29 0.23 0.16]
# mean_track_maps_pers [0.43 0.42 0.41 0.39 0.37 0.32 0.2 ]
# stopped at iter 94

# re-run (as fre00), also stopped at 94
# mean_proposal_maps_3d [0.49 0.46 0.41 0.34 0.25 0.12 0.02]
# mean_proposal_maps_2d [0.5  0.49 0.45 0.4  0.36 0.28 0.2 ]
# mean_proposal_maps_pers [0.51 0.5  0.49 0.47 0.43 0.35 0.23]
# mean_track_maps_3d [0.42 0.41 0.38 0.3  0.22 0.11 0.03]
# mean_track_maps_2d [0.42 0.42 0.41 0.38 0.33 0.25 0.16]
# mean_track_maps_pers [0.45 0.44 0.43 0.42 0.4  0.38 0.25]

mod = '"box16"' # redo that with eval bug fixed; stationary cam
mod = '"box17"' # redo that with eval bug fixed; moving cam
# mean_proposal_maps_3d [0.65 0.62 0.54 0.41 0.24 0.11 0.03]
# mean_proposal_maps_2d [0.66 0.65 0.61 0.54 0.42 0.27 0.15]
# mean_proposal_maps_pers [0.68 0.67 0.64 0.61 0.53 0.41 0.21]
# mean_track_maps_3d [0.44 0.38 0.3  0.18 0.07 0.02 0.  ]
# mean_track_maps_2d [0.46 0.41 0.35 0.31 0.24 0.14 0.09]
# mean_track_maps_pers [0.5  0.48 0.44 0.4  0.32 0.21 0.08]



mod = '"box18"' # flow proposals; discard huge boxes; moving cam
# mean_proposal_maps_3d [0.14 0.08 0.03 0.01 0.01 0.   0.  ]
# mean_proposal_maps_2d [0.19 0.14 0.1  0.06 0.03 0.02 0.  ]
# mean_proposal_maps_pers [0.22 0.16 0.11 0.07 0.03 0.02 0.01]

mod = '"box19"' # flow proposals; discard huge boxes; stationary camera
# mean_proposal_maps_3d [0.04 0.02 0.01 0.   0.   0.   0.  ]
# mean_proposal_maps_2d [0.07 0.05 0.03 0.02 0.01 0.   0.  ]
# mean_proposal_maps_pers [0.08 0.05 0.03 0.02 0.01 0.   0.  ]




mod = '"box20"' # boost with occ_memXAI_g and free


mod = '"box21"' # return after proposing
# mean_proposal_maps_3d [0.48 0.45 0.41 0.33 0.24 0.11 0.02]
# mean_proposal_maps_2d [0.49 0.48 0.44 0.4  0.35 0.27 0.19]
# mean_proposal_maps_pers [0.5  0.49 0.48 0.46 0.42 0.34 0.22]

mod = '"box22"' # return after proposing; moving camera
# mean_proposal_maps_3d [0.65 0.62 0.54 0.41 0.24 0.11 0.03]
# mean_proposal_maps_2d [0.66 0.65 0.61 0.54 0.42 0.27 0.15]
# mean_proposal_maps_pers [0.68 0.67 0.64 0.61 0.53 0.41 0.21]


mod = '"box23"' # return after proposing; moving camera; only boost with occ
# mean_proposal_maps_3d [0.65 0.62 0.54 0.41 0.24 0.11 0.03]
# mean_proposal_maps_2d [0.66 0.65 0.61 0.54 0.42 0.27 0.15]
# mean_proposal_maps_pers [0.68 0.67 0.64 0.61 0.53 0.41 0.21]
# this is identical


# boost was still turned on. but i know it's no big deal
mod = '"box24"' # moving; single scale
# mean_proposal_maps_3d [0.65 0.62 0.54 0.41 0.24 0.11 0.03]
# mean_proposal_maps_2d [0.66 0.65 0.61 0.54 0.42 0.27 0.15]
# mean_proposal_maps_pers [0.68 0.67 0.64 0.61 0.53 0.41 0.21]
# mean_track_maps_3d [0.44 0.4  0.32 0.21 0.12 0.05 0.01]
# mean_track_maps_2d [0.45 0.43 0.36 0.31 0.22 0.13 0.08]
# mean_track_maps_pers [0.49 0.48 0.45 0.42 0.34 0.23 0.14]

mod = '"box25"' # static; single scale
# mean_track_maps_3d [0.44 0.42 0.35 0.29 0.18 0.05 0.  ]
# mean_track_maps_2d [0.45 0.44 0.41 0.35 0.3  0.23 0.11]
# mean_track_maps_pers [0.47 0.46 0.45 0.43 0.4  0.34 0.19]


mod = '"box26"' # again but occrel = ones; moving
# mean_proposal_maps_3d [0.67 0.63 0.54 0.39 0.21 0.09 0.02]
# mean_proposal_maps_2d [0.67 0.65 0.6  0.5  0.37 0.23 0.13]
# mean_proposal_maps_pers [0.68 0.67 0.65 0.62 0.53 0.39 0.21]
# mean_track_maps_3d [0.34 0.25 0.15 0.09 0.04 0.01 0.  ]
# mean_track_maps_2d [0.37 0.3  0.23 0.15 0.07 0.03 0.02]
# mean_track_maps_pers [0.54 0.44 0.34 0.28 0.22 0.17 0.09]


mod = '"box27"' # just proposal; moving
# mean_proposal_maps_3d [0.67 0.63 0.54 0.39 0.21 0.09 0.02]
# mean_proposal_maps_2d [0.67 0.65 0.6  0.5  0.37 0.23 0.13]
# mean_proposal_maps_pers [0.68 0.67 0.65 0.62 0.53 0.39 0.21]

mod = '"box28"' # just proposal; stationary
# mean_proposal_maps_3d [0.48 0.46 0.42 0.35 0.25 0.12 0.03]
# mean_proposal_maps_2d [0.48 0.48 0.46 0.41 0.36 0.29 0.21]
# mean_proposal_maps_pers [0.49 0.48 0.47 0.46 0.42 0.34 0.23]

mod = '"box29"' # full tracking; stationary
# mean_proposal_maps_3d [0.48 0.46 0.42 0.35 0.25 0.12 0.03]
# mean_proposal_maps_2d [0.48 0.48 0.46 0.41 0.36 0.29 0.21]
# mean_proposal_maps_pers [0.49 0.48 0.47 0.46 0.42 0.34 0.23]
# mean_track_maps_3d [0.43 0.32 0.24 0.13 0.02 0.01 0.  ]
# mean_track_maps_2d [0.44 0.4  0.33 0.26 0.13 0.03 0.01]
# mean_track_maps_pers [0.45 0.44 0.42 0.38 0.34 0.22 0.09]


mod = '"box30"' # just proposal; stationary; 10% of points; do not return based on inb
# faulty
mod = '"box31"' # just proposal; stationary; 50% of points; do not return based on inb
# faulty

mod = '"box32"' # again, but no cache; 50% of points
# mean_proposal_maps_3d [0.43 0.39 0.32 0.25 0.17 0.08 0.02]
# mean_proposal_maps_2d [0.43 0.42 0.38 0.31 0.25 0.19 0.12]
# mean_proposal_maps_pers [0.44 0.43 0.41 0.37 0.34 0.27 0.16]

mod = '"box33"' # 10% of points
# mean_proposal_maps_3d [0.38 0.33 0.27 0.18 0.09 0.02 0.  ]
# mean_proposal_maps_2d [0.38 0.36 0.31 0.25 0.17 0.1  0.03]
# mean_proposal_maps_pers [0.4  0.39 0.37 0.33 0.29 0.2  0.11]



mod = '"box34"' # 10% of points and moving camera

mod = '"box35"' # 50% of points and moving camera
# mean_proposal_maps_3d [0.56 0.52 0.45 0.32 0.2  0.08 0.02]
# mean_proposal_maps_2d [0.57 0.54 0.5  0.43 0.32 0.22 0.12]
# mean_proposal_maps_pers [0.59 0.57 0.55 0.51 0.45 0.35 0.21]


# mod = '"box36"' # 100% of points; get feat dif


mod = '"re00"' # redo the last thing
# mod = '"re01"' # dense; 
# mod = '"re02"' # pret 02_s4_1e-4_P_c1_s.01_mags7i3t_occ09 



mod = '"re03"' # pret 43 again, but SIZE=16 < moving camera!
# mean_proposal_maps_3d [0.54 0.51 0.44 0.33 0.2  0.08 0.01]
# mean_proposal_maps_2d [0.54 0.53 0.5  0.43 0.34 0.25 0.14]
# mean_proposal_maps_pers [0.56 0.55 0.54 0.51 0.45 0.35 0.18]
mod = '"re04"' # back to size=20; do NOT mult by occ*vis, to see if the diff visualization gets clenaer and more complete
# mean_proposal_maps_3d [0.42 0.33 0.24 0.13 0.05 0.01 0.  ]
# mean_proposal_maps_2d [0.44 0.36 0.28 0.19 0.11 0.04 0.02]
# mean_proposal_maps_pers [0.46 0.42 0.37 0.31 0.24 0.16 0.07]
mod = '"re05"' # do mult; stationary cam; see if i replicate an old result
# mean_proposal_maps_3d [0.46 0.44 0.39 0.31 0.21 0.1  0.02]
# mean_proposal_maps_2d [0.47 0.45 0.43 0.37 0.31 0.24 0.17]
# mean_proposal_maps_pers [0.47 0.47 0.45 0.43 0.39 0.32 0.2 ]
# something is slightly off
mod = '"re06"' # return early based on inb, to see if we replicate more cleanly
# mean_proposal_maps_3d [0.48 0.46 0.42 0.35 0.25 0.12 0.03]
# mean_proposal_maps_2d [0.48 0.48 0.46 0.41 0.36 0.29 0.21]
# mean_proposal_maps_pers [0.49 0.48 0.47 0.46 0.42 0.34 0.23]
# ok this is repli
mod = '"re07"' # use_occrel = True
# mean_proposal_maps_3d [0.48 0.45 0.41 0.33 0.24 0.11 0.02]
# mean_proposal_maps_2d [0.49 0.48 0.44 0.4  0.35 0.27 0.19]
# mean_proposal_maps_pers [0.5  0.49 0.48 0.46 0.42 0.34 0.22]
# so, slightly worse. good thing i deleted it from teh paper
mod = '"re08"' # show what occrel looks like, in summ_oneds
# (same quant results as 07)
mod = '"re09"' # use_occrel = False; use featdiff
# mean_proposal_maps_3d [0.38 0.35 0.32 0.27 0.18 0.08 0.02]
# mean_proposal_maps_2d [0.4  0.38 0.35 0.31 0.28 0.22 0.14]
# mean_proposal_maps_pers [0.42 0.4  0.38 0.36 0.33 0.25 0.17]
mod = '"re10"' # SIZE = 16; req only 100 inb
# mean_proposal_maps_3d [0.39 0.34 0.26 0.17 0.1  0.03 0.  ]
# mean_proposal_maps_2d [0.39 0.38 0.33 0.26 0.19 0.13 0.06]
# mean_proposal_maps_pers [0.41 0.39 0.36 0.3  0.25 0.18 0.09]
mod = '"re11"' # SIZE = 12
# mean_proposal_maps_3d [0.31 0.28 0.21 0.12 0.05 0.01 0.  ]
# mean_proposal_maps_2d [0.33 0.31 0.26 0.2  0.14 0.09 0.03]
# mean_proposal_maps_pers [0.34 0.33 0.3  0.24 0.16 0.1  0.06]
mod = '"re12"' # back to inb 1k, but use 80x80x80 for that; SIZE = 22
mod = '"re13"' # size=24
# ah wait, these have been using featdiff since re09


mod = '"re14"' # size=16; use occdiff
# mean_proposal_maps_3d [0.45 0.42 0.38 0.29 0.19 0.08 0.03]
# mean_proposal_maps_2d [0.46 0.45 0.41 0.37 0.3  0.24 0.14]
# mean_proposal_maps_pers [0.46 0.45 0.43 0.4  0.35 0.28 0.13]
mod = '"re15"' # size=12; use occdiff
# mean_proposal_maps_3d [0.32 0.3  0.22 0.15 0.08 0.03 0.  ]
# mean_proposal_maps_2d [0.33 0.31 0.28 0.21 0.16 0.1  0.05]
# mean_proposal_maps_pers [0.33 0.33 0.31 0.25 0.19 0.11 0.05]
mod = '"re16"' # size=24; use occdiff
# mean_proposal_maps_3d [0.54 0.52 0.47 0.4  0.29 0.15 0.05]
# mean_proposal_maps_2d [0.54 0.53 0.51 0.45 0.39 0.31 0.21]
# mean_proposal_maps_pers [0.55 0.54 0.53 0.51 0.47 0.41 0.27]



mod = '"re17"' # size=24; use occdiff; S=100; track, 5 iters
mod = '"re18"' # use occrel
mod = '"re19"' # req camera motion
mod = '"re20"' # S_test = 100
mod = '"re21"' # max_along_y=True
mod = '"re22"' # eval_track_maps=True; S_test = 20 for speed
# mean_track_maps_3d [0.41 0.29 0.17 0.11 0.   0.   0.  ]
# mean_track_maps_2d [0.44 0.44 0.24 0.11 0.02 0.   0.  ]
# mean_track_maps_pers [0.48 0.44 0.33 0.31 0.19 0.07 0.05]

# the new tracking eval i want is:
# for each tracklet,
# evaluate it against the (single) gt tracklet available
# report the best of these
mod = '"re23"' # measure those ious
mod = '"re24"' # 2 super iters
mod = '"re25"' # 4 super iters; S_test = 50; 100 iters; stationary cam
mod = '"re26"' # fix small bug
mod = '"re27"' # give super_iter suffix to traj vis
mod = '"re28"' # one more bug; super_iters=1 to debug faster
mod = '"re29"' # super_iters = 4
mod = '"re30"' # use_occrel = False
mod = '"re31"' # compute sN and use it, since we do not always arrive at 4 objs
mod = '"re32"' # weak feats, as basline
mod = '"re33"' # strong feats; show occg
mod = '"re34"' # strong feats; show occg; moving cam


############## define experiment ##############

exps['builder'] = [
    'carla_explain', # mode
    # 'carla_multiview_train_data', # dataset
    # 'carla_multiview_test_data', # dataset
    # 'carla_tatt_trainset_data', # dataset
    'carla_tatv_testset_data', # dataset
    # 'carla_16-16-16_bounds_train',
    'carla_16-16-16_bounds_test',
    # 'carla_32-32-32_bounds_test',
    # '10k_iters',
    '1k_iters',
    # '100_iters',
    # '20_iters',
    # '10_iters',
    # '5_iters',
    'use_cache', 
    'lr4',
    'B1',
    'no_shuf',
    'pretrained_feat3D', 
    'pretrained_occ', 
    'pretrained_occrel', 
    'train_feat3D',
    'train_occ',
    'train_occrel',
    'train_sub',
    'no_backprop',
    # 'do_test', 
    'log1',
    # 'log10',
    # 'log50',
    # 'log5',
]
exps['trainer'] = [
    'carla_explain', # mode
    # 'carla_tat1_trainset_data', # dataset
    # 'carla_tat10_trainset_data', # dataset
    'carla_tat100_trainset_data', # dataset
    'carla_16-16-16_bounds_train',
    # 'carla_16-16-16_bounds_val',
    '100k_iters',
    'lr3',
    'B1',
    'use_cache',
    'pretrained_feat3D', 
    'pretrained_occ',
    'pretrained_occrel',
    'frozen_feat3D', 
    'frozen_occ', 

    'frozen_occrel', 
    'train_feat3D',
    'train_occ',
    'train_occrel',
    'train_sub',
    'log10',
]
exps['tester'] = [
    'carla_explain', # mode
    'carla_tatv_testset_data', # dataset
    'carla_16-16-16_bounds_train',
    'carla_16-16-16_bounds_test',
    # 'carla_16-16-16_bounds_zoom',
    # 'carla_12-12-12_bounds_zoom',
    'carla_8-8-8_bounds_zoom',
    # 'carla_8-4-8_bounds_zoom',
    # '100_iters',
    # '5_iters',
    # '100_iters',
    '20_iters',
    # '20_iters',
    # '15_iters',
    'lr4',
    'B1',
    # 'use_cache',
    'no_shuf',
    'pretrained_feat3D', 
    'pretrained_occ', 
    'pretrained_occrel', 
    'train_feat3D',
    'train_occ',
    'train_occrel',
    # 'train_flow', 
    'no_backprop',
    'do_test', 
    'log1',
]
# exps['tester'] = [
#     'carla_explain', # mode
#     # 'carla_tast_trainset_data', # dataset
#     'carla_tatv_testset_data', # dataset
#     'carla_16-8-16_bounds_train',
#     'carla_16-8-16_bounds_test',
#     '25_iters',
#     # '10k_iters',
#     # '100k_iters',
#     # 'do_test', 
#     'B1',
#     'pretrained_feat3D', 
#     'pretrained_occ',
#     'pretrained_mot',
#     'frozen_feat3D', 
#     'frozen_occ', 
#     'frozen_mot', 
#     'train_feat3D',
#     'train_occ',
#     'train_mot',
#     'log1',
# ]
exps['render_trainer'] = [
    'carla_explain', # mode
    'carla_multiview_train_data', # dataset
    'carla_multiview_ep09_data', # dataset
    # 'carla_multiview_ep09one_data', # dataset
    # 'carla_multiview_one_data', # dataset
    # 'carla_wide_nearcube_bounds',
    # 'carla_nearcube_bounds',
    'carla_narrow_nearcube_bounds',
    # 'carla_narrow_flat_bounds',
    # '5k_iters',
    '500k_iters',
    'lr3',
    'B1',
    'pretrained_latents',
    # 'train_vq3d',
    # 'train_up3D',
    'train_occ',
    'train_render',
    # 'no_shuf',
    'snap50',
    'log50',
    # 'log50',
]
exps['center_trainer'] = [
    'carla_explain', # mode
    'carla_multiview_train_data', # dataset
    'carla_wide_cube_bounds',
    '100k_iters',
    'lr3',
    'B2',
    'pretrained_feat3D', 
    'pretrained_occ', 
    'pretrained_center', 
    'train_feat3D',
    'train_occ',
    'train_center',
    'log50',
]
exps['seg_trainer'] = [
    'carla_explain', # mode
    'carla_multiview_all_data', # dataset
    'carla_wide_cube_bounds',
    '200k_iters',
    'lr4',
    'B2',
    'pretrained_feat3D', 
    'pretrained_up3D', 
    'pretrained_occ', 
    'pretrained_center', 
    'pretrained_seg', 
    'train_feat3D',
    'train_up3D',
    'train_occ',
    'train_center',
    'train_seg',
    'snap5k',
    'log500',
]
exps['vq_trainer'] = [
    'carla_explain', # mode
    'carla_multiview_all_data', # dataset
    'carla_wide_cube_bounds',
    '200k_iters',
    'lr4',
    'B2',
    'pretrained_feat3D', 
    'pretrained_vq3d', 
    'pretrained_up3D',
    'pretrained_occ', 
    'pretrained_center', 
    'pretrained_seg', 
    'train_feat3D',
    'train_up3D',
    'train_occ',
    'train_center',
    'train_seg',
    'train_vq3d',
    # # 'frozen_feat3D',
    # 'frozen_up3D',
    # # 'frozen_vq3d',
    # 'frozen_occ',
    # 'frozen_center',
    # 'frozen_seg',
    'snap5k',
    'log500',
]
exps['vq_exporter'] = [
    'carla_explain', # mode
    'carla_multiview_all_data_as_test', # dataset
    'carla_wide_cube_bounds',
    '5k_iters', # iter more than necessary, since we have some augs
    # '100_iters', 
    'no_shuf',
    'do_test', 
    'do_export_inds', 
    'lr4',
    'B1',
    'pretrained_feat3D', 
    'pretrained_up3D', 
    'pretrained_vq3d', 
    'pretrained_occ', 
    'pretrained_center', 
    'pretrained_seg', 
    'frozen_feat3D',
    'frozen_up3D',
    'frozen_vq3d',
    'frozen_occ',
    'frozen_center',
    'frozen_seg',
    'log50',
]

############## net configs ##############

groups['do_test'] = ['do_test = True']
groups['do_export_inds'] = ['do_export_inds = True']
groups['use_cache'] = ['do_use_cache = True']
groups['carla_explain'] = ['do_carla_explain = True']

groups['train_feat3D'] = [
    'do_feat3D = True',
    'feat3D_dim = 32',
    # 'feat3D_skip = True',
]
groups['train_flow'] = [
    'do_flow = True',
    # 'flow_l1_coeff = 1.0',
    'flow_l2_coeff = 1.0',
    'flow_heatmap_size = 7',
]
groups['train_up3D'] = [
    'do_up3D = True',
    # 'up3D_smooth_coeff = 0.01',
]
groups['train_emb3D'] = [
    'do_emb3D = True',
    'emb_3D_ce_coeff = 1.0',
    # 'emb_3D_l2_coeff = 0.1',
    # 'emb_3D_mindist = 16.0',
    'emb_3D_num_samples = 2',
]
groups['train_vq3d'] = [
    'do_vq3d = True',
    'vq3d_latent_coeff = 1.0',
    'vq3d_num_embeddings = 512', 
]
groups['train_view'] = [
    'do_view = True',
    'view_depth = 64',
    'view_l1_coeff = 1.0',
    # 'view_smooth_coeff = 1.0',
]
groups['train_render'] = [
    'do_render = True',
    'render_depth = 64',
    'render_l2_coeff = 10.0',
]
groups['train_occ'] = [
    'do_occ = True',
    'occ_coeff = 1.0',
    # 'occ_smooth_coeff = 0.1',
]
groups['train_occrel'] = [
    'do_occrel = True',
    'occrel_coeff = 1.0',
]
groups['train_sub'] = [
    'do_sub = True',
    'sub_coeff = 1.0',
    'sub_smooth_coeff = 2.0',
]
groups['train_center'] = [
    'do_center = True',
    'center_prob_coeff = 1.0',
    'center_size_coeff = 0.1', # this loss tends to be large
    'center_rot_coeff = 1.0',
]
groups['train_seg'] = [
    'do_seg = True',
    'seg_prob_coeff = 1.0',
    'seg_smooth_coeff = 0.001',
]
groups['train_mot'] = [
    'do_mot = True',
    'mot_prob_coeff = 1.0',
    'mot_smooth_coeff = 0.01',
]
groups['train_linclass'] = [
    'do_linclass = True',
    'linclass_coeff = 1.0',
]


############## datasets ##############

# dims for mem
# SIZE = 20
# Z = int(SIZE*16)
# Y = int(SIZE*16)
# X = int(SIZE*16)
# SIZE = 20
# Z = 180
# Y = 60
# X = 180
# Z_test = 180
# Y_test = 60
# X_test = 180

# SIZE = 16
# SIZE_test = 16

# SIZE = 20
# SIZE_val = 20
# SIZE_test = 20
# SIZE_zoom = 20

# SIZE = 20
# SIZE_val = 20
# SIZE_test = 20
# SIZE_zoom = 20

# SIZE = 22
# SIZE_val = 22
# SIZE_test = 22
# SIZE_zoom = 22

SIZE = 24
SIZE_val = 24
SIZE_test = 24
SIZE_zoom = 24

# SIZE = 16
# SIZE_val = 16
# SIZE_test = 16
# SIZE_zoom = 16

# SIZE = 12
# SIZE_val = 12
# SIZE_test = 12
# SIZE_zoom = 12

# Z = 160
# Y = 80
# X = 160
# Z_test = 160
# Y_test = 80
# X_test = 160
# # Z = 128
# Y = 64
# X = 128
# Z_test = 128
# Y_test = 64
# X_test = 128

K = 2 # how many objects to consider
N = 16 # how many objects per npz
S = 100
S_val = 2
S_test = 50
H = 128
W = 384
# H and W for proj stuff
# PH = int(H/2.0)
# PW = int(W/2.0)
PH = int(H)
PW = int(W)

# dataset_location = "/scratch"
dataset_location = "/projects/katefgroup/datasets/carla/processed/npzs"
# dataset_location = "/data/carla/processed/npzs"
# dataset_location = "/data4/carla/processed/npzs"

groups['carla_multiview_train1_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mags7i3one"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_multiview_all_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mads7i3a"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_multiview_ep09_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mags7i3ep09"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_multiview_ep09one_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mags7i3ep09one"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_multiview_one_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mags7i3one"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_multiview_all_data_as_test'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'testset = "mads7i3a"',
    'testset_format = "multiview"', 
    'testset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_multiview_train10_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mabs7i3ten"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_multiview_train100_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mabs7i3hun"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_multiview_train_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mags7i3t"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_multiview_val_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'valset = "mags7i3v"',
    'valset_format = "multiview"', 
    'valset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_multiview_test_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'testset = "mags7i3v"',
    'testset_format = "multiview"', 
    'testset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_multiview_train10_val10_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mabs7i3ten"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'valset = "mabs7i3ten"',
    'valset_format = "multiview"', 
    'valset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_multiview_train_val_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mabs7i3t"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'valset = "mabs7i3v"',
    'valset_format = "multiview"', 
    'valset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_multiview_train_val_test_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mabs7i3t"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'valset = "mabs7i3v"',
    'valset_format = "multiview"', 
    'valset_seqlen = %d' % S, 
    'testset = "mabs7i3v"',
    'testset_format = "multiview"', 
    'testset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_multiview_train_test_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "mags7i3t"',
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'testset = "taqs100i2v"',
    'testset_format = "traj"', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_taqv_testset_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'testset = "taqs100i2v"',
    'testset_format = "traj"', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_tasa_testset_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'testset = "tass100i2a"',
    'testset_format = "traj"', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_tat1_trainset_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "tats100i2one"',
    'trainset_format = "traj"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_tatt_trainset_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "tats100i2t"',
    'trainset_format = "traj"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_tat10_trainset_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "tats100i2ten"',
    'trainset_format = "traj"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_tat100_trainset_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "tats100i2hun"',
    'trainset_format = "traj"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_tatv_testset_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'testset = "tats100i2v"',
    'testset_format = "traj"', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_tatv_testset_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'testset = "tats100i2v"',
    'testset_format = "traj"', 
    'testset_seqlen = %d' % S_test, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_regular_bounds'] = [
    'XMIN = -16.0', # right (neg is left)
    'XMAX = 16.0', # right
    'YMIN = -4.0', # down (neg is up)
    'YMAX = 4.0', # down
    'ZMIN = -16.0', # forward
    'ZMAX = 16.0', # forward
]
groups['carla_16-8-16_bounds_train'] = [
    'XMIN = -16.0', # right (neg is left)
    'XMAX = 16.0', # right
    'YMIN = -8.0', # down (neg is up)
    'YMAX = 8.0', # down
    'ZMIN = -16.0', # forward
    'ZMAX = 16.0', # forward
    'Z = %d' % (int(SIZE*8)),
    'Y = %d' % (int(SIZE*4)),
    'X = %d' % (int(SIZE*8)),
]
groups['carla_16-16-16_bounds_train'] = [
    'XMIN = -16.0', # right (neg is left)
    'XMAX = 16.0', # right
    'YMIN = -16.0', # down (neg is up)
    'YMAX = 16.0', # down
    'ZMIN = -16.0', # forward
    'ZMAX = 16.0', # forward
    'Z = %d' % (int(SIZE*8)),
    'Y = %d' % (int(SIZE*8)),
    'X = %d' % (int(SIZE*8)),
]
groups['carla_16-16-16_bounds_test'] = [
    'XMIN_test = -16.0', # right (neg is left)
    'XMAX_test = 16.0', # right
    'YMIN_test = -16.0', # down (neg is up)
    'YMAX_test = 16.0', # down
    'ZMIN_test = -16.0', # forward
    'ZMAX_test = 16.0', # forward
    'Z_test = %d' % (int(SIZE_test*8)),
    'Y_test = %d' % (int(SIZE_test*8)),
    'X_test = %d' % (int(SIZE_test*8)),
]
groups['carla_8-8-8_bounds_zoom'] = [
    'XMIN_zoom = -8.0', # right (neg is left)
    'XMAX_zoom = 8.0', # right
    'YMIN_zoom = -8.0', # down (neg is up)
    'YMAX_zoom = 8.0', # down
    'ZMIN_zoom = -8.0', # forward
    'ZMAX_zoom = 8.0', # forward
    'Z_zoom = %d' % (int(SIZE_zoom*8)),
    'Y_zoom = %d' % (int(SIZE_zoom*8)),
    'X_zoom = %d' % (int(SIZE_zoom*8)),
]
groups['carla_16-16-16_bounds_zoom'] = [
    'XMIN_zoom = -16.0', # right (neg is left)
    'XMAX_zoom = 16.0', # right
    'YMIN_zoom = -16.0', # down (neg is up)
    'YMAX_zoom = 16.0', # down
    'ZMIN_zoom = -16.0', # forward
    'ZMAX_zoom = 16.0', # forward
    'Z_zoom = %d' % (int(SIZE_zoom*8)),
    'Y_zoom = %d' % (int(SIZE_zoom*8)),
    'X_zoom = %d' % (int(SIZE_zoom*8)),
]
groups['carla_12-12-12_bounds_zoom'] = [
    'XMIN_zoom = -12.0', # right (neg is left)
    'XMAX_zoom = 12.0', # right
    'YMIN_zoom = -12.0', # down (neg is up)
    'YMAX_zoom = 12.0', # down
    'ZMIN_zoom = -12.0', # forward
    'ZMAX_zoom = 12.0', # forward
    'Z_zoom = %d' % (int(SIZE_zoom*8)),
    'Y_zoom = %d' % (int(SIZE_zoom*8)),
    'X_zoom = %d' % (int(SIZE_zoom*8)),
]
groups['carla_8-4-8_bounds_zoom'] = [
    'XMIN_zoom = -8.0', # right (neg is left)
    'XMAX_zoom = 8.0', # right
    'YMIN_zoom = -4.0', # down (neg is up)
    'YMAX_zoom = 4.0', # down
    'ZMIN_zoom = -8.0', # forward
    'ZMAX_zoom = 8.0', # forward
    'Z_zoom = %d' % (int(SIZE_zoom*8)),
    'Y_zoom = %d' % (int(SIZE_zoom*4)),
    'X_zoom = %d' % (int(SIZE_zoom*8)),
]
groups['carla_32-32-32_bounds_test'] = [
    'XMIN_test = -32.0', # right (neg is left)
    'XMAX_test = 32.0', # right
    'YMIN_test = -32.0', # down (neg is up)
    'YMAX_test = 32.0', # down
    'ZMIN_test = -32.0', # forward
    'ZMAX_test = 32.0', # forward
    'Z_test = %d' % (int(SIZE_test*8)),
    'Y_test = %d' % (int(SIZE_test*8)),
    'X_test = %d' % (int(SIZE_test*8)),
]
groups['carla_16-8-16_bounds_val'] = [
    'XMIN_val = -16.0', # right (neg is left)
    'XMAX_val = 16.0', # right
    'YMIN_val = -8.0', # down (neg is up)
    'YMAX_val = 8.0', # down
    'ZMIN_val = -16.0', # forward
    'ZMAX_val = 16.0', # forward
    'Z_val = %d' % (int(SIZE_val*8)),
    'Y_val = %d' % (int(SIZE_val*4)),
    'X_val = %d' % (int(SIZE_val*8)),
]
groups['carla_16-8-16_bounds_test'] = [
    'XMIN_test = -16.0', # right (neg is left)
    'XMAX_test = 16.0', # right
    'YMIN_test = -8.0', # down (neg is up)
    'YMAX_test = 8.0', # down
    'ZMIN_test = -16.0', # forward
    'ZMAX_test = 16.0', # forward
    'Z_test = %d' % (int(SIZE_test*8)),
    'Y_test = %d' % (int(SIZE_test*4)),
    'X_test = %d' % (int(SIZE_test*8)),
]
groups['carla_8-4-8_bounds_test'] = [
    'XMIN_test = -8.0', # right (neg is left)
    'XMAX_test = 8.0', # right
    'YMIN_test = -4.0', # down (neg is up)
    'YMAX_test = 4.0', # down
    'ZMIN_test = -8.0', # forward
    'ZMAX_test = 8.0', # forward
    # 'XMIN_test = -12.0', # right (neg is left)
    # 'XMAX_test = 12.0', # right
    # 'YMIN_test = -6.0', # down (neg is up)
    # 'YMAX_test = 6.0', # down
    # 'ZMIN_test = -12.0', # forward
    # 'ZMAX_test = 12.0', # forward
    'Z_test = %d' % (int(SIZE_test*8)),
    'Y_test = %d' % (int(SIZE_test*4)),
    'X_test = %d' % (int(SIZE_test*8)),
]
groups['carla_12-6-12_bounds_test'] = [
    'XMIN_test = -12.0', # right (neg is left)
    'XMAX_test = 12.0', # right
    'YMIN_test = -6.0', # down (neg is up)
    'YMAX_test = 6.0', # down
    'ZMIN_test = -12.0', # forward
    'ZMAX_test = 12.0', # forward
    'Z_test = %d' % (int(SIZE_test*8)),
    'Y_test = %d' % (int(SIZE_test*4)),
    'X_test = %d' % (int(SIZE_test*8)),
]
groups['carla_8-6-8_bounds_test'] = [
    # 'XMIN_test = -8.0', # right (neg is left)
    # 'XMAX_test = 8.0', # right
    # 'YMIN_test = -6.0', # down (neg is up)
    # 'YMAX_test = 6.0', # down
    # 'ZMIN_test = -8.0', # forward
    # 'ZMAX_test = 8.0', # forward
    'XMIN_test = -12.0', # right (neg is left)
    'XMAX_test = 12.0', # right
    'YMIN_test = -9.0', # down (neg is up)
    'YMAX_test = 9.0', # down
    'ZMIN_test = -12.0', # forward
    'ZMAX_test = 12.0', # forward
    'Z_test = %d' % (int(SIZE_test*8)),
    'Y_test = %d' % (int(SIZE_test*6)),
    'X_test = %d' % (int(SIZE_test*8)),
]
groups['carla_8-8-8_bounds_test'] = [
    'XMIN_test = -8.0', # right (neg is left)
    'XMAX_test = 8.0', # right
    'YMIN_test = -8.0', # down (neg is up)
    'YMAX_test = 8.0', # down
    'ZMIN_test = -8.0', # forward
    'ZMAX_test = 8.0', # forward
    'Z_test = %d' % (int(SIZE_test*8)),
    'Y_test = %d' % (int(SIZE_test*8)),
    'X_test = %d' % (int(SIZE_test*8)),
]

groups['carla_313_bounds'] = [
    'XMIN = -18.0', # right (neg is left)
    'XMAX = 18.0', # right
    'YMIN = -6.0', # down (neg is up)
    'YMAX = 6.0', # down
    'ZMIN = -18.0', # forward
    'ZMAX = 18.0', # forward
]
groups['carla_flat_bounds'] = [
    'XMIN = -16.0', # right (neg is left)
    'XMAX = 16.0', # right
    'YMIN = -4.0', # down (neg is up)
    'YMAX = 4.0', # down
    'ZMIN = -16.0', # forward
    'ZMAX = 16.0', # forward
]
groups['carla_narrow_nearcube_bounds'] = [
    'XMIN = -8.0', # right (neg is left)
    'XMAX = 8.0', # right
    'YMIN = -4.0', # down (neg is up)
    'YMAX = 4.0', # down
    'ZMIN = -8.0', # forward
    'ZMAX = 8.0', # forward
]
groups['carla_narrow_flat_bounds'] = [
    'XMIN = -8.0', # right (neg is left)
    'XMAX = 8.0', # right
    'YMIN = -2.0', # down (neg is up)
    'YMAX = 2.0', # down
    'ZMIN = -8.0', # forward
    'ZMAX = 8.0', # forward
]
groups['carla_cube_bounds'] = [
    'XMIN = -16.0', # right (neg is left)
    'XMAX = 16.0', # right
    'YMIN = -16.0', # down (neg is up)
    'YMAX = 16.0', # down
    'ZMIN = -16.0', # forward
    'ZMAX = 16.0', # forward
]
groups['carla_wide_nearcube_bounds'] = [
    'XMIN = -32.0', # right (neg is left)
    'XMAX = 32.0', # right
    'YMIN = -16.0', # down (neg is up)
    'YMAX = 16.0', # down
    'ZMIN = -32.0', # forward
    'ZMAX = 32.0', # forward
]
groups['carla_wide_cube_bounds'] = [
    'XMIN = -32.0', # right (neg is left)
    'XMAX = 32.0', # right
    'YMIN = -32.0', # down (neg is up)
    'YMAX = 32.0', # down
    'ZMIN = -32.0', # forward
    'ZMAX = 32.0', # forward
    'XMIN_test = -32.0', # right (neg is left)
    'XMAX_test = 32.0', # right
    'YMIN_test = -32.0', # down (neg is up)
    'YMAX_test = 32.0', # down
    'ZMIN_test = -32.0', # forward
    'ZMAX_test = 32.0', # forward
]

############## verify and execute ##############

def _verify_(s):
    varname, eq, val = s.split(' ')
    assert varname in globals()
    assert eq == '='
    assert type(s) is type('')

print(current)
assert current in exps
for group in exps[current]:
    print("  " + group)
    assert group in groups
    for s in groups[group]:
        print("    " + s)
        _verify_(s)
        exec(s)

s = "mod = " + mod
_verify_(s)

exec(s)

