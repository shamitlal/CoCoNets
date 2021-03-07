def collapse_past_and_refresh_future(hypotheses, k, s, forecaster, xyz_camR0s):
    # trajectory k wins
    # so up to step s, we need hypothesis k in every slot
    winning_traj = hypotheses[:,k,:s]
    hypotheses[:,:,:s] = winning_traj.unsqueeze(1)
    confidences[:,:,:s] = 1.0

    # now we need to refresh the future
    lrt_now = winning_traj[:,-1]
    latest_input = place_object_at_delta(lrt_now, xyz_camR0s[:,s])
    futures = forecaster(latest_input)
    hypotheses[:,:,s+1:min(len(futures),S)] = futures
    
    return hypotheses

hypotheses = zeros(B, K, S, 19)
confidences = zeros(B, K, S)
hypotheses = collapse_past_and_refresh_future()
for s in range(S):
    found = False
    for k in range(K):
        if not found:
	    hypothesis = hypotheses[:,k,s]
	    vis = get_visibility(hypothesis)
	    if vis:
    	        search_region = place_object_at_delta(hypothesis, xyz_camR0s[:,s])
    	        conf, rad, xyz = match(template, search_region)
	        lrt = update_lrt(rad, xyz, hypothesis)
	        if conf > thresh:
	      	    # we found it
	      	    hypotheses = collapse_past_and_refresh_future()
	      	    found = True
    # end loop over k

    # this means that if we do not find a match for any hypothesis,
    # then we just keep going with the current set, no stress

    # i will probably need to add alternate hypotheses,
    # and also reject current ones based on built-up evidence,
    # via the hypothesis tester
    
