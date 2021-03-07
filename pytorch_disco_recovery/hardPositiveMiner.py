import hyperparams as hyp
import numpy as np 
import torch 
import random
import torch.nn.functional as F
from itertools import product
from torch.autograd import Variable
# import multiprocess_flag 
import cross_corr


class HardPositiveMiner():
    def __init__(self):
        '''
        TODO: initialize all this using exp_nel_sta later after debugging
        '''
        self.device = torch.device("cuda")
        self.searchRegion = hyp.max.searchRegion
        self.queryScale = [20,26]
        self.shouldResizeToRandomScale = hyp.max.shouldResizeToRandomScale
        self.margin = hyp.max.margin
        self.imgFeatMin = self.searchRegion + 2 * self.margin + 1 ## Minimum dimension of feature map in a image
        self.validRegion = hyp.max.validRegion
        # For each tensor in pool_e, we will extract top 'retrievalsForEachQueryEmb'
        # best matches from rank matrix. Hard positives will then be selected from this
        # collection of tensors.
        self.numRetrievalsForEachQueryEmb = hyp.max.numRetrievalsForEachQueryEmb
        self.topK = hyp.max.topK
        self.trainRegion = hyp.max.trainRegion
        self.mbr = cross_corr.meshgrid_based_rotation(hyp.box_size_xz-2*self.margin, hyp.box_size_y-2*self.margin, hyp.box_size_xz-2*self.margin)
    # def addToPool(self, embeddings, images=None, classes=None):
    #     self.pool.update(embeddings.to(self.device), images, classes)

    def resizeEmbed(self, embed, scale=None):
        if scale is None:
            scale = np.random.choice(self.queryScale) ## Randomly choose a scale
        # print("Using scale %s" %(scale))
        S,C,D,H,W = embed.shape
        dimlist = [D,H,W]
        maxdim = max(dimlist)
        dimlist = sorted(dimlist)
        ratio1 = float(dimlist[1])/dimlist[0]
        ratio2 = float(dimlist[2])/dimlist[1]
        
        size2 = scale
        size1 = round(float(size2)/ratio2)
        size0 = round(float(size1)/ratio1)

        if D == maxdim:
            D = size2
            if H>=W:
                H = size1
                W = size0
            else:
                W = size1
                H = size0

        elif H == maxdim:
            H = size2
            if D>=W:
                D = size1
                W = size0
            else:
                W = size1
                D = size0
        else:
            W = size2
            if D>=H:
                D = size1
                H = size0
            else:
                H = size1
                D = size0

        D, H, W = max(D, self.imgFeatMin), max(H, self.imgFeatMin), max(W, self.imgFeatMin)
        resized_embed = F.interpolate(embed, size = (int(D), int(H), int(W)), mode='trilinear')
        return resized_embed

    '''
    Image feature for searching image :
    1. features in different scales;
    2. remove feature in the border
    '''
    def DataShuffle(self,sample, batchSize):
        nbSample = len(sample)
        iterEpoch = nbSample // batchSize
        permutationIndex = np.random.permutation(range(nbSample))
        sampleIndex = permutationIndex.reshape(( iterEpoch, batchSize)).astype(int)
        return sampleIndex

    def SearchImgFeat(self, embed):
        searchFeat = {}
        margin = self.margin
        if self.shouldResizeToRandomScale:
            for s in self.queryScale:
                emb_resized = self.resizeEmbed(embed, s)
                feat_d, feat_h, feat_w = emb_resized.shape[2], emb_resized.shape[3], emb_resized.shape[4]
                searchFeat[s] = emb_resized[:, :, margin : feat_d - margin, margin : feat_h - margin, margin : feat_w - margin].clone()
        else:
            feat_d, feat_h, feat_w = (embed.shape[2], embed.shape[3], embed.shape[4])
            searchFeat[16] = embed[:, :, margin : feat_d - margin, margin : feat_h - margin, margin : feat_w - margin].clone()
        return searchFeat

    '''
    Cosine similarity Implemented as a Convolutional Layer
    Note: we don't normalize kernel
    '''
    def CosineSimilarity(self, emb_feat, kernel, kernel_one) :
        # st()
        emb_feat = emb_feat.float()
        kernel = kernel.float()
        dot = F.conv3d(emb_feat, kernel, stride = 1)
        emb_feat_norm = F.conv3d(emb_feat ** 2, kernel_one, stride = 1) ** 0.5 + 1e-7
        score = dot/emb_feat_norm.expand(dot.size())
        return score.data

    def PairPos(self, pos_d1, pos_h1, pos_w1, pos_d2, pos_h2, pos_w2, trainRegion) :

        pos1 = [(pos_d1, pos_h1 , pos_w1), (pos_d1, pos_h1, pos_w1 + trainRegion - 1), (pos_d1, pos_h1 + trainRegion - 1, pos_w1), (pos_d1, pos_h1 + trainRegion - 1, pos_w1 + trainRegion - 1), (pos_d1 + trainRegion -1, pos_h1 , pos_w1), (pos_d1 + trainRegion -1, pos_h1, pos_w1 + trainRegion - 1), (pos_d1 + trainRegion -1, pos_h1 + trainRegion - 1, pos_w1), (pos_d1 + trainRegion -1, pos_h1 + trainRegion - 1, pos_w1 + trainRegion - 1)]
        pos2 = [(pos_d2, pos_h2 , pos_w2), (pos_d2, pos_h2, pos_w2 + trainRegion - 1), (pos_d2, pos_h2 + trainRegion - 1, pos_w2), (pos_d2, pos_h2 + trainRegion - 1, pos_w2 + trainRegion - 1), (pos_d2 + trainRegion -1, pos_h2 , pos_w2), (pos_d2 + trainRegion -1, pos_h2, pos_w2 + trainRegion - 1), (pos_d2 + trainRegion -1, pos_h2 + trainRegion - 1, pos_w2), (pos_d2 + trainRegion -1, pos_h2 + trainRegion - 1, pos_w2 + trainRegion - 1)]

        return pos1, pos2

    '''
    Positive loss for a pair of positive matching
    '''
    def PosCosineSimilaritytop1(self, feat1, feat2, pos_d1, pos_h1, pos_w1, pos_d2, pos_h2, pos_w2, variableAllOne) :

        feat1x1 = feat1[:, :, pos_d1, pos_h1, pos_w1].clone().contiguous()
        feat1x1 = feat1x1 / ((torch.sum(feat1x1 ** 2, dim = 1, keepdim= True).expand(feat1x1.size())) ** 0.5)

        tmp_pos_w2 = max(pos_w2 - 1, 0)
        tmp_pos_h2 = max(pos_h2 - 1, 0)
        tmp_pos_d2 = max(pos_d2 - 1, 0)

        tmp_end_d2 = min(pos_d2 + 2, feat2.size()[2])
        tmp_end_h2 = min(pos_h2 + 2, feat2.size()[3])
        tmp_end_w2 = min(pos_w2 + 2, feat2.size()[4])

        featRec = feat2[:, :, tmp_pos_d2 : tmp_end_d2, tmp_pos_h2 : tmp_end_h2, tmp_pos_w2 : tmp_end_w2].clone().contiguous()
        featRecNorm = F.conv3d(featRec ** 2, variableAllOne, stride = 1) ** 0.5 + 1e-7

        return self.CosineSimilarityTopK(featRec, featRecNorm, feat1x1.unsqueeze(2).unsqueeze(3).unsqueeze(4), K = 1)[0][0]


    def PosCosineSimilaritytop1_val(self, feat1, feat2, pos_d1, pos_h1, pos_w1, pos_d2, pos_h2, pos_w2, variableAllOne) :

        feat1x1 = feat1[:, :, pos_d1, pos_h1, pos_w1].clone().contiguous()
        feat1x1_normed = feat1x1 / (((torch.sum(feat1x1 ** 2, dim = 1, keepdim= True).expand(feat1x1.size())) ** 0.5)+ 1e-6)

        tmp_pos_w2 = max(pos_w2 - 1, 0)
        tmp_pos_h2 = max(pos_h2 - 1, 0)
        tmp_pos_d2 = max(pos_d2 - 1, 0)

        tmp_end_d2 = min(pos_d2 + 2, feat2.size()[2])
        tmp_end_h2 = min(pos_h2 + 2, feat2.size()[3])
        tmp_end_w2 = min(pos_w2 + 2, feat2.size()[4])
        featRec = feat2[:, :, tmp_pos_d2 : tmp_end_d2, tmp_pos_h2 : tmp_end_h2, tmp_pos_w2 : tmp_end_w2].clone().contiguous()
        featRecNorm = F.conv3d(featRec ** 2, variableAllOne, stride = 1) ** 0.5 + 1e-7
        try:
            topk_score, topk_d, topk_h, topk_w = self.CosineSimilarityTopK(featRec, featRecNorm, feat1x1_normed.unsqueeze(2).unsqueeze(3).unsqueeze(4), K = 1)
        except Exception as e:
            st()
            print('check')
        (d_val,h_val,w_val) = (topk_d.squeeze(),topk_h.squeeze(),topk_w.squeeze())
        pos_sample = featRec[:,:, d_val,h_val,w_val]

        anchor = feat1x1
        return [anchor,pos_sample]




    def PosSimilarity(self,embeds, posPair, posIndex, topkImg, topkScale, topkD, topkH, topkW):
        # Pair information: image name, scale, W, H
        _,_,featChannel,_,_,_ = list(embeds.shape)
        pair = posPair[posIndex]
        queryIndex = int(pair[0])
        pairIndex = [int(pair[1]), int(pair[2])]
        trainRegion = self.trainRegion

        # Replace with queryIndex, pairIndex[0]. Don't think topkImg will be necessary here.
        info1 = torch.stack([topkImg[queryIndex, pairIndex[0]], topkScale[queryIndex, pairIndex[0]], topkD[queryIndex, pairIndex[0]] - (trainRegion + 1) // 2 + 1, topkH[queryIndex, pairIndex[0]] - (trainRegion + 1) // 2 + 1, topkW[queryIndex, pairIndex[0]] - (trainRegion + 1) // 2 + 1]).to(torch.int32)
        info2 = torch.stack([topkImg[queryIndex, pairIndex[1]], topkScale[queryIndex, pairIndex[1]], topkD[queryIndex, pairIndex[1]] - (trainRegion + 1) // 2 + 1, topkH[queryIndex, pairIndex[1]] - (trainRegion + 1) // 2 + 1, topkW[queryIndex, pairIndex[1]] - (trainRegion + 1) // 2 + 1]).to(torch.int32)

        #infos contain the position of the center

        feat1 = embeds[:,0]
        feat1 = self.resizeEmbed(feat1,info1[1])
        
        feat2 = embeds[:,1]
        feat2 = self.resizeEmbed(feat2,info2[1])

        variableAllOne = torch.ones(1, featChannel, 1, 1, 1).to(self.device)

        pos1, pos2 = self.PairPos(info1[2], info1[3], info1[4], info2[2], info2[3], info2[4], trainRegion)

        posTop1Similarity = []

        for (pair1, pair2) in zip(pos1, pos2):
            posTop1Similarity.append(self.PosCosineSimilaritytop1(feat1, feat2, pair1[0], pair1[1], pair1[2], pair2[0], pair2[1], pair2[2], variableAllOne) )
        return posTop1Similarity


    def Pos_Examples(self,embeds, posPair, posIndex, topkImg, topkScale, topkD, topkH, topkW):
        # Pair information: image name, scale, W, H
        emb3D_e_current,emb3D_g_current = embeds
        _,featChannel,_,_,_ = list(emb3D_e_current.shape)
        pair = posPair[posIndex]
        queryIndex = int(pair[0])
        pairIndex = [int(pair[1]), int(pair[2])]
        trainRegion = self.trainRegion

        # Replace with queryIndex, pairIndex[0]. Don't think topkImg will be necessary here.
        info1 = torch.stack([topkImg[queryIndex, pairIndex[0]], topkScale[queryIndex, pairIndex[0]], topkD[queryIndex, pairIndex[0]] - (trainRegion + 1) // 2 + 1, topkH[queryIndex, pairIndex[0]] - (trainRegion + 1) // 2 + 1, topkW[queryIndex, pairIndex[0]] - (trainRegion + 1) // 2 + 1]).to(torch.int32)
        info2 = torch.stack([topkImg[queryIndex, pairIndex[1]], topkScale[queryIndex, pairIndex[1]], topkD[queryIndex, pairIndex[1]] - (trainRegion + 1) // 2 + 1, topkH[queryIndex, pairIndex[1]] - (trainRegion + 1) // 2 + 1, topkW[queryIndex, pairIndex[1]] - (trainRegion + 1) // 2 + 1]).to(torch.int32)

        #infos contain the position of the center
        feat1 = emb3D_e_current
        feat1 = self.resizeEmbed(feat1,info1[1])
        
        feat2 = emb3D_g_current
        feat2 = self.resizeEmbed(feat2,info2[1])

        variableAllOne = torch.ones(1, featChannel, 1, 1, 1).to(self.device)

        pos1, pos2 = self.PairPos(info1[2], info1[3], info1[4], info2[2], info2[3], info2[4], trainRegion)

        anchors = []
        pos_samples = []

        for (pair1, pair2) in zip(pos1, pos2):
            anchor, pos_sample = self.PosCosineSimilaritytop1_val(feat1, feat2, pair1[0], pair1[1], pair1[2], pair2[0], pair2[1], pair2[2], variableAllOne)
            pos_samples.append(pos_sample.squeeze(0))
            anchors.append(anchor.squeeze(0))

        anchors = torch.stack(anchors)
        pos_samples = torch.stack(pos_samples)

        return (anchors, pos_samples)

    '''
    This will be called in expectation step.
    '''
    def RetrievalResForExpectation(self, pool_g, featQuery): #Tested for rotation

        # emb_e, _, _, _ = pool_e.fetch()
        emb_g, _ = pool_g.fetch()

        # TODO: remove after debugging
        # unprs = pool_e.fetchUnpRs()
        # unpr = unprs[0]
        # rotated_unprs = self.mbr_unpr.rotate2D(torch.stack(unprs), "bilinear") # torch.Size([100, 3, 36, 32, 32])
        # rotated_unpr = rotated_unprs[0:1] # torch.Size([1, 3, 36, 32, 32])

        # for i in range(0, rotated_unpr.shape[2], 9):
        #     summ_writer.summ_rgbs('rotation_hpm/rotated_unpr_angle_{}'.format(str(10*i)), [rotated_unpr[:,:,i]])


        # emb_e = torch.stack(emb_e).unsqueeze(1).to(self.device)
        emb_g = torch.stack(emb_g).unsqueeze(1).to(self.device)

        margin = self.margin
        scales = self.queryScale
        nbPatchTotal = featQuery.shape[0]
        numEmbedGs = len(emb_g)

        #Initialize everything
        resScale = torch.zeros((nbPatchTotal, numEmbedGs)).to(self.device)
        resW = torch.zeros((nbPatchTotal, numEmbedGs)).to(self.device)
        resH = torch.zeros((nbPatchTotal, numEmbedGs)).to(self.device)
        resD = torch.zeros((nbPatchTotal, numEmbedGs)).to(self.device)
        resR = torch.zeros((nbPatchTotal, numEmbedGs)).to(self.device)
        resScore = torch.zeros((nbPatchTotal, numEmbedGs)).to(self.device)

        variableAllOne =  torch.ones(1, featQuery.size()[1], featQuery.size()[2], featQuery.size()[3], featQuery.size()[4]).to(self.device)
        # Loop over all embed_g's
        for k in range(numEmbedGs):
            
            embed = emb_g[k] # torch.Size([1, 32, 16, 16, 16])
            # st()
            searchFeat = self.SearchImgFeat(embed)

            tmpScore = torch.zeros((nbPatchTotal, len(scales))).to(self.device)
            tmpH = torch.zeros((nbPatchTotal, len(scales)), dtype=torch.long).to(self.device)
            tmpD = torch.zeros((nbPatchTotal, len(scales)), dtype=torch.long).to(self.device)
            tmpW = torch.zeros((nbPatchTotal, len(scales)), dtype=torch.long).to(self.device)
            tmpR = torch.zeros((nbPatchTotal, len(scales)), dtype=torch.long).to(self.device)
            tmpScale = torch.zeros((nbPatchTotal, len(scales))).to(self.device)
            
            # We will find features over different scales.
            for j, scale in enumerate(searchFeat.keys()) :
                
                # torch.Size([1, 32, 8, 8, 8])
                featEmbed = searchFeat[scale]
                rotatedFeatEmbed = self.mbr.rotateTensor(featEmbed) # B=1, numAngles, C, D, H, W
                
                rotatedFeatEmbed = rotatedFeatEmbed.squeeze(0) # Drop the batch dimension. torch.Size([36, 32, 8, 8, 8])
    
                # score -> torch.Size([36, 3, 8, 8, 8])
                # featQuery -> torch.Size([3, 1, 2, 2, 2])
                # featEmbed -> torch.Size([36, 1, 9, 9, 9])
                score = self.CosineSimilarity(rotatedFeatEmbed, featQuery, variableAllOne)
                # st()
                # Update tmp matrix
                outD = score.size()[2]
                outH = score.size()[3]
                outW = score.size()[4]
                
                score = score.permute(1, 0, 2, 3, 4)
                score = score.reshape(score.shape[0], -1) # patches, angles*D*H*W

                # selecting the best R,H,W,D
                score, index= score.max(1)
                '''
                        scale1 scale2 ... scaleM
                patch1  
                patch2
                ...
                patchN

                '''
                tmpR[:, j] = index//(outD*outH*outW)
                index -= (tmpR[:, j])*(outD*outH*outW)

                tmpD[:, j] = index//(outH*outW)
                index -= (tmpD[:, j])*(outH*outW)

                tmpH[:, j] = index//outW
                tmpW[:, j] = index%outW
                
                tmpScore[:, j] = score
                tmpScale[:, j] = scale

            # selecting the best scale 
            tmpScore, tmpScaleIndex = tmpScore.max(1)
            tmpScaleIndex = tmpScaleIndex.unsqueeze(1)
            # Update res matrix
            resScore[:, k] = tmpScore
            resScale[:, k] = torch.gather(tmpScale, 1, tmpScaleIndex)[:,0]
            resD[:, k] = torch.gather(tmpD, 1, tmpScaleIndex)[:,0]
            resW[:, k] = torch.gather(tmpW, 1, tmpScaleIndex)[:,0]
            resH[:, k] = torch.gather(tmpH, 1, tmpScaleIndex)[:,0]
            resR[:, k] = torch.gather(tmpR, 1, tmpScaleIndex)[:,0]
            
        # Get *Topk 
        # st()
        topkValue, topkImg = resScore.topk(k = numEmbedGs, dim = 1)
        topkScale = torch.gather(resScale, 1, topkImg.long())
        topkD = torch.gather(resD, 1, topkImg)
        topkW = torch.gather(resW, 1, topkImg)
        topkH = torch.gather(resH, 1, topkImg)
        topkR = torch.gather(resR, 1, topkImg)
        return topkImg, topkScale, topkValue, topkW + margin, topkH + margin, topkD + margin, topkR

    '''
    This version of retrievalRes will be called in 
    maximization step.
    '''
    def RetrievalRes(self, pool_e, ranks, pool_g, featQuery, perm, negativeSamples=False, numRetrievalsForEachQueryEmb=None, top_k = None):
        if numRetrievalsForEachQueryEmb == None:
            numRetrievalsForEachQueryEmb = self.numRetrievalsForEachQueryEmb
        # st()
        if top_k == None:
            top_k = self.topK
        emb_e, _, _, fname_e = pool_e.fetch()
        emb_g, _, _, fname_g = pool_g.fetch()

        emb_e = torch.stack(emb_e).unsqueeze(1)
        emb_g = torch.stack(emb_g).unsqueeze(1)
        num_embed_e = emb_e.shape[0]
        
        if negativeSamples:
            # st()
            _1_percent = int(num_embed_e//100)
            _1_percent = max(_1_percent,2)
            top5 = torch.randint(0, _1_percent, (num_embed_e,))
            top5 = torch.from_numpy(ranks[perm, top5])
        
            bottom5 = torch.randint(num_embed_e - _1_percent, num_embed_e, (num_embed_e,))
            bottom5 = torch.from_numpy(ranks[perm, bottom5])

            top_retrieval = torch.stack([top5, bottom5]).T
            numRetrievalsForEachQueryEmb = 2
            top_k = 2
            # top_retrieval = torch.from_numpy(ranks[perm][:,-numRetrievalsForEachQueryEmb:])
        else:
            top_retrieval = torch.from_numpy(ranks[perm][:,:numRetrievalsForEachQueryEmb])
        
        ranks = top_retrieval

        fname_e = np.array(fname_e)[perm]
        fname_g = np.array(fname_g)[top_retrieval]

        emb_e_p = emb_e[perm]
        emb_g_p = emb_g[top_retrieval]
        
        margin = self.margin
        scales = self.queryScale

        
        topkValue = torch.zeros((ranks.shape[0], top_k))
        topkImg = torch.zeros((ranks.shape[0], top_k))
        topkScale = torch.zeros((ranks.shape[0], top_k))
        topkD = torch.zeros((ranks.shape[0], top_k))
        topkW = torch.zeros((ranks.shape[0], top_k))
        topkH = torch.zeros((ranks.shape[0], top_k))
        topPoolgFnames = []

        variableAllOne =  torch.ones(1, featQuery.size()[1], featQuery.size()[2], featQuery.size()[3], featQuery.size()[4]).to(self.device)
        # Loop over each patch extracted from pool_e
        for k in range(featQuery.shape[0]):
            featQuery_k = featQuery[k:k+1]
            #Initialize everything
            resScale = torch.zeros((1, numRetrievalsForEachQueryEmb)).to(self.device)
            resW = torch.zeros((1, numRetrievalsForEachQueryEmb)).to(self.device)
            resH = torch.zeros((1, numRetrievalsForEachQueryEmb)).to(self.device)
            resD = torch.zeros((1, numRetrievalsForEachQueryEmb)).to(self.device)
            resScore = torch.zeros((1, numRetrievalsForEachQueryEmb)).to(self.device)
            # Loop over corresponding best matching embeddings in pool_g.
            # Hard positives will be extracted from these best matches.
            for i in range(numRetrievalsForEachQueryEmb):
                embed = emb_g_p[k, i]
                searchFeat = self.SearchImgFeat(embed)
                tmpScore = torch.zeros((1, len(scales))).to(self.device)
                tmpH = torch.zeros((1, len(scales)), dtype=torch.long).to(self.device)
                tmpD = torch.zeros((1, len(scales)), dtype=torch.long).to(self.device)
                tmpW = torch.zeros((1, len(scales)), dtype=torch.long).to(self.device)

                tmpScale = torch.zeros((1, len(scales))).to(self.device)
                
                # We will find features over different scales.
                for j, scale in enumerate(searchFeat.keys()) :
                    
                    # torch.Size([1, 4, 24, 41, 58])
                    featEmbed = searchFeat[scale]

                    # score -> torch.Size([1, 1, 23, 40, 57])
                    # featQuery -> torch.Size([10, 4, 2, 2, 2])
                    # featQuery_k -> torch.Size([1, 4, 2, 2, 2])
                    score = self.CosineSimilarity(featEmbed, featQuery_k, variableAllOne)
                    # st()
                    # Update tmp matrix
                    outD = score.size()[2]
                    outH = score.size()[3]
                    outW = score.size()[4]
                    
                    score = score.view(score.size()[1], outW * outH * outD)
                    
                    # selecting the best H,W,D

                    score, index= score.max(1)
                    
                    tmpD[:, j] = index//(outH*outW)
                    index -= (tmpD[:, j])*(outH*outW)
                    tmpH[:, j] = index//outW
                    tmpW[:, j] = index%outW
                    tmpScore[:, j] = score
                    tmpScale[:, j] = scale

                # selecting the best scale 
                tmpScore, tmpScaleIndex = tmpScore.max(1)
                tmpScaleIndex = tmpScaleIndex.unsqueeze(1)
                # Update res matrix
                resScore[:, i] = tmpScore
                resScale[:, i] = torch.gather(tmpScale, 1, tmpScaleIndex)
                resD[:, i] = torch.gather(tmpD, 1, tmpScaleIndex)
                resW[:, i] = torch.gather(tmpW, 1, tmpScaleIndex)
                resH[:, i] = torch.gather(tmpH, 1, tmpScaleIndex)
            
            # Get *Topk 
            topkValue_, topkImg_ = resScore.topk(k = top_k, dim = 1)
            
            topkScale_ = torch.gather(resScale, 1, topkImg_.long())
            topkD_ = torch.gather(resD, 1, topkImg_)
            topkW_ = torch.gather(resW, 1, topkImg_)
            topkH_ = torch.gather(resH, 1, topkImg_)
            
            topPoolgFnames.append(fname_g[k][topkImg_.cpu()])
            topkValue[k] = topkValue_[0]
            topkImg[k] = topkImg_[0]
            topkScale[k] = topkScale_[0]
            topkD[k] = topkD_[0]
            topkW[k] = topkW_[0]
            topkH[k] = topkH_[0]

        return topkImg, topkScale, topkValue, topkW + margin, topkH + margin, topkD + margin, topPoolgFnames, fname_e,ranks

    '''
    Extract random patches (at random scales) from each 3D tensor in pool.
    '''
    def extractPatches(self, pool_e):
        emb_e, _, _, _ = pool_e.fetch()
        emb_e = torch.stack(emb_e)
        emb_e = emb_e.unsqueeze(1)
        num_exms = pool_e.num
        # perm = np.random.permutation(num_exms)
        perm = np.arange(num_exms)
        emb_e = emb_e[perm]
        featChannel = emb_e[0].shape[1]

        featQuery = torch.zeros([num_exms, featChannel, self.searchRegion, self.searchRegion, self.searchRegion]).to(self.device) # Store feature
        count = 0
        for embed in emb_e:
            if self.shouldResizeToRandomScale:
                embed = self.resizeEmbed(embed)
            S, C, D, H, W = embed.shape

            feat_d_pos = np.random.choice(np.arange(self.margin, D - self.margin - self.searchRegion, 1), 1)[0]
            feat_h_pos = np.random.choice(np.arange(self.margin, H - self.margin - self.searchRegion, 1), 1)[0]
            feat_w_pos = np.random.choice(np.arange(self.margin, W - self.margin - self.searchRegion, 1), 1)[0]

            featQuery[count] = embed[:, :, feat_d_pos : feat_d_pos + self.searchRegion, feat_h_pos : feat_h_pos + self.searchRegion, feat_w_pos : feat_w_pos + self.searchRegion].clone()

            count += 1
        return featQuery, perm

    def extractPatches_det(self, pool_e,d_init,h_init,w_init):
        emb_e, _ = pool_e.fetch()
        emb_e = torch.stack(emb_e)
        emb_e = emb_e.unsqueeze(1)
        num_exms = pool_e.num
        # perm = np.random.permutation(num_exms)
        perm = np.arange(num_exms)
        emb_e = emb_e[perm]
        featChannel = emb_e[0].shape[1]

        featQuery = torch.zeros([num_exms, featChannel, self.searchRegion, self.searchRegion, self.searchRegion]).to(self.device) # Store feature
        count = 0
        for embed in emb_e:
            if self.shouldResizeToRandomScale:
                embed = self.resizeEmbed(embed)
            S, C, D, H, W = embed.shape

            feat_d_pos = d_init
            feat_h_pos = h_init
            feat_w_pos = w_init

            featQuery[count] = embed[:, :, feat_d_pos : feat_d_pos + self.searchRegion, feat_h_pos : feat_h_pos + self.searchRegion, feat_w_pos : feat_w_pos + self.searchRegion].clone()

            count += 1
        return featQuery, perm
    
    def CosineSimilarityTopK(self,img_feat, img_feat_norm, kernel, K) :
        dot = F.conv3d(img_feat, kernel, stride = 1)
        score = dot/img_feat_norm.expand_as(dot)
        _, _,score_d, score_h, score_w =  score.size()
        score = score.view(kernel.size()[0],  score_d * score_h * score_w)
        topk_score, topk_index = score.topk(k = K, dim = 1)
        topk_d, topk_h, topk_w = topk_index // (score_h*score_w), (topk_index % (score_h*score_w))//score_w, topk_index % (score_w)
        try:
            assert topk_d[0,0].squeeze() < 1000    
        except Exception as e:
            st()
            print('check')
        return topk_score, topk_d, topk_h, topk_w


    def VotePair(self, topkImg, topkScale, topkW, topkH,topkD,  margin, poolg, poole, ranks):
        # Sample a pair
        emb_e, _, _, fname_e = poole.fetch()
        emb_g, _, _, fname_g = poolg.fetch()

        emb_e = torch.stack(emb_e).unsqueeze(1)
        emb_g = torch.stack(emb_g).unsqueeze(1)

        _,_,featChannel,_,_,_  =  list(emb_g.shape)
        queryIndex = np.random.choice(len(topkImg))

        _ , num_retrieved = list(topkImg.shape)
        imgPair = np.random.choice(range(num_retrieved), 2, replace=False)

        info1 = torch.stack([topkImg[queryIndex, imgPair[0]], topkScale[queryIndex, imgPair[0]], topkD[queryIndex, imgPair[0]] - ( self.validRegion + 1) // 2 + 1, topkH[queryIndex, imgPair[0]] - (self.validRegion + 1) // 2 + 1, topkW[queryIndex, imgPair[0]] - (self.validRegion + 1) // 2 + 1]).to(torch.int32)
        info2 = torch.stack([topkImg[queryIndex, imgPair[1]], topkScale[queryIndex, imgPair[1]], topkD[queryIndex, imgPair[1]] - ( self.validRegion + 1) // 2 + 1, topkH[queryIndex, imgPair[1]] - (self.validRegion + 1) // 2 + 1, topkW[queryIndex, imgPair[0]] - (self.validRegion + 1) // 2 + 1]).to(torch.int32)

        emb_g_index_1 = ranks[queryIndex,int(info1[0])]
        feat1 = emb_g[emb_g_index_1]
        feat1 = self.resizeEmbed(feat1,info1[1])
        
        emb_g_index_2 = ranks[queryIndex,int(info2[0])]
        feat2 = emb_g[emb_g_index_2]
        feat2 = self.resizeEmbed(feat2,info2[1])

        # Normalized Feature of validated region in Image 1
        validFeat1 = torch.cat([feat1[:, :, pos_i, pos_j, pos_k].clone()
                            for pos_i, pos_j, pos_k in product(range(info1[2], info1[2] +  self.validRegion),
                                                        range(info1[3], info1[3] +  self.validRegion),
                                                        range(info1[4], info1[4] +  self.validRegion))
                            if pos_i == info1[2] or pos_i == info1[2] + self.validRegion - 1 or pos_j == info1[3] or pos_j == info1[3] + self.validRegion - 1 \
                            or pos_k == info1[4] or pos_k == info1[4] + self.validRegion - 1], dim = 0)

        # poses = [(pos_i,pos_j,pos_k) for pos_i, pos_j, pos_k in product(range(info1[2], info1[2] +  self.validRegion),range(info1[3], info1[3] +  self.validRegion),range(info1[4], info1[4] +  self.validRegion)) if pos_i == info1[2] or pos_i == info1[2] + self.validRegion - 1 or pos_j == info1[3] or pos_j == info1[3] + self.validRegion - 1 or pos_k == info1[4] or pos_k == info1[4] + self.validRegion - 1]
        validFeat1 = validFeat1 / ((torch.sum(validFeat1 ** 2, dim = 1, keepdim = True).expand_as(validFeat1) )**0.5)
        validFeat1 = validFeat1.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        # Top1 match in feat2
        variableAllOne = Variable(torch.ones(1, featChannel, 1, 1,1)).to(self.device)
        
        featNorm2 = F.conv3d(Variable(feat2) ** 2, variableAllOne, stride = 1) ** 0.5 + 1e-7
        

        topkScore1, topkD1, topkH1, topkW1 = self.CosineSimilarityTopK(Variable(feat2), featNorm2, Variable(validFeat1), K = 1)

        pos2 = [[pos_i, pos_j, pos_k]
                for pos_i, pos_j, pos_k in product(range(info2[2], info2[2] + self.validRegion), range(info2[3] , info2[3] + self.validRegion), range(info2[4] , info2[4] + self.validRegion))
                if pos_i == info2[2] or pos_i == info2[2] + self.validRegion - 1  or pos_j == info2[3] or pos_j == info2[3] + self.validRegion - 1 or pos_k == info2[4] or pos_k == info2[4] + self.validRegion - 1]


        pos2 = np.array(pos2).astype('int')
        posD2 = torch.from_numpy(pos2[:, 0]).to(self.device)
        posH2 = torch.from_numpy(pos2[:, 1]).to(self.device)
        posW2 = torch.from_numpy(pos2[:, 2]).to(self.device)
        topkD1, topkH1,topkW1 = topkD1.data.squeeze(),topkH1.data.squeeze(),topkW1.data.squeeze()
        mask = (torch.abs(posD2 - topkD1) <= 1) &  (torch.abs(posH2 - topkH1) <= 1) 
        mask = mask & (torch.abs(posW2 - topkW1) <= 1)
        score = torch.sum(mask)
        return queryIndex, imgPair, score

    def TrainPair(self, nbImgEpoch, topkImg, topkScale, topkW, topkH,  topkD,  poolg, poole,rank, topPoolgFnames):
        nbPairTotal = poole.num
        margin = self.margin
        pairInfo = torch.zeros((nbPairTotal, 4)).to(self.device)
        fnamesForDataLoader = []
        count = 0
        while count < nbPairTotal :
            queryIndex, imgPair, score = self.VotePair(topkImg, topkScale, topkW, topkH, topkD, margin, poolg, poole,rank)
            pairInfo[count, 0] = queryIndex
            pairInfo[count, 1] = int(imgPair[0])
            pairInfo[count, 2] = int(imgPair[1])
            pairInfo[count, 3] = score
            fnamesForDataLoader.append([topPoolgFnames[queryIndex][0][imgPair[0]], topPoolgFnames[queryIndex][0][imgPair[1]]])
            count += 1
            if count % 500 == 499 :
                print(count)
        fnamesForDataLoader = np.stack(fnamesForDataLoader, axis=0)
        score_sort, score_sort_index = pairInfo[:, 3].sort(descending=True)
        fnamesForDataLoader = fnamesForDataLoader[score_sort_index.cpu()]
        pairInfo = pairInfo[score_sort_index]
        return pairInfo[:nbImgEpoch], fnamesForDataLoader[:nbImgEpoch]      

    


    



