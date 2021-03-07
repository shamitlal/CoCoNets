# import tensorflow as tf
import numpy as np
import torch

def match(xs, ys): # nested zip
    result = {}
    for i, (x, y) in enumerate(zip(xs, ys)):
        if type(x) == type([]):
            subresult = match(x, y)
            result.update(subresult)
        else:
            result[x] = y
    return result

def load_graph(frozen_graph_filename):
    # with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
    with tf.io.gfile.GFile(frozen_graph_filename, "rb") as f:
        # graph_def = tf.GraphDef()
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="prefix")

    return graph

def pt_to_tf(feat_pt):
    # feat is B x C x Z x Y x X
    # print('converting this thing to tf:', feat_pt.shape)
    feat_tf = feat_pt.permute(0, 3, 4, 2, 1)
    feat_tf = feat_tf.detach().cpu().numpy()
    return feat_tf

def pts_to_tfs(feats_pt):
    # feats is B x S x C x Z x Y x X
    # print('converting this thing to tf:', feats_pt.shape)
    feats_tf = feats_pt.permute(0, 1, 4, 5, 3, 2)
    feats_tf = feats_tf.detach().cpu().numpy()
    return feats_tf

def tf_to_pt(feat_tf):
    # feat is B x H x W x D x C
    # print('converting this thing to pt:', feat_tf.shape)
    feat_pt = np.transpose(feat_tf, axes=[0, 4, 3, 1, 2])
    feat_pt = torch.from_numpy(feat_pt).to('cuda')
    return feat_pt

def tfs_to_pts(feats_tf):
    # feats is B x S x H x W x D x C
    # print('converting this thing to pt:', feats_tf.shape)
    feats_pt = np.transpose(feats_tf, axes=[0, 1, 5, 4, 2, 3])
    feats_pt = torch.from_numpy(feats_pt).to('cuda')
    return feats_pt

class FrozenFeatNet(object):
    def __init__(self, frozen_graph_filename):
        self.graph = load_graph(frozen_graph_filename)
        # self.sess = tf.Session(graph=self.graph)
        self.sess = tf.compat.v1.Session(graph=self.graph)

    def infer_pt(self, feat_input):
        feat_input_tf = pt_to_tf(feat_input)
        feat_output_tf = self.infer([feat_input_tf])
        feat_output = tf_to_pt(feat_output_tf)
        return feat_output
    
    def infer(self, feed):
        [feat_input] = feed
        # feat_input is B x 2 x MH x MW x MD x 4
        
        placeholder_names = ['placeholders/feat_input:0']
        placeholders = [self.graph.get_tensor_by_name('prefix/' + placeholder_name) for placeholder_name in placeholder_names]
        feed_dict = match(placeholders, feed)

        output_node_names = ['prefix/feat_output:0', ]

        tensors_to_run = {}
        with self.graph.as_default() as graph:
            for output_node_name in output_node_names:
                tensors_to_run[output_node_name] = graph.get_tensor_by_name(output_node_name)

        results = self.sess.run(tensors_to_run, feed_dict=feed_dict)

        feats = [results[output_node_name] for output_node_name in output_node_names]

        return feats[0]

class FrozenFlowNet(object):
    def __init__(self, frozen_graph_filename):
        self.graph = load_graph(frozen_graph_filename)
        # self.sess = tf.Session(graph=self.graph)
        self.sess = tf.compat.v1.Session(graph=self.graph)

    def infer_pt(self, feed):
        # pytorch wrapper for infer
        [feat_a, feat_b] = feed
        feat_a_tf = pt_to_tf(feat_a)
        feat_b_tf = pt_to_tf(feat_b)
        feed_tf = [feat_a_tf, feat_b_tf]
        flow_tf = self.infer(feed_tf)
        flow = tf_to_pt(flow_tf)
        return flow
            
    def infer(self, feed):
        [feat_a, feat_b] = feed

        # feat_input is B x 2 x MH x MW x MD x 4
        placeholder_names = ['placeholders/feat_a:0', 'placeholders/feat_b:0']
        placeholders = [self.graph.get_tensor_by_name('prefix/' + placeholder_name) for placeholder_name in placeholder_names]
        feed_dict = match(placeholders, feed)
        output_node_names = ['prefix/flow_output:0', ]

        tensors_to_run = {}
        with self.graph.as_default() as graph:
            for output_node_name in output_node_names:
                tensors_to_run[output_node_name] = graph.get_tensor_by_name(output_node_name)

        results = self.sess.run(tensors_to_run, feed_dict=feed_dict)

        flow_outs = [results[output_node_name] for output_node_name in output_node_names]

        return flow_outs[0]

if __name__ == "__main__":
    SIZE = 32
    MH = SIZE*1 # 
    MW = SIZE*4 # 
    MD = SIZE*4 #

    feat_net = FrozenFeatNet('/projects/katefgroup/cvpr2020_share/frozen_flow_net/feat_model.pb')
    # set_num = np.array([0, ])
    feat_input = np.random.randn(1, 2, MH, MW, MD, 4)
    feat_net.infer([feat_input])

    print('finished featnet test')

    flow_net = FrozenFlowNet('/projects/katefgroup/cvpr2020_share/frozen_flow_net/flow_model_no_dep.pb')
    feature_dim = 16
    MH2 = MH // 2
    MW2 = MW // 2
    MD2 = MD // 2
    feat_a = np.random.randn(1, MH2, MW2, MD2, feature_dim)
    feat_b = np.random.randn(1, MH2, MW2, MD2, feature_dim)
    flow_e = flow_net.infer([feat_a, feat_b])

    print('finished flownet test')
    print(flow_e.shape)
