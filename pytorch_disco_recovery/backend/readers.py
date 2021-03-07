# import tensorflow as tf
import hyperparams as hyp
import utils.improc

# H = hyp.H
# W = hyp.W
# N = hyp.N
# V = hyp.V
# S = hyp.S

def carla_parser(record):
    keys_to_features={
        'pix_T_cams_raw': tf.FixedLenFeature([], tf.string),
        'cam_T_velos_raw': tf.FixedLenFeature([], tf.string),
        'origin_T_camRs_raw': tf.FixedLenFeature([], tf.string),
        'origin_T_camXs_raw': tf.FixedLenFeature([], tf.string),
        'rgb_camRs_raw': tf.FixedLenFeature([], tf.string),
        'rgb_camXs_raw': tf.FixedLenFeature([], tf.string),
        'xyz_veloXs_raw': tf.FixedLenFeature([], tf.string),
        'boxes3D_raw': tf.FixedLenFeature([], tf.string),
        'tids_raw': tf.FixedLenFeature([], tf.string),
        'scores_raw': tf.FixedLenFeature([], tf.string),
    }
    parsed = tf.parse_single_example(record, keys_to_features)

    pix_T_cams = tf.decode_raw(parsed['pix_T_cams_raw'], tf.float32, name="pix_T_cams_raw")
    cam_T_velos = tf.decode_raw(parsed['cam_T_velos_raw'], tf.float32, name="cam_T_velos_raw")
    origin_T_camRs = tf.decode_raw(parsed['origin_T_camRs_raw'], tf.float32, name="origin_T_camRs_raw")
    origin_T_camXs = tf.decode_raw(parsed['origin_T_camXs_raw'], tf.float32, name="origin_T_camXs_raw")
    #
    rgb_camRs = tf.decode_raw(parsed['rgb_camRs_raw'], tf.uint8, name="rgb_camRs")
    rgb_camXs = tf.decode_raw(parsed['rgb_camXs_raw'], tf.uint8, name="rgb_camXs")
    xyz_veloXs = tf.decode_raw(parsed['xyz_veloXs_raw'], tf.float32, name="xyz_veloXs_raw")
    # 
    boxes3D = tf.decode_raw(parsed['boxes3D_raw'], tf.float32, name="boxes3D_raw")
    tids = tf.decode_raw(parsed['tids_raw'], tf.int32, name="tids_raw")
    scores = tf.decode_raw(parsed['scores_raw'], tf.float32, name="scores_raw")

    ## reshape
    pix_T_cams = tf.reshape(pix_T_cams, [S, 4, 4])
    cam_T_velos = tf.reshape(cam_T_velos, [S, 4, 4])
    origin_T_camRs = tf.reshape(origin_T_camRs, [S, 4, 4])
    origin_T_camXs = tf.reshape(origin_T_camXs, [S, 4, 4])
    #
    rgb_camRs = tf.reshape(rgb_camRs, [S, H, W, 3])
    rgb_camXs = tf.reshape(rgb_camXs, [S, H, W, 3])
    # move channel dim inward, like pytorch wants
    rgb_camRs = tf.transpose(rgb_camRs, perm=[0, 3, 1, 2])
    rgb_camXs = tf.transpose(rgb_camXs, perm=[0, 3, 1, 2])
    
    xyz_veloXs = tf.reshape(xyz_veloXs, [S, V, 3])
    #
    boxes3D = tf.reshape(boxes3D, [S, N, 9])
    tids = tf.reshape(tids, [S, N])
    scores = tf.reshape(scores, [S, N])
    
    ## preprocess
    rgb_camRs = utils.improc.preprocess_color_tf(rgb_camRs)
    rgb_camXs = utils.improc.preprocess_color_tf(rgb_camXs)

    return (
        pix_T_cams,
        cam_T_velos,
        origin_T_camRs,
        origin_T_camXs,
        #
        rgb_camRs,
        rgb_camXs,
        xyz_veloXs,
        #
        boxes3D,
        tids,
        scores,
        #
    )

