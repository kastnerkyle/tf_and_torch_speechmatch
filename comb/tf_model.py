from __future__ import print_function
import os
import argparse
import numpy as np
import tensorflow as tf
from collections import namedtuple

import logging
import shutil
from tfbldr.datasets import rsync_fetch, fetch_ljspeech
from tfbldr.datasets import wavfile_caching_mel_tbptt_iterator
from tfbldr.utils import next_experiment_path
from tfbldr import get_logger
from tfbldr import run_loop
from tfbldr.nodes import Linear
from tfbldr.nodes import LSTMCell
from tfbldr.nodes import BiLSTMLayer
from tfbldr.nodes import SequenceConv1dStack
from tfbldr.nodes import Embedding
from tfbldr.nodes import GaussianAttentionCell
from tfbldr.nodes import DiscreteMixtureOfLogistics
from tfbldr.nodes import DiscreteMixtureOfLogisticsCost
from tfbldr.nodes import AdditiveGaussianNoise
from tfbldr import scan

seq_len = 48
batch_size = 10
window_mixtures = 10
cell_dropout = .925
cell_dropout = 1.
#noise_scale = 8.
prenet_units = 128
n_filts = 128
n_stacks = 3
enc_units = 128
dec_units = 512
emb_dim = 15
truncation_len = seq_len
cell_dropout_scale = cell_dropout
epsilon = 1E-8
forward_init = "truncated_normal"
rnn_init = "truncated_normal"

#basedir = "/Tmp/kastner/lj_speech/LJSpeech-1.0/"
#ljspeech = rsync_fetch(fetch_ljspeech, "leto01")

# THESE ARE CANNOT BE PAIRED (SOME MISSING), ITERATOR PAIRS THEM UP BY NAME
#wavfiles = ljspeech["wavfiles"]
#jsonfiles = ljspeech["jsonfiles"]

# THESE HAVE TO BE THE SAME TO ENSURE SPLIT IS CORRECT
train_random_state = np.random.RandomState(3122)
valid_random_state = np.random.RandomState(3122)

fake_random_state = np.random.RandomState(1234)

class FakeItr(object):
    def __init__(self, batch_size, seq_len):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocabulary_sizes=[44, 44]
        self.n_mel_filters = 80

    def next_masked_batch(self):
        # need to make int "strings" of batch_size, random_len (10-50?)
        # need to make batches of 256, 
        # dummy batch sizes from validation iterator in training code
        mels = fake_random_state.randn(self.seq_len, self.batch_size, 80)
        mel_mask = 0. * mels[..., 0] + 1.
        text = fake_random_state.randint(0, 44, size=(145, self.batch_size, 1)).astype("float32")
        text_mask = 0. * text[..., 0] + 1.
        mask = 0. * text_mask + 1.
        mask_mask = 0. * text_mask + 1.
        reset = 0. * mask_mask[0] + 1.
        reset = reset[:, None]
        # mels = (256, 64, 80)
        # mel_mask = (256, 64)
        # text = (145, 64, 1)
        # text_mask = (145, 64)
        # mask = (145, 64)
        # mask_mask = (145, 64)
        # reset = (64, 1)    
        return mels, mel_mask, text, text_mask, mask, mask_mask, reset


train_itr = FakeItr(batch_size, seq_len)
valid_itr = FakeItr(batch_size, seq_len)
#train_itr = wavfile_caching_mel_tbptt_iterator(wavfiles, jsonfiles, batch_size, seq_len, stop_index=.95, shuffle=True, symbol_processing="chars_only", random_state=train_random_state)
#valid_itr = wavfile_caching_mel_tbptt_iterator(wavfiles, jsonfiles, batch_size, seq_len, start_index=.95, shuffle=True, symbol_processing="chars_only", random_state=valid_random_state)

"""
for i in range(10000):
    print(i)
    mels, mel_mask, text, text_mask, mask, mask_mask, reset = train_itr.next_masked_batch()
print("done")
"""

"""
# STRONG CHECK TO ENSURE NO OVERLAP IN TRAIN/VALID
for tai in train_itr.all_indices_:
    assert tai not in valid_itr.all_indices_
for vai in valid_itr.all_indices_:
    assert vai not in train_itr.all_indices_
"""

random_state = np.random.RandomState(1442)
# use the max of the two blended types...
vocabulary_size = max(train_itr.vocabulary_sizes)
output_size = train_itr.n_mel_filters

def create_graph():
    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(2899)

        text = tf.placeholder(tf.float32, shape=[None, batch_size, 1])
        text_mask = tf.placeholder(tf.float32, shape=[None, batch_size])

        #mask = tf.placeholder(tf.float32, shape=[None, batch_size, 1])
        #mask_mask = tf.placeholder(tf.float32, shape=[None, batch_size])

        mels = tf.placeholder(tf.float32, shape=[None, batch_size, output_size])
        mel_mask = tf.placeholder(tf.float32, shape=[None, batch_size])

        bias = tf.placeholder_with_default(tf.zeros(shape=[]), shape=[])
        cell_dropout = tf.placeholder_with_default(cell_dropout_scale * tf.ones(shape=[]), shape=[])
        prenet_dropout = tf.placeholder_with_default(1. * tf.ones(shape=[]), shape=[])
        bn_flag = tf.placeholder_with_default(tf.zeros(shape=[]), shape=[])

        att_w_init = tf.placeholder(tf.float32, shape=[batch_size, 2 * enc_units])
        att_k_init = tf.placeholder(tf.float32, shape=[batch_size, window_mixtures])
        att_h_init = tf.placeholder(tf.float32, shape=[batch_size, dec_units])
        att_c_init = tf.placeholder(tf.float32, shape=[batch_size, dec_units])
        h1_init = tf.placeholder(tf.float32, shape=[batch_size, dec_units])
        c1_init = tf.placeholder(tf.float32, shape=[batch_size, dec_units])
        h2_init = tf.placeholder(tf.float32, shape=[batch_size, dec_units])
        c2_init = tf.placeholder(tf.float32, shape=[batch_size, dec_units])

        in_mels = mels[:-1, :, :]
        in_mel_mask = mel_mask[:-1]
        out_mels = mels[1:, :, :]
        out_mel_mask = mel_mask[1:]

        random_state = np.random.RandomState(1442)
        projmel1 = Linear([in_mels], [output_size], prenet_units,
                          dropout_flag_prob_keep=prenet_dropout, name="prenet1",
                          random_state=random_state)
        random_state = np.random.RandomState(1442)
        projmel2 = Linear([projmel1], [prenet_units], prenet_units,
                          dropout_flag_prob_keep=prenet_dropout, name="prenet2",
                          random_state=random_state)

        random_state = np.random.RandomState(1442)
        text_char_e, t_c_emb = Embedding(text, vocabulary_size, emb_dim, random_state=random_state,
                                         name="text_char_emb")
        #text_phone_e, t_p_emb = Embedding(text, vocabulary_size, emb_dim, random_state=random_state,
        #                                  name="text_phone_emb")

        #text_e = (1. - mask) * text_char_e + mask * text_phone_e
        text_e = text_char_e

        # masks are either 0 or 1... use embed + voc size of two so that text and mask embs have same size / same impact on the repr
        #mask_e, m_emb = Embedding(mask, 2, emb_dim, random_state=random_state,
        #                          name="mask_emb")
        random_state = np.random.RandomState(1442)
        conv_text, conv_intermediates = SequenceConv1dStack([text_e], [emb_dim], n_filts, bn_flag,
                                        n_stacks=n_stacks,
                                        kernel_sizes=[(1, 1), (3, 3), (5, 5)],
                                        name="enc_conv1", random_state=random_state)
        # text_mask and mask_mask should be the same, doesn't matter which one we use
        random_state = np.random.RandomState(1442)
        bitext, bilstm_intermediates = BiLSTMLayer([conv_text], [n_filts],
                                                   enc_units,
                                                   input_mask=text_mask,
                                                   name="encode_bidir",
                                                   init=rnn_init,
                                                   random_state=random_state)


        def step(inp_t, inp_mask_t,
                 corr_inp_t,
                 att_w_tm1, att_k_tm1, att_h_tm1, att_c_tm1,
                 h1_tm1, c1_tm1, h2_tm1, c2_tm1):

            random_state = np.random.RandomState(1442)
            o = GaussianAttentionCell([corr_inp_t], [prenet_units],
                                      (att_h_tm1, att_c_tm1),
                                      att_k_tm1,
                                      bitext,
                                      2 * enc_units,
                                      dec_units,
                                      att_w_tm1,
                                      input_mask=inp_mask_t,
                                      conditioning_mask=text_mask,
                                      #attention_scale=1. / 10.,
                                      attention_scale=1.,
                                      step_op="softplus",
                                      name="att",
                                      random_state=random_state,
                                      cell_dropout=1.,#cell_dropout,
                                      init=rnn_init)
            att_w_t, att_k_t, att_phi_t, s = o
            att_h_t = s[0]
            att_c_t = s[1]

            random_state = np.random.RandomState(1442)
            output, s = LSTMCell([corr_inp_t, att_w_t, att_h_t],
                                 [prenet_units, 2 * enc_units, dec_units],
                                 h1_tm1, c1_tm1, dec_units,
                                 input_mask=inp_mask_t,
                                 random_state=random_state,
                                 cell_dropout=cell_dropout,
                                 name="rnn1", init=rnn_init)
            h1_t = s[0]
            c1_t = s[1]

            random_state = np.random.RandomState(1442)
            output, s = LSTMCell([corr_inp_t, att_w_t, h1_t],
                                 [prenet_units, 2 * enc_units, dec_units],
                                 h2_tm1, c2_tm1, dec_units,
                                 input_mask=inp_mask_t,
                                 random_state=random_state,
                                 cell_dropout=cell_dropout,
                                 name="rnn2", init=rnn_init)
            h2_t = s[0]
            c2_t = s[1]
            return output, att_w_t, att_k_t, att_phi_t, att_h_t, att_c_t, h1_t, c1_t, h2_t, c2_t

        r = scan(step,
                 [in_mels, in_mel_mask, projmel2],
                 [None, att_w_init, att_k_init, None, att_h_init, att_c_init,
                  h1_init, c1_init, h2_init, c2_init])
        output = r[0]
        att_w = r[1]
        att_k = r[2]
        att_phi = r[3]
        att_h = r[4]
        att_c = r[5]
        h1 = r[6]
        c1 = r[7]
        h2 = r[8]
        c2 = r[9]

        random_state = np.random.RandomState(1442)
        pred = Linear([output], [dec_units], output_size, name="out_proj", random_state=random_state)
        """
        mix, means, lins = DiscreteMixtureOfLogistics([proj], [output_size], n_output_channels=1,
                                                      name="dml", random_state=random_state)
        cc = DiscreteMixtureOfLogisticsCost(mix, means, lins, out_mels, 256)
        """

        # correct masking
        cc = (pred - out_mels) ** 2
        #cc = out_mel_mask[..., None] * cc
        #loss = tf.reduce_sum(tf.reduce_sum(cc, axis=-1)) / tf.reduce_sum(out_mel_mask)
        loss = tf.reduce_mean(tf.reduce_sum(cc, axis=-1))

        learning_rate = 0.0001
        #learning_rate = 0.000
        #steps = tf.Variable(0.)
        #learning_rate = tf.train.exponential_decay(0.001, steps, staircase=True,
        #                                           decay_steps=50000, decay_rate=0.5)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, use_locking=True)
        grad, var = zip(*optimizer.compute_gradients(loss))
        grad, _ = tf.clip_by_global_norm(grad, 10.)
        #train_step = optimizer.apply_gradients(zip(grad, var), global_step=steps)
        train_step = optimizer.apply_gradients(zip(grad, var))

    things_names = ["projmel1",
                    "projmel2",
                    "text_char_e",
                    "t_c_emb",
                    "text_e",
                    "conv_text",
                    "bitext",
                    "cc",

                    "mels",
                    "mel_mask",
                    "in_mels",
                    "in_mel_mask",
                    "out_mels",
                    "out_mel_mask",
                    "text",
                    "text_mask",
                    #"mask",
                    #"mask_mask",
                    "bias",
                    "cell_dropout",
                    "prenet_dropout",
                    "bn_flag",
                    "pred",
                    #"mix", "means", "lins",
                    "att_w_init",
                    "att_k_init",
                    "att_h_init",
                    "att_c_init",
                    "h1_init",
                    "c1_init",
                    "h2_init",
                    "c2_init",
                    "att_w",
                    "att_k",
                    "att_phi",
                    "att_h",
                    "att_c",
                    "h1",
                    "c1",
                    "h2",
                    "c2",
                    "loss",
                    "train_step",
                    "learning_rate"]
    things_tf = [eval(name) for name in things_names]
    things_names = things_names + [k for k in sorted(conv_intermediates.keys())]
    things_tf = things_tf + [conv_intermediates[k] for k in sorted(conv_intermediates.keys())]
    things_names = things_names + [k for k in sorted(bilstm_intermediates.keys())]
    things_tf = things_tf + [bilstm_intermediates[k] for k in sorted(bilstm_intermediates.keys())]
    for tn, tt in zip(things_names, things_tf):
        graph.add_to_collection(tn, tt)
    train_model = namedtuple('Model', things_names)(*things_tf)
    return graph, train_model

g, vs = create_graph()

att_w_init = np.zeros((batch_size, 2 * enc_units))
att_k_init = np.zeros((batch_size, window_mixtures))
att_h_init = np.zeros((batch_size, dec_units))
att_c_init = np.zeros((batch_size, dec_units))
h1_init = np.zeros((batch_size, dec_units))
c1_init = np.zeros((batch_size, dec_units))
h2_init = np.zeros((batch_size, dec_units))
c2_init = np.zeros((batch_size, dec_units))

stateful_args = [att_w_init,
                 att_k_init,
                 att_h_init,
                 att_c_init,
                 h1_init,
                 c1_init,
                 h2_init,
                 c2_init]
step_count = 0
def loop(sess, itr, extras, stateful_args):
    """
    global step_count
    global noise_scale
    step_count += 1
    if step_count > 10000:
        step_count = 0
        if noise_scale == 2:
           noise_scale = 1.
        else:
            noise_scale = noise_scale - 2.
        if noise_scale < .5:
            noise_scale = .5
    """
    mels, mel_mask, text, text_mask, mask, mask_mask, reset = itr.next_masked_batch()
    in_m = mels[:-1]
    in_mel_mask = mel_mask[:-1]

    #noise_block = np.clip(random_state.randn(*in_m.shape), -6, 6)
    #in_m = in_m + noise_scale * noise_block

    out_m = mels[1:]
    out_mel_mask = mel_mask[1:]

    att_w_init = stateful_args[0]
    att_k_init = stateful_args[1]
    att_h_init = stateful_args[2]
    att_c_init = stateful_args[3]
    h1_init = stateful_args[4]
    c1_init = stateful_args[5]
    h2_init = stateful_args[6]
    c2_init = stateful_args[7]

    att_w_init *= reset
    att_k_init *= reset
    att_h_init *= reset
    att_c_init *= reset
    h1_init *= reset
    c1_init *= reset
    h2_init *= reset
    c2_init *= reset

    feed = {
            vs.in_mels: in_m,
            vs.in_mel_mask: in_mel_mask,
            vs.out_mels: out_m,
            vs.out_mel_mask: out_mel_mask,
            vs.bn_flag: 0.,
            vs.text: text,
            vs.text_mask: text_mask,
            #vs.mask: mask,
            #vs.mask_mask: mask_mask,
            vs.att_w_init: att_w_init,
            vs.att_k_init: att_k_init,
            vs.att_h_init: att_h_init,
            vs.att_c_init: att_c_init,
            vs.h1_init: h1_init,
            vs.c1_init: c1_init,
            vs.h2_init: h2_init,
            vs.c2_init: c2_init}
    outs = [vs.att_w, vs.att_k,
            vs.att_h, vs.att_c,
            vs.h1, vs.c1, vs.h2, vs.c2,
            vs.att_phi,
            vs.loss, vs.train_step,

            vs.projmel1,
            vs.projmel2,
            vs.text_char_e,
            vs.t_c_emb,
            vs.conv_text,
            vs.bitext,
            vs.pred,
            vs.cc,
            vs.in_mels,
            vs.in_mel_mask,
            vs.out_mels,
            vs.out_mel_mask,
            vs.text,
            vs.text_mask,

            vs.seqconv1d_enc_conv1_conv_0_relu,
            vs.seqconv1d_enc_conv1_conv_2_res,
            vs.seqconv1d_enc_conv1_conv_2_comb,
            vs.seqconv1d_enc_conv1_conv_post,
            vs.seqconv1d_enc_conv1_conv_1_1,
            vs.seqconv1d_enc_conv1_conv_1_0,
            vs.seqconv1d_enc_conv1_conv_1_2,
            vs.seqconv1d_enc_conv1_conv_0_0,
            vs.seqconv1d_enc_conv1_conv_0_1,
            vs.seqconv1d_enc_conv1_conv_0_2,
            vs.seqconv1d_enc_conv1_conv_0_res,
            vs.seqconv1d_enc_conv1_conv_1_comb,
            vs.seqconv1d_enc_conv1_conv_0_bn,
            vs.seqconv1d_enc_conv1_conv_2_bn,
            vs.seqconv1d_enc_conv1_conv_1_relu,
            vs.seqconv1d_enc_conv1_conv_2_relu,
            vs.seqconv1d_enc_conv1_conv_0_comb,
            vs.seqconv1d_enc_conv1_conv_2_2,
            vs.seqconv1d_enc_conv1_conv_2_0,
            vs.seqconv1d_enc_conv1_conv_2_1,
            vs.seqconv1d_enc_conv1_pre,
            vs.seqconv1d_enc_conv1_conv_1_res,
            vs.seqconv1d_enc_conv1_conv_1_bn,

            vs.bilstm_in_proj,
            vs.bilstm_fwd_hidden,
            vs.bilstm_fwd_cell,
            vs.bilstm_rev_hidden,
            vs.bilstm_rev_cell]

    r = sess.run(outs, feed_dict=feed)

    att_w_np = r[0]
    att_k_np = r[1]
    att_h_np = r[2]
    att_c_np = r[3]
    h1_np = r[4]
    c1_np = r[5]
    h2_np = r[6]
    c2_np = r[7]
    att_phi_np = r[8]
    l = r[9]
    _ = r[10]

    projmel1_np = r[11]
    projmel2_np = r[12]
    text_char_e_np = r[13]
    t_c_emb_np = r[14]
    conv_text_np = r[15]
    bitext_np = r[16]
    pred_np = r[17]
    cc_np = r[18]
    in_mels_np = r[19]
    in_mel_mask_np = r[20]
    out_mels_np = r[21]
    out_mel_mask_np = r[22]
    text_np = r[23]
    text_mask_np = r[24]

    # here come the intermediates, rip
    seqconv1d_enc_conv1_conv_0_relu_np = r[25]
    seqconv1d_enc_conv1_conv_2_res_np = r[26]
    seqconv1d_enc_conv1_conv_2_comb_np = r[27]
    seqconv1d_enc_conv1_conv_post_np = r[28]
    seqconv1d_enc_conv1_conv_1_1_np = r[29]
    seqconv1d_enc_conv1_conv_1_0_np = r[30]
    seqconv1d_enc_conv1_conv_1_2_np = r[31]
    seqconv1d_enc_conv1_conv_0_0_np = r[32]
    seqconv1d_enc_conv1_conv_0_1_np = r[33]
    seqconv1d_enc_conv1_conv_0_2_np = r[34]
    seqconv1d_enc_conv1_conv_0_res_np = r[35]
    seqconv1d_enc_conv1_conv_1_comb_np = r[36]
    seqconv1d_enc_conv1_conv_0_bn_np = r[37]
    seqconv1d_enc_conv1_conv_2_bn_np = r[38]
    seqconv1d_enc_conv1_conv_1_relu_np = r[39]
    seqconv1d_enc_conv1_conv_2_relu_np = r[40]
    seqconv1d_enc_conv1_conv_0_comb_np = r[41]
    seqconv1d_enc_conv1_conv_2_2_np = r[42]
    seqconv1d_enc_conv1_conv_2_0_np = r[43]
    seqconv1d_enc_conv1_conv_2_1_np = r[44]
    seqconv1d_enc_conv1_pre_np = r[45]
    seqconv1d_enc_conv1_conv_1_res_np = r[46]
    seqconv1d_enc_conv1_conv_1_bn_np = r[47]

    bilstm_in_proj_np = r[48]
    bilstm_fwd_hidden_np = r[49]
    bilstm_fwd_cell_np = r[50]
    bilstm_rev_hidden_np = r[51]
    bilstm_rev_cell_np = r[52]

    # set next inits
    att_w_init_old = att_w_init
    att_k_init_old = att_k_init
    att_h_init_old = att_h_init
    att_c_init_old = att_c_init
    h1_init_old = h1_init
    c1_init_old = c1_init
    h2_init_old = h2_init
    c2_init_old = c2_init

    att_w_init = att_w_np[-1]
    att_k_init = att_k_np[-1]
    att_h_init = att_h_np[-1]
    att_c_init = att_c_np[-1]
    h1_init = h1_np[-1]
    c1_init = c1_np[-1]
    h2_init = h2_np[-1]
    c2_init = c2_np[-1]

    stateful_args = [att_w_init,
                     att_k_init,
                     att_h_init,
                     att_c_init,
                     h1_init,
                     c1_init,
                     h2_init,
                     c2_init]
    check = {"projmel1": projmel1_np,
             "projmel2": projmel2_np,
             "text_char_e": text_char_e_np,
             "t_c_emb": t_c_emb_np,
             "conv_text": conv_text_np,
             "bitext": bitext_np,
             "pred": pred_np,
             "cc": cc_np,
             "loss": l,
             "in_mels": in_mels_np,
             "in_mel_mask": in_mel_mask_np,
             "out_mels": out_mels_np,
             "out_mel_mask": out_mel_mask_np,
             "text": text_np,
             "text_mask": text_mask_np,

             "seqconv1d_enc_conv1_conv_0_relu": seqconv1d_enc_conv1_conv_0_relu_np,
            "seqconv1d_enc_conv1_conv_2_res": seqconv1d_enc_conv1_conv_2_res_np,
            "seqconv1d_enc_conv1_conv_2_comb": seqconv1d_enc_conv1_conv_2_comb_np,
            "seqconv1d_enc_conv1_conv_post": seqconv1d_enc_conv1_conv_post_np,
            "seqconv1d_enc_conv1_conv_1_1": seqconv1d_enc_conv1_conv_1_1_np,
            "seqconv1d_enc_conv1_conv_1_0": seqconv1d_enc_conv1_conv_1_0_np,
            "seqconv1d_enc_conv1_conv_1_2": seqconv1d_enc_conv1_conv_1_2_np,
            "seqconv1d_enc_conv1_conv_0_0": seqconv1d_enc_conv1_conv_0_0_np,
            "seqconv1d_enc_conv1_conv_0_1": seqconv1d_enc_conv1_conv_0_1_np,
            "seqconv1d_enc_conv1_conv_0_2": seqconv1d_enc_conv1_conv_0_2_np,
            "seqconv1d_enc_conv1_conv_0_res": seqconv1d_enc_conv1_conv_0_res_np,
            "seqconv1d_enc_conv1_conv_1_comb": seqconv1d_enc_conv1_conv_1_comb_np,
            "seqconv1d_enc_conv1_conv_0_bn": seqconv1d_enc_conv1_conv_0_bn_np,
            "seqconv1d_enc_conv1_conv_2_bn": seqconv1d_enc_conv1_conv_2_bn_np,
            "seqconv1d_enc_conv1_conv_1_relu": seqconv1d_enc_conv1_conv_1_relu_np,
            "seqconv1d_enc_conv1_conv_2_relu": seqconv1d_enc_conv1_conv_2_relu_np,
            "seqconv1d_enc_conv1_conv_0_comb": seqconv1d_enc_conv1_conv_0_comb_np,
            "seqconv1d_enc_conv1_conv_2_2": seqconv1d_enc_conv1_conv_2_2_np,
            "seqconv1d_enc_conv1_conv_2_0": seqconv1d_enc_conv1_conv_2_0_np,
            "seqconv1d_enc_conv1_conv_2_1": seqconv1d_enc_conv1_conv_2_1_np,
            "seqconv1d_enc_conv1_pre": seqconv1d_enc_conv1_pre_np,
            "seqconv1d_enc_conv1_conv_1_res": seqconv1d_enc_conv1_conv_1_res_np,
            "seqconv1d_enc_conv1_conv_1_bn": seqconv1d_enc_conv1_conv_1_bn_np,
            "bilstm_in_proj": bilstm_in_proj_np,
            "bilstm_fwd_hidden": bilstm_fwd_hidden_np,
            "bilstm_fwd_cell": bilstm_fwd_cell_np,
            "bilstm_rev_hidden": bilstm_rev_hidden_np,
            "bilstm_rev_cell": bilstm_rev_cell_np,
            "att_w_init": att_w_init_old,
            "att_k_init": att_k_init_old,
            "att_h_init": att_h_init_old,
            "att_c_init": att_c_init_old,
            "h1_init": h1_init_old,
            "c1_init": c1_init_old,
            "h2_init": h2_init_old,
            "c2_init": c2_init_old,
            "att_w": att_w_np,
            "att_k": att_k_np,
            "att_h": att_h_np,
            "att_c": att_c_np,
            "h1": h1_np,
            "c1": c1_np,
            "h2": h2_np,
            "c2": c2_np,
            "att_phi": att_phi_np,
            }
    return l, None, stateful_args, check

with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    rr = loop(sess, train_itr, {}, stateful_args)
    np.savez("saved_tf_0.npz", **rr[-1])
    rr = loop(sess, train_itr, {}, stateful_args)
    np.savez("saved_tf_1.npz", **rr[-1])
    print("saved activations to 'saved_tf_*.npz'")
    #run_loop(sess,
    #         loop, train_itr,
    #         loop, train_itr,
    #         n_steps=1000000,
    #         n_train_steps_per=1000,
    #         train_stateful_args=stateful_args,
    #         n_valid_steps_per=0,
    #         valid_stateful_args=stateful_args)
