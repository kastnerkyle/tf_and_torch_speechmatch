from __future__ import print_function
import os
import argparse
import numpy as np
import torch
from collections import namedtuple

import logging
import shutil
from kkpthlib import Embedding
from kkpthlib import Linear
from kkpthlib import SequenceConv1dStack
#from kkpthlib import LSTMCell
#from kkpthlib import BiLSTMLayer
#from kkpthlib import GaussianAttentionCell
#from kkpthlib import DiscreteMixtureOfLogistics
#from kkpthlib import DiscreteMixtureOfLogisticsCost
#from kkpthlib import AdditiveGaussianNoise
#from tfbldr import scan

seq_len = 48
batch_size = 10
window_mixtures = 10
cell_dropout = .925
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
bn_flag = 0.

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
prenet_dropout = 0.5

mels, mel_mask, text, text_mask, mask, mask_mask, reset = train_itr.next_masked_batch()
text = torch.FloatTensor(text)
text_mask = torch.FloatTensor(text_mask)
mels = torch.FloatTensor(mels)
mel_mask = torch.FloatTensor(mel_mask)

in_mels = mels[:-1, :, :]
in_mel_mask = mel_mask[:-1]
out_mels = mels[1:, :, :]
out_mel_mask = mel_mask[1:]
prenet_dropout=1.


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
conv_text = SequenceConv1dStack([text_e], [emb_dim], n_filts, bn_flag,
                                n_stacks=n_stacks,
                                kernel_sizes=[(1, 1), (3, 3), (5, 5)],
                                name="enc_conv1", random_state=random_state)
from IPython import embed; embed(); raise ValueError()

# text_mask and mask_mask should be the same, doesn't matter which one we use
bitext = BiLSTMLayer([conv_text], [n_filts],
                     enc_units,
                     input_mask=text_mask,
                     name="encode_bidir",
                     init=rnn_init,
                     random_state=random_state)


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
        prenet_dropout = tf.placeholder_with_default(0.5 * tf.ones(shape=[]), shape=[])
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

        projmel1 = Linear([in_mels], [output_size], prenet_units,
                          dropout_flag_prob_keep=prenet_dropout, name="prenet1",
                          random_state=random_state)
        projmel2 = Linear([projmel1], [prenet_units], prenet_units,
                          dropout_flag_prob_keep=prenet_dropout, name="prenet2",
                          random_state=random_state)

        text_char_e, t_c_emb = Embedding(text, vocabulary_size, emb_dim, random_state=random_state,
                                         name="text_char_emb")
        #text_phone_e, t_p_emb = Embedding(text, vocabulary_size, emb_dim, random_state=random_state,
        #                                  name="text_phone_emb")

        #text_e = (1. - mask) * text_char_e + mask * text_phone_e
        text_e = text_char_e

        # masks are either 0 or 1... use embed + voc size of two so that text and mask embs have same size / same impact on the repr
        #mask_e, m_emb = Embedding(mask, 2, emb_dim, random_state=random_state,
        #                          name="mask_emb")
        conv_text = SequenceConv1dStack([text_e], [emb_dim], n_filts, bn_flag,
                                        n_stacks=n_stacks,
                                        kernel_sizes=[(1, 1), (3, 3), (5, 5)],
                                        name="enc_conv1", random_state=random_state)

        # text_mask and mask_mask should be the same, doesn't matter which one we use
        bitext = BiLSTMLayer([conv_text], [n_filts],
                             enc_units,
                             input_mask=text_mask,
                             name="encode_bidir",
                             init=rnn_init,
                             random_state=random_state)


        def step(inp_t, inp_mask_t,
                 corr_inp_t,
                 att_w_tm1, att_k_tm1, att_h_tm1, att_c_tm1,
                 h1_tm1, c1_tm1, h2_tm1, c2_tm1):

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

            output, s = LSTMCell([corr_inp_t, att_w_t, att_h_t],
                                 [prenet_units, 2 * enc_units, dec_units],
                                 h1_tm1, c1_tm1, dec_units,
                                 input_mask=inp_mask_t,
                                 random_state=random_state,
                                 cell_dropout=cell_dropout,
                                 name="rnn1", init=rnn_init)
            h1_t = s[0]
            c1_t = s[1]

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
        #steps = tf.Variable(0.)
        #learning_rate = tf.train.exponential_decay(0.001, steps, staircase=True,
        #                                           decay_steps=50000, decay_rate=0.5)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, use_locking=True)
        grad, var = zip(*optimizer.compute_gradients(loss))
        grad, _ = tf.clip_by_global_norm(grad, 10.)
        #train_step = optimizer.apply_gradients(zip(grad, var), global_step=steps)
        train_step = optimizer.apply_gradients(zip(grad, var))

    things_names = ["mels",
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
            vs.loss, vs.train_step]

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
    from IPython import embed; embed(); raise ValueError()
    l = r[-2]
    _ = r[-1]

    # set next inits
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
    return l, None, stateful_args

with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        loop(sess, train_itr, {}, stateful_args)
        print(i)
    #
    #run_loop(sess,
    #         loop, train_itr,
    #         loop, train_itr,
    #         n_steps=1000000,
    #         n_train_steps_per=1000,
    #         train_stateful_args=stateful_args,
    #         n_valid_steps_per=0,
    #         valid_stateful_args=stateful_args)
