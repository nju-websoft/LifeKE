import os
import random

import numpy as np
import tensorflow as tf
from utils import get_available_gpus, average_gradients, get_multi_batch_size
from transformer import transformer_model, gelu


class LPModel:
    def __init__(self, args, session, entity_num, relation_num, model_saved_path):

        self.global_step = tf.train.get_or_create_global_step()
        self.args = args
        self.session = session

        self.entity_num = entity_num + 10  # used for mask representations
        self.relation_num = relation_num + 10
        self.mask_token_id = entity_num + 2
        self.mask_token_dict = dict()
        for i in range(args.max_length):
            self.mask_token_dict[i] = self.mask_token_id + i

        self.ent_embeddings = None
        self.rel_embeddings = None
        self.ent_weight = None
        self.rel_weight = None
        self.rel_bias = None

        self.train_seq = None
        self.train_hr_seq = None
        self.loss = None
        self.train_op = None
        self.lr = None
        self.optimizer = None

        self.eval_seq = None
        self.eval_hr_seq = None
        self.eval_seq1 = None
        self.eval_hr_seq1 = None
        self.local_ents = None
        self.entity_probs = None

        self.mask_mat_dict = dict()

        self.model_saved_path = model_saved_path

        total_gpus = len(get_available_gpus())
        print("# available gpus:", total_gpus)
        self.num_gpu = total_gpus

    def init_variables(self):
        self.ent_embeddings = tf.get_variable('entity_embedding', [self.entity_num, self.args.hidden_size],
                                              initializer=tf.contrib.layers.xavier_initializer())
        self.rel_embeddings = tf.get_variable('relation_embedding', [self.relation_num, self.args.hidden_size],
                                              initializer=tf.contrib.layers.xavier_initializer())

        self.ent_weight = self.ent_embeddings
        # self.ent_weight = tf.get_variable("entity_softmax_w", [self.entity_num, self.args.hidden_size],
        #                                   initializer=tf.contrib.layers.xavier_initializer())

        if self.args.rel_weight > 0.0:
            self.rel_weight = self.rel_embeddings
            # self.rel_weight = tf.get_variable("relation_softmax_w", [self.relation_num, self.args.hidden_size],
            #                                   initializer=tf.contrib.layers.xavier_initializer())
            self.rel_bias = tf.get_variable("relation_softmax_b", [self.relation_num],
                                            initializer=tf.contrib.layers.xavier_initializer())

        self.lr = tf.Variable(self.args.learning_rate, trainable=False)
        self.optimizer = tf.train.AdamOptimizer(self.args.learning_rate)

    def dropout(self, features, evaluation, keep_prob=0.5):
        if evaluation:
            return features
        else:
            return tf.nn.dropout(features, keep_prob=keep_prob)

    def get_label_mat(self, label_tensor, soft_label):
        label_tensor = tf.reshape(label_tensor, [-1, 1])
        label_num = label_tensor.shape[0]
        zero_val = (1 - soft_label) / (self.entity_num - 1)
        soft_label_mat = np.zeros([label_num, self.entity_num], dtype=np.float32)
        soft_label_mat.fill(zero_val)
        soft_label_mat = tf.constant(soft_label_mat)
        idx_tensor = tf.reshape(tf.constant([i for i in range(label_num)]), [-1, 1])
        print(type(label_tensor), label_tensor.shape, idx_tensor.shape)
        indices = tf.concat([idx_tensor, label_tensor], axis=1)
        print(type(indices), indices.shape)
        updates = tf.constant([soft_label] * label_num)
        soft_label_mat = tf.tensor_scatter_update(soft_label_mat, indices, updates)
        return soft_label_mat

    def bce_loss(self, inputs, labels):
        ent_embeds = self.ent_embeddings
        soft_targets = self.get_label_mat(labels, self.args.soft_label)
        logs = tf.matmul(inputs, ent_embeds, transpose_b=True)
        log_softmax = tf.nn.log_softmax(logs + 1e-8)
        return tf.reduce_sum(tf.reduce_sum(- soft_targets * log_softmax, axis=1))

    def transformer(self, em_seq, evaluation, num_layers, name='transformer'):
        dropout_prob = 1. - self.args.transformer_keep_prob
        if evaluation:
            dropout_prob = 0.0

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            all_encoder_layers = transformer_model(input_tensor=em_seq,
                                                   attention_mask=None,
                                                   hidden_size=self.args.hidden_size,
                                                   num_hidden_layers=num_layers,
                                                   num_attention_heads=self.args.num_attention_heads,
                                                   intermediate_size=self.args.intermediate_size,
                                                   hidden_dropout_prob=dropout_prob,
                                                   attention_probs_dropout_prob=dropout_prob,
                                                   do_return_all_layers=True)
            outputs = all_encoder_layers[-1]
        return outputs

    def masked_encoder(self, seq, hr_seq, length, mask_position=-1, evaluation=False):
        return self.masked_transformer_encoder(seq, hr_seq, length, mask_position, evaluation)

    def masked_transformer_encoder(self, seq, hr_seq, length, mask_pos, evaluation):
        em_seq = self.lookup_masked_path(seq, length, mask_pos)  # (batch, len, dim)
        assert em_seq.shape[2] % 2 == 0

        flat_hr_output = None
        target_r_em = tf.reshape(em_seq[:, 1], [-1, 1, self.args.hidden_size])

        if hr_seq is not None:
            em_hr_seq = self.lookup_masked_path(hr_seq, length, mask_pos)  # (win*batch, len, dim)
            hr_r_emb = tf.reshape(em_hr_seq[:, 1, :], [-1, self.args.win_size, self.args.hidden_size])

            em_hr_seq = self.dropout(em_hr_seq, evaluation, keep_prob=self.args.input_keep_prob)
            hr_transformer_output = self.transformer(em_hr_seq, evaluation, self.args.num_layers, name='n')
            flat_hr_output = hr_transformer_output[:, mask_pos, :]
            hr_output = tf.reshape(flat_hr_output, [-1, self.args.win_size, self.args.hidden_size])
            hr_r_emb = tf.reshape(hr_transformer_output[:, 1, :], [-1, self.args.win_size, self.args.hidden_size])

            hr_output = self.dropout(hr_output, evaluation, keep_prob=self.args.intermediate_keep_prob)
            hr_output = gelu(hr_output)

            # target_em = tf.reshape(em_seq[:, 0], [-1, 1, self.args.hidden_size])
            # weights = tf.matmul(hr_output, target_em, transpose_b=True)

            weights = tf.matmul(hr_r_emb, target_r_em, transpose_b=True)
            print("weights:", weights.shape)

            attention = tf.nn.softmax(weights)
            print("attention:", attention.shape)
            hr_output = tf.reduce_mean(hr_output * attention, axis=1)  # (batch, dim)

            # hr_output = tf.reduce_mean(hr_output, axis=1)  # (batch, dim)
            hr_output = self.dropout(hr_output, evaluation, keep_prob=self.args.intermediate_keep_prob)
            hr_output = gelu(hr_output)

            enhanced_em_seq = []
            for i in range(length):
                em = em_seq[:, i]
                if i == 0:
                    em = (em + hr_output) / 2
                enhanced_em_seq.append(tf.reshape(em, [-1, 1, self.args.hidden_size]))

            em_seq = tf.concat(enhanced_em_seq, axis=1)  # (batch_size, length, dim)

        em_seq = self.dropout(em_seq, evaluation, keep_prob=self.args.input_keep_prob)
        conv_output = self.transformer(em_seq, evaluation, self.args.second_num_layers, name='triple')[:, mask_pos, :]
        return flat_hr_output, conv_output  # (batch, dim)

    def sampled_loss(self, inputs, labels, w, b, weight=1.0, is_entity=True):
        assert self.args.num_samples < self.entity_num
        if is_entity:
            num_sampled = self.args.num_samples
            num_sampled = max(1, num_sampled)
            # print("# sampled negative samples for entity prediction:", num_sampled)
        else:
            num_sampled = self.args.num_samples * self.relation_num // self.entity_num
            num_sampled = max(1, num_sampled)
            # print("# sampled negative samples for relation prediction:", num_sampled)
        labels = tf.reshape(labels, [-1, 1])
        # print("using sampled softmax loss")
        # losses = tf.nn.sampled_softmax_loss(weights=w,
        #                                     biases=b,
        #                                     labels=labels,
        #                                     inputs=inputs,
        #                                     num_sampled=num_sampled,
        #                                     num_classes=w.shape[0],
        #                                     partition_strategy='div', )
        print("using nce loss")
        losses = tf.nn.nce_loss(weights=w,
                                biases=b,
                                labels=labels,
                                inputs=inputs,
                                num_sampled=num_sampled,
                                num_classes=w.shape[0],
                                partition_strategy='div', )
        return tf.reduce_sum(losses) * weight

    @staticmethod
    def sample(data):
        choices = np.random.choice(len(data), size=len(data), replace=False)
        return data.iloc[choices]

    @staticmethod
    def padding_data(data, batch_size):
        padding_num = batch_size - len(data) % batch_size
        data = np.concatenate([data, np.zeros((padding_num, data.shape[1]), dtype=np.int32)])
        return data, padding_num

    @staticmethod
    def batch_norm(inputs, reuse, is_train=True, scope='bn'):
        return tf.contrib.layers.batch_norm(inputs,
                                            center=True,
                                            scale=True,
                                            is_training=is_train,
                                            reuse=reuse,
                                            scope=scope,
                                            # data_format='NCHW',
                                            updates_collections=tf.GraphKeys.UPDATE_OPS)

    def lookup_masked_path(self, seq, length, mask_position):
        indices = tf.constant([[i, mask_position] for i in range(seq.shape[0])])
        updates = tf.constant([self.mask_token_dict.get(mask_position, self.mask_token_id)] * seq.shape[0])
        seq = tf.tensor_scatter_update(seq, indices, updates)
        ent = seq[:, 0::2]
        rel = seq[:, 1::2]
        ent_em = tf.nn.embedding_lookup(self.ent_embeddings, ent)
        rel_em = tf.nn.embedding_lookup(self.rel_embeddings, rel)
        em_seq = []
        with tf.variable_scope('input_bn', reuse=tf.AUTO_REUSE):
            for i in range(length):
                em = ent_em[:, i // 2] if i % 2 == 0 else rel_em[:, i // 2]
                em_seq.append(tf.reshape(em, [-1, 1, self.args.hidden_size]))
        em_seq = tf.concat(em_seq, axis=1)  # (batch_size, length, dim)
        return em_seq

    def eval_logits(self, inputs, targets=None):
        w = self.ent_weight
        if targets is not None:
            w = tf.nn.embedding_lookup(w, targets)
        return tf.matmul(inputs, tf.transpose(w))

    def compute_seq_loss(self, seq, hr_seq, length):

        mask_position = length - 1  # tail entity prediction
        hr_features, features = self.masked_encoder(seq, hr_seq, length, mask_position=mask_position, evaluation=False)
        labels = seq[:, mask_position]
        tail_prediction_loss = self.bce_loss(features, labels)

        _, features = self.masked_encoder(seq, None, length, mask_position=mask_position, evaluation=False)
        labels = seq[:, mask_position]
        tail_prediction_loss += self.bce_loss(features, labels)

        if self.args.central_ent_weight > 0:
            hr_labels = hr_seq[:, mask_position]
            tail_prediction_loss += self.args.central_ent_weight * self.bce_loss(hr_features, hr_labels)

        head_prediction_loss = 0.
        if self.args.head_ent_weight > 0:
            mask_position = 0  # head entity prediction
            _, features = self.masked_encoder(seq, None, length, mask_position=mask_position, evaluation=False)
            labels = seq[:, mask_position]
            head_prediction_loss = self.args.head_ent_weight * self.bce_loss(features, labels)

        relation_loss = 0.
        if self.args.rel_weight > 0:
            mask_position = 1  # relation prediction
            _, features = self.masked_encoder(seq, None, length, mask_position=mask_position, evaluation=False)
            labels = seq[:, mask_position]
            relation_loss = self.sampled_loss(features, labels, self.rel_weight, self.rel_bias,
                                              weight=self.args.rel_weight, is_entity=False)

        seq_loss = (tail_prediction_loss + head_prediction_loss + relation_loss) / self.args.batch_size
        return seq_loss

    def build_optimization_graph(self):
        batch_size = self.args.batch_size
        self.train_seq = tf.placeholder(tf.int32, [batch_size, self.args.max_length], name='train_seq')
        self.train_hr_seq = tf.placeholder(tf.int32, [self.args.win_size * batch_size, self.args.max_length],
                                           name='train_hr_seq')
        self.loss = self.compute_seq_loss(self.train_seq, self.train_hr_seq, self.args.max_length)
        trainable_vars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, trainable_vars), 2.0)

        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        #     self.train_op = self.optimizer.apply_gradients(zip(grads, trainable_vars),
        #                                                    global_step=self.global_step)

        train_op = self.optimizer.apply_gradients(zip(grads, trainable_vars), global_step=self.global_step)
        ema = tf.train.ExponentialMovingAverage(decay=self.args.decay, num_updates=self.global_step)
        avg_train_op = ema.apply(trainable_vars)
        with tf.control_dependencies([train_op]):
            self.train_op = tf.group(avg_train_op)

    def compute_eval_probs(self, seq, hr_seq, seq1, hr_seq1, length, local_ents):
        mask_position = length - 1

        _, bn_outputs1 = self.masked_encoder(seq, hr_seq, length, mask_position=mask_position, evaluation=True)
        logits1 = self.eval_logits(bn_outputs1, targets=local_ents)
        _, bn_outputs2 = self.masked_encoder(seq, None, length, mask_position=mask_position, evaluation=True)
        logits2 = self.eval_logits(bn_outputs2, targets=local_ents)

        _, bn_outputs1 = self.masked_encoder(seq1, hr_seq1, length, mask_position=mask_position, evaluation=True)
        logits3 = self.eval_logits(bn_outputs1, targets=local_ents)
        _, bn_outputs2 = self.masked_encoder(seq1, None, length, mask_position=mask_position, evaluation=True)
        logits4 = self.eval_logits(bn_outputs2, targets=local_ents)

        return tf.nn.softmax(logits1 + logits2 + logits3 + logits4)

    def build_evaluation_graph(self, local_ent_num, length=3):
        eval_batch_size = get_multi_batch_size(self.args.eval_batch_size, self.num_gpu, self.args.opt_on_cpu)

        self.eval_seq = tf.placeholder(tf.int32, [eval_batch_size, length], name='eval_seq')
        self.eval_hr_seq = tf.placeholder(tf.int32, [eval_batch_size * self.args.win_size, length], name='eval_hr_seq')

        self.eval_seq1 = tf.placeholder(tf.int32, [eval_batch_size, length], name='eval_seq1')
        self.eval_hr_seq1 = tf.placeholder(tf.int32, [eval_batch_size * self.args.win_size, length], name='eval_hr_seq1')

        self.local_ents = tf.placeholder(tf.int32, [local_ent_num], name='local_ents')

        self.entity_probs = self.compute_eval_probs(self.eval_seq, self.eval_hr_seq,
                                                    self.eval_seq1, self.eval_hr_seq1,
                                                    length, self.local_ents)

        tf.identity(self.entity_probs, name="entity_probs")

    def save(self):
        print("saving model to:", self.model_saved_path)
        if not os.path.exists(self.model_saved_path):
            os.makedirs(self.model_saved_path)

        saver = tf.train.Saver()
        model_checkpoint_path = saver.save(self.session, self.model_saved_path + "model.ckpt")
        print("Model saved to", model_checkpoint_path)
