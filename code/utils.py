import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
from datetime import datetime
from rich.console import Console
from rich.table import Table


def get_multi_batch_size(original_batch_size, num_gpu, opt_on_cpu):
    if opt_on_cpu:
        batch_size = original_batch_size * num_gpu
    else:
        batch_size = original_batch_size * (num_gpu - 1)
    if batch_size == 0:
        batch_size = original_batch_size
    return batch_size


def set_seed():
    seed = 2021
    random.seed(seed)
    np.random.seed(seed)
    # tf.random.set_random_seed(seed)
    tf.compat.v1.random.set_random_seed(seed)
    # tf.set_random_seed(seed)
    tf.compat.v1.set_random_seed(seed)


def tf_config(gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def tf_session():
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config, graph=tf.Graph())
    return sess


def write2log_and_print(path, content, is_print=True):
    if is_print:
        print(content)
    with open(path, 'a+') as f:
        print(content, file=f)


def get_dataset_name(path):
    name = path.strip().strip("/").split("/")[-1]
    return name


def get_log_output_path(opts):
    stamp = datetime.now().strftime('%Y%m%d/')

    if opts.mode == "stu":
        data_name = get_dataset_name(opts.data_path)
        log_folder = os.path.join("../logs/", data_name)
        log_folder = os.path.join(log_folder, stamp)
    elif opts.mode == "joint" or opts.mode == "kt":
        data_name1 = get_dataset_name(opts.data_path)
        data_name2 = get_dataset_name(opts.teacher_data_path)
        log_folder = os.path.join("../logs/", data_name1 + "_" + data_name2)
        log_folder = os.path.join(log_folder, stamp)
    else:
        data_name = get_dataset_name(opts.teacher_data_path)
        log_folder = os.path.join("../logs/", data_name)
        log_folder = os.path.join(log_folder, stamp)

    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if opts.mode == "stu":
        log_file_path = log_folder + '%s_%dl_%s_%s.log' % (opts.data_path.split('/')[-2] + "_" + opts.mode,
                                                           opts.max_length, opts.encoder, stamp)
    elif opts.mode == "joint" or opts.mode == "kt":
        log_file_path = log_folder + '%s_%dl_%s_%s.log' % (opts.data_path.split('/')[-2] + "_" +
                                                           opts.teacher_data_path.split('/')[-2] + "_" + opts.mode,
                                                           opts.max_length, opts.encoder, stamp)
    else:
        log_file_path = log_folder + '%s_%dl_%s_%s.log' % (opts.teacher_data_path.split('/')[-2] + "_" + opts.mode,
                                                           opts.max_length, opts.encoder, stamp)

    print("log output path:", log_file_path)
    return stamp, log_file_path


def distribute_model(lp_model, num):
    if lp_model.args.vars_on_cpu:
        with tf.device("/cpu:0"):
            lp_model.init_variables()
        lp_model.build_optimization_graph()
        if num > 0:
            lp_model.build_evaluation_graph(num)
    else:
        lp_model.init_variables()
        lp_model.build_optimization_graph()
        if num > 0:
            lp_model.build_evaluation_graph(num)
    tf.global_variables_initializer().run()


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    gpu_names = [x.name for x in local_device_protos if x.device_type == 'GPU']
    gpu_idx = [int(name.split(":")[-1]) for name in gpu_names]
    print("available GPUs:", gpu_names, gpu_idx)
    return gpu_idx


# def average_gradients(tower_grads):
#     """Calculate the average gradient for each shared variable across all towers.
#
#     Note that this function provides a synchronization point across all towers.
#
#     Args:
#       tower_grads: List of lists of (gradient, variable) tuples. The outer list
#         is over individual gradients. The inner list is over the gradient
#         calculation for each tower.
#     Returns:
#        List of pairs of (gradient, variable) where the gradient has been averaged
#        across all towers.
#     """
#     average_grads = []
#     for grad_and_vars in zip(*tower_grads):
#         # Note that each grad_and_vars looks like the following:
#         #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
#         grads = []
#         for g, var in grad_and_vars:
#             # Add 0 dimension to the gradients to represent the tower.
#             assert g is not None, var.name
#             expanded_g = tf.expand_dims(g, 0)
#             # Append on a 'tower' dimension which we will average over below.
#             grads.append(expanded_g)
#         # Average over the 'tower' dimension.
#         grad = tf.concat(axis=0, values=grads)
#         grad = tf.reduce_mean(grad, 0)
#         # Keep in mind that the Variables are redundant because they are shared
#         # across towers. So .. we will just return the first tower's pointer to
#         # the Variable.
#         v = grad_and_vars[0][1]
#         grad_and_var = (grad, v)
#         average_grads.append(grad_and_var)
#     return average_grads


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, var in grad_and_vars:
            assert g is not None, var.name
            grads.append(g)
        grad = tf.math.add_n(grads) / len(grads)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def print_results(h1, h3, h10, mrr, mr, k=3):
    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Hits@1", justify="center")
    table.add_column("Hits@3", justify="center")
    table.add_column("Hits@10", justify="center")
    table.add_column("MRR", justify="center")
    table.add_column("MR", justify="center")
    table.add_row(str(round(h1, k)), str(round(h3, k)), str(round(h10, k)), str(round(mrr, k)), str(int(mr)))
    console.print(table)


def get_link_path(data_path, teacher_data_path):
    links_path = None
    if "fb" in data_path:
        links_path = os.path.join(teacher_data_path, "links_fb15k237.txt")
    if "yago" in data_path:
        links_path = os.path.join(teacher_data_path, "links_yago3.txt")
    if "wn" in data_path:
        links_path = os.path.join(teacher_data_path, "links_wn18rr.txt")
    return links_path


def padding_data(data, batch_size):
    padding_num = batch_size - len(data) % batch_size
    data = np.concatenate([data, np.zeros((padding_num, data.shape[1]), dtype=np.int32)])
    return data, padding_num
