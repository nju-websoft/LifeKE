import numpy as np
import pandas as pd
import scipy.sparse as sp


class BasicSampler(object):

    def __init__(self, opts, kb, path_save_folder):
        self.options = opts
        self.kb = kb
        self.train_data = None
        self.path_save_folder = path_save_folder

    def sample_paths(self, model, repeat_times=2):
        opts = self.options

        kb = self.kb.copy()

        kb = kb[['h_id', 'r_id', 't_id']]

        if opts.max_length == 3:
            data = kb.values
            data = pd.DataFrame(data)

            self.train_data = data
            print("# sampled paths:", len(data))
            print("save path to", self.path_save_folder)
            data.to_csv(self.path_save_folder)
            return

        # sampling triples with the h_id-(r_id,t_id) form.

        rtlist = np.unique(kb[['r_id', 't_id']].values, axis=0)

        rtdf = pd.DataFrame(rtlist, columns=['r_id', 't_id'])

        rtdf = rtdf.reset_index().rename({'index': 'tail_id'}, axis='columns')

        rtkb = kb.merge(
            rtdf, left_on=['r_id', 't_id'], right_on=['r_id', 't_id'])

        htail = np.unique(rtkb[['h_id', 'tail_id']].values, axis=0)

        htailmat = sp.csr_matrix((np.ones(len(htail)), (htail[:, 0], htail[:, 1])),
                                 shape=(model.entity_num, rtlist.shape[0]))

        em = np.ndarray(shape=(0, ))
        rtkb['across'] = rtkb.t_id.isin(em)
        rtkb.loc[rtkb.across, 'across'] = opts.beta
        rtkb.loc[rtkb.across == 0, 'across'] = 1 - opts.beta
        rtailkb = rtkb[['h_id', 't_id', 'tail_id', 'across']]

        def gen_tail_dict(x):
            return x.tail_id.values, x.across.values / x.across.sum()

        rtailkb = rtailkb.groupby('h_id').apply(gen_tail_dict)

        rtailkb = pd.DataFrame({'tails': rtailkb})

        # start sampling

        hrt = np.repeat(kb.values, repeat_times, axis=0)

        # for initial triples
        def perform_random(x):
            return np.random.choice(x.tails[0], 1, p=x.tails[1].astype(np.float))

        # else
        def perform_random2(x):
            # calculate depth bias
            pre_c = htailmat[np.repeat(x.pre, x.tails[0].shape[0]), x.tails[0]]
            pre_c[pre_c == 0] = opts.alpha
            pre_c[pre_c == 1] = 1 - opts.alpha
            p = x.tails[1].astype(np.float).reshape(
                [-1, ]) * pre_c.A.reshape([-1, ])
            p = p / p.sum()
            return np.random.choice(x.tails[0], 1, p=p)

        rt_x = rtailkb.loc[hrt[:, 2]].apply(perform_random, axis=1)
        rt_x = rtlist[np.concatenate(rt_x.values)]

        rts = [hrt, rt_x]
        c_length = 5
        while c_length < opts.max_length:
            curr = rtailkb.loc[rt_x[:, 1]]
            curr.loc[:, 'pre'] = hrt[:, 0]

            rt_x = curr.apply(perform_random2, axis=1)
            rt_x = rtlist[np.concatenate(rt_x.values)]

            rts.append(rt_x)
            c_length += 2

        data = np.concatenate(rts, axis=1)
        data = pd.DataFrame(data)

        self.train_data = data
        print("# sampled paths:", len(data))
        print("save path to", self.path_save_folder)
        data.to_csv(self.path_save_folder)
