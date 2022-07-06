import os
import pickle

import numpy as np
import pandas as pd
import scipy.sparse as sp


class LocalReader(object):

    def __init__(self, data_path, mode):
        self.mode = mode

        self.data_path = data_path
        self.training_data_path = os.path.join(self.data_path, self.mode + "_data/")
        if not os.path.exists(self.training_data_path):
            os.mkdir(self.training_data_path)

        self.entity_num = None
        self.relation_num = None
        self.relation_num_for_eval = None

        self.train_data = None
        self.test_data = None
        self.valid_data = None
        self.local_train_data = None

        self.entity_id = None
        self.relation_id = None

        self.total_entities = None
        self.filtered_other_entities = None
        self.local_entities = None

        self.kb = None

        self.valid_filtered_mat = None
        self.test_filtered_mat = None

        saved_data_path = self.training_data_path + self.mode + '_trainer_saved.pkl'

        if os.path.exists(saved_data_path):
            print('load data from', saved_data_path)
            (self.entity_num, self.relation_num, self.relation_num_for_eval,
             self.train_data, self.test_data, self.valid_data, self.entity_id, self.relation_id,
             self.filtered_other_entities, self.local_entities) = pickle.load(open(saved_data_path, 'rb'))
        else:
            if "dbp" in self.data_path:
                self.read_data()
            else:
                self.read_fb15k_data()
            self.merge_id()
            self.add_reverse()
            self.reindex_kb()
            self.generate_tail_label()

            print('save trainer to', saved_data_path)
            saved = (self.entity_num, self.relation_num, self.relation_num_for_eval,
                     self.train_data, self.test_data, self.valid_data, self.entity_id, self.relation_id,
                     self.filtered_other_entities, self.local_entities)
            pickle.dump(saved, open(saved_data_path, 'wb'))

        test_filtered_mat_path = self.training_data_path + self.mode + '_test_filtered_mat.npz'
        valid_filtered_mat_path = self.training_data_path + self.mode + '_valid_filtered_mat.npz'
        if os.path.exists(test_filtered_mat_path) and os.path.exists(valid_filtered_mat_path):
            print("load filtered mat from", valid_filtered_mat_path, test_filtered_mat_path)
            self.test_filtered_mat = sp.load_npz(test_filtered_mat_path)
            self.valid_filtered_mat = sp.load_npz(valid_filtered_mat_path)
        else:
            self.generate_filter_mat()
            sp.save_npz(test_filtered_mat_path, self.test_filtered_mat)
            sp.save_npz(valid_filtered_mat_path, self.valid_filtered_mat)
            print("save filtered mat...")
        self.kb = self.train_data

    def read_data(self):
        print("\nread train/test/valid triples from", self.data_path)

        col = ['h', 'r', 't']
        self.test_data = pd.read_csv(self.data_path + 'triples_test', header=None, sep='\t', names=col)
        self.valid_data = pd.read_csv(self.data_path + 'triples_valid', header=None, sep='\t', names=col)

        if self.mode == "local":
            self.train_data = pd.read_csv(self.data_path + 'triples_train', header=None, sep='\t', names=col)
        elif self.mode == "joint":
            self.train_data = pd.read_csv(self.training_data_path + 'joint_triples_train',
                                          header=None, sep='\t', names=col)
            self.local_train_data = pd.read_csv(self.data_path + 'triples_train', header=None, sep='\t', names=col)
        elif self.mode == "dbpedia_joint":
            self.train_data = pd.read_csv(self.training_data_path + 'joint_triples_train',
                                          header=None, sep='\t', names=col)
            self.local_train_data = pd.read_csv(self.data_path + 'triples_train', header=None, sep='\t', names=col)
        else:
            teach_data = pd.read_csv(self.training_data_path + 'teacher_triples_train',
                                     header=None, sep='\t', names=col)
            stu_data = pd.read_csv(self.training_data_path + 'student_triples_train',  header=None, sep='\t', names=col)
            self.train_data = pd.concat([teach_data, stu_data], ignore_index=True, sort=False)
            self.local_train_data = pd.read_csv(self.data_path + 'triples_train', header=None, sep='\t', names=col)

        full_data = pd.concat([self.train_data, self.test_data, self.valid_data], ignore_index=True, sort=False)
        total_entities = pd.concat((full_data['h'], full_data['t']), ignore_index=True, sort=False)
        total_entities = pd.unique(total_entities)
        self.entity_id = pd.Series(range(len(total_entities)), index=total_entities)
        self.total_entities = set(total_entities)
        print("total entity set:", len(total_entities))
        print("total triples:", len(full_data))

        local_data = pd.concat([self.local_train_data, self.test_data, self.valid_data], ignore_index=True, sort=False)
        local_entities = pd.concat((local_data['h'], local_data['t']), ignore_index=True, sort=False)
        local_entities = set(pd.unique(local_entities))
        self.filtered_other_entities = self.total_entities - local_entities
        print("filtered other entities:", len(self.filtered_other_entities))

        total_relations = pd.unique(full_data['r'])
        self.relation_id = pd.Series(range(len(total_relations)), index=total_relations)

        self.entity_num = self.entity_id.shape[0]
        self.relation_num = self.relation_id.shape[0]
        self.local_entities = local_entities

    def read_fb15k_data(self):
        print("\nread train/test/valid triples from", self.data_path)

        col = ['h', 'r', 't']
        self.test_data = pd.read_csv(self.data_path + 'test.txt', header=None, sep='\t', names=col)
        self.valid_data = pd.read_csv(self.data_path + 'valid.txt', header=None, sep='\t', names=col)
        self.train_data = pd.read_csv(self.data_path + 'train.txt', header=None, sep='\t', names=col)

        full_data = pd.concat([self.train_data, self.test_data, self.valid_data], ignore_index=True, sort=False)
        total_entities = pd.concat((full_data['h'], full_data['t']), ignore_index=True, sort=False)
        total_entities = pd.unique(total_entities)
        self.entity_id = pd.Series(range(len(total_entities)), index=total_entities)
        self.total_entities = set(total_entities)
        print("total entity set:", len(total_entities))
        print("total triples:", len(full_data))

        local_data = pd.concat([self.train_data, self.test_data, self.valid_data], ignore_index=True, sort=False)
        local_entities = pd.concat((local_data['h'], local_data['t']), ignore_index=True, sort=False)
        local_entities = set(pd.unique(local_entities))
        self.filtered_other_entities = self.total_entities - local_entities
        print("filtered other entities:", len(self.filtered_other_entities))

        total_relations = pd.unique(full_data['r'])
        self.relation_id = pd.Series(range(len(total_relations)), index=total_relations)

        self.entity_num = self.entity_id.shape[0]
        self.relation_num = self.relation_id.shape[0]
        self.local_entities = local_entities

    def merge_id(self):
        print("convert triples to ids...")
        self.train_data['h_id'] = self.entity_id[self.train_data.h].values
        self.train_data['r_id'] = self.relation_id[self.train_data.r].values
        self.train_data['t_id'] = self.entity_id[self.train_data.start_time].values

        self.test_data['h_id'] = self.entity_id[self.test_data.h].values
        self.test_data['r_id'] = self.relation_id[self.test_data.r].values
        self.test_data['t_id'] = self.entity_id[self.test_data.start_time].values

        self.valid_data['h_id'] = self.entity_id[self.valid_data.h].values
        self.valid_data['r_id'] = self.relation_id[self.valid_data.r].values
        self.valid_data['t_id'] = self.entity_id[self.valid_data.start_time].values

        self.filtered_other_entities = pd.DataFrame({"filter": list(self.filtered_other_entities)})
        self.filtered_other_entities['filter_id'] = self.entity_id[self.filtered_other_entities['filter']].values
        self.local_entities = pd.DataFrame({"local": list(self.local_entities)})
        self.local_entities['local_id'] = self.entity_id[self.local_entities['local']].values

    def add_reverse(self):
        print("add reverse triples...")

        def add_reverse_data(data):
            reversed_data = data.rename(columns={'h_id': 't_id', 't_id': 'h_id'})
            reversed_data.r_id += self.relation_num
            data = pd.concat(([data, reversed_data]), ignore_index=True, sort=False)
            return data

        self.train_data = add_reverse_data(self.train_data)
        self.test_data = add_reverse_data(self.test_data)
        self.valid_data = add_reverse_data(self.valid_data)
        self.relation_num_for_eval = self.relation_num
        self.relation_num *= 2

    def reindex_kb(self):
        """
        Generate new ids for KGs based on the occurrence frequency of entities and relations
        """

        entity_ids = pd.concat([self.train_data.h_id, self.train_data.t_id, ], ignore_index=True)

        tv_ent_ids = np.unique(pd.concat([self.test_data.h_id, self.test_data.t_id,
                                          self.valid_data.t_id, self.valid_data.h_id]))
        not_train_ent_ids = tv_ent_ids[~np.in1d(tv_ent_ids, entity_ids)]

        rids = pd.concat([self.train_data.r_id, ], ignore_index=True)

        def gen_map(ent_ids, rel_ids):
            e_num = ent_ids.groupby(ent_ids.values).size().sort_values()[::-1]
            not_train = pd.Series(np.zeros_like(not_train_ent_ids), index=not_train_ent_ids)
            e_num = pd.concat([e_num, not_train])

            r_num = rel_ids.groupby(rel_ids.values).size().sort_values()[::-1]

            ent_map = pd.Series(range(e_num.shape[0]), index=e_num.index)
            rel_map = pd.Series(range(r_num.shape[0]), index=r_num.index)
            return ent_map, rel_map

        def remap_kb(kb, e_map, r_map):
            kb.loc[:, 'h_id'] = e_map.loc[kb.h_id.values].values
            kb.loc[:, 'r_id'] = r_map.loc[kb.r_id.values].values
            kb.loc[:, 't_id'] = e_map.loc[kb.t_id.values].values
            return kb

        def remap_id(s, rm):
            s = rm.loc[s.values].values
            return s

        entity_map, relation_map = gen_map(entity_ids, rids)

        self.train_data = remap_kb(self.train_data, entity_map, relation_map)
        self.valid_data = remap_kb(self.valid_data, entity_map, relation_map)
        self.test_data = remap_kb(self.test_data, entity_map, relation_map)

        self.entity_id = remap_id(self.entity_id, entity_map)
        self.relation_id = remap_id(self.relation_id, relation_map)

        return not_train_ent_ids

    def generate_tail_label(self):
        print("generate tail labels...")
        full_data = pd.concat([self.train_data, self.test_data, self.valid_data], ignore_index=True, sort=False)
        labels = full_data['t_id'].groupby([full_data['h_id'], full_data['r_id']]).apply(lambda x: pd.unique(x.values))
        labels.name = 't_label'

        self.test_data = self.test_data.join(labels, on=['h_id', 'r_id'])
        self.valid_data = self.valid_data.join(labels, on=['h_id', 'r_id'])

    def generate_filter_mat(self):

        def gen_filter_vector(index_2be_filtered):
            selected_local_index = list(set(self.local_entities['local_id']) - set(index_2be_filtered))
            selected_local_num = len(selected_local_index)
            data = np.array([1] * selected_local_num)
            indptr = np.array([0, selected_local_num])
            indices = np.array(selected_local_index)
            v = sp.csr_matrix((data, indices, indptr), shape=(1, self.entity_num), dtype=np.float32)
            return v

        print('generate filter mat...')
        self.valid_filtered_mat = sp.vstack(self.valid_data.t_label.apply(gen_filter_vector).values, dtype=np.float32)
        self.test_filtered_mat = sp.vstack(self.test_data.t_label.apply(gen_filter_vector).values, dtype=np.float32)
