import math
import os
import pickle
import random
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from path_func import sample_paths, augment_paths, path_uri2id, path_id2uri


class StuKG4LP(object):

    def __init__(self, data_path, mode):

        print("\nInit student KG from", data_path)

        self.data_path = data_path
        self.mode = mode

        self.entity_num = None
        self.relation_num = None
        self.relation_num_for_eval = None

        self.test_triples = None
        self.valid_triples = None
        self.train_triples = None
        self.full_triples = None

        self.entities = None
        self.relations = None
        self.rev_relations = None
        self.local_entities = None

        self.entity_id = None
        self.relation_id = None

        self.entity_id2uri = None
        self.relation_id2uri = None

        self.valid_filter_mat = None
        self.test_filter_mat = None

        self.output_path = os.path.join(data_path, "stu")
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)

        saved_data_path = os.path.join(self.output_path, 'kg_info_saved.pkl')

        if os.path.exists(saved_data_path):
            print('loading kg info from', saved_data_path)
            (self.entities, self.relations, self.rev_relations,
             self.entity_num, self.relation_num, self.relation_num_for_eval,
             self.train_triples, self.test_triples, self.valid_triples,
             self.entity_id, self.relation_id, self.entity_id2uri, self.relation_id2uri) = \
                pickle.load(open(saved_data_path, 'rb'))
        else:
            self.read_triples()
            self.generate_id()
            self.add_reverse()
            self.get_rev_ids()
            self.generate_tail_label()

            saved = (self.entities, self.relations, self.rev_relations,
                     self.entity_num, self.relation_num, self.relation_num_for_eval,
                     self.train_triples, self.test_triples, self.valid_triples,
                     self.entity_id, self.relation_id, self.entity_id2uri, self.relation_id2uri)
            pickle.dump(saved, open(saved_data_path, 'wb'))
            print('kg info saved to', saved_data_path)

        self.local_entities = self.entity_id.values.tolist()

        if mode == "stu":
            test_filter_mat_path = os.path.join(self.output_path, 'test_filter_mat.npz')
            valid_filter_mat_path = os.path.join(self.output_path, 'valid_filter_mat.npz')

            if os.path.exists(test_filter_mat_path) and os.path.exists(valid_filter_mat_path):
                self.test_filter_mat = sp.load_npz(test_filter_mat_path)
                self.valid_filter_mat = sp.load_npz(valid_filter_mat_path)
                print("filter mat loaded from", valid_filter_mat_path, test_filter_mat_path)
            else:
                self.generate_filter_mat()
                sp.save_npz(test_filter_mat_path, self.test_filter_mat)
                sp.save_npz(valid_filter_mat_path, self.valid_filter_mat)
                print("filter mat saved")

        print("entity num:", self.entity_num)
        print("relation num:", self.relation_num)
        print("train/valid/test triples:", len(self.train_triples), len(self.valid_triples), len(self.test_triples))

    def read_triples(self):
        print("\nreading train/test/valid triples from", self.data_path)

        col = ['h', 'r', 't']
        self.test_triples = pd.read_csv(self.data_path + 'test.txt', header=None, sep='\t', names=col, dtype=object)
        self.valid_triples = pd.read_csv(self.data_path + 'valid.txt', header=None, sep='\t', names=col, dtype=object)
        self.train_triples = pd.read_csv(self.data_path + 'train.txt', header=None, sep='\t', names=col, dtype=object)
        self.full_triples = pd.concat([self.train_triples, self.test_triples, self.valid_triples],
                                      ignore_index=True, sort=False)
        print("training triples:", len(self.train_triples))
        print("validation triples:", len(self.valid_triples))
        print("test triples:", len(self.test_triples))

        entities = pd.unique(pd.concat((self.full_triples['h'], self.full_triples['t']), ignore_index=True, sort=False))
        relations = pd.unique(self.full_triples['r'])

        print("generating ids for entities and relations (student mode)...")
        # self.entity_id = pd.Series(range(len(entities)), index=entities)
        ent_freq = pd.concat([self.full_triples['h'], self.full_triples['t']], ignore_index=True).value_counts()
        self.entity_id = pd.Series(range(len(entities)), index=ent_freq.index.tolist())
        self.relation_id = pd.Series(range(len(relations)), index=relations)

        print("entity_id", self.entity_id.index, self.entity_id.dtypes)

        self.entities = list(set(entities))
        self.relations = list(set(relations))

        self.entity_num = len(self.entities)
        self.relation_num = len(self.relations)

        print("total entities:", len(self.entities))
        print("total relations:", len(self.relations))
        print("total triples:", len(self.full_triples))

    def generate_id(self):
        print("converting triples to ids...")
        self.train_triples['h_id'] = self.entity_id[self.train_triples['h']].values
        self.train_triples['r_id'] = self.relation_id[self.train_triples['r']].values
        self.train_triples['t_id'] = self.entity_id[self.train_triples['t']].values

        self.test_triples['h_id'] = self.entity_id[self.test_triples['h']].values
        self.test_triples['r_id'] = self.relation_id[self.test_triples['r']].values
        self.test_triples['t_id'] = self.entity_id[self.test_triples['t']].values

        self.valid_triples['h_id'] = self.entity_id[self.valid_triples['h']].values
        self.valid_triples['r_id'] = self.relation_id[self.valid_triples['r']].values
        self.valid_triples['t_id'] = self.entity_id[self.valid_triples['t']].values

    def add_reverse(self):
        print("adding reverse triples...")

        def add_reverse_data(data):
            reversed_data = data.rename(columns={'h_id': 't_id', 't_id': 'h_id'})
            reversed_data.r_id += self.relation_num
            data = pd.concat(([data, reversed_data]), ignore_index=True, sort=False)
            return data

        self.train_triples = add_reverse_data(self.train_triples)
        self.test_triples = add_reverse_data(self.test_triples)
        self.valid_triples = add_reverse_data(self.valid_triples)
        self.relation_num_for_eval = self.relation_num
        self.relation_num *= 2

    def get_rev_ids(self, suffix="/reverse"):
        print("============")
        print(self.entity_id)
        self.entity_id2uri = pd.Series(data=self.entity_id.index, index=self.entity_id.values)
        print(self.entity_id2uri)
        print("============")
        real_rel_num = len(self.relation_id)
        relation_id2uri = pd.Series(data=self.relation_id.index, index=self.relation_id.values)
        rev_relation_list = []
        rev_relation_id_list = []
        for idx, val in relation_id2uri.iteritems():
            rev_relation_id_list.append(idx + real_rel_num)
            rev_relation_list.append(val + suffix)
        rev_relation_id2uri = pd.Series(data=rev_relation_list, index=rev_relation_id_list)
        self.rev_relations = rev_relation_list
        self.relation_id2uri = pd.concat((relation_id2uri, rev_relation_id2uri))
        # update relation ids
        self.relation_id = pd.Series(data=self.relation_id2uri.index, index=self.relation_id2uri.values)
        assert len(self.relation_id2uri) == 2 * real_rel_num

    def generate_tail_label(self):
        print("generating tail labels...")
        full_data = pd.concat([self.train_triples, self.test_triples, self.valid_triples],
                              ignore_index=True, sort=False)
        labels = full_data['t_id'].groupby([full_data['h_id'], full_data['r_id']]).apply(lambda x: pd.unique(x.values))
        labels.name = 't_label'
        self.test_triples = self.test_triples.join(labels, on=['h_id', 'r_id'])
        self.valid_triples = self.valid_triples.join(labels, on=['h_id', 'r_id'])

    def generate_filter_mat(self):
        local_entity_ids = set(self.entity_id.values)

        def gen_filter_vector(index_2be_filtered):
            selected_local_index = list(local_entity_ids - set(index_2be_filtered))
            selected_local_num = len(selected_local_index)
            data = np.array([1] * selected_local_num)
            indptr = np.array([0, selected_local_num])
            indices = np.array(selected_local_index)
            v = sp.csr_matrix((data, indices, indptr), shape=(1, self.entity_num), dtype=np.float32)
            return v

        print('generating filter mat...')
        self.valid_filter_mat = sp.vstack(self.valid_triples.t_label.apply(gen_filter_vector).values, dtype=np.float32)
        self.test_filter_mat = sp.vstack(self.test_triples.t_label.apply(gen_filter_vector).values, dtype=np.float32)

    def get_paths(self, max_length, repeat_times):
        path_folder = os.path.join(self.data_path, "stu")
        sequence_path = os.path.join(path_folder, "path_len" + str(max_length) + "_repeat" + str(repeat_times))
        print("student path saved path:", sequence_path)

        if os.path.exists(sequence_path):
            paths = pd.read_csv(sequence_path, index_col=0, dtype=object)
            print("paths loaded:", paths.columns.values, len(paths))
            id_paths = path_uri2id(paths, max_length, self.entity_id, self.relation_id)
            print("id paths:", id_paths.columns.values, len(id_paths))
            return paths, id_paths
        else:
            if not os.path.exists(path_folder):
                os.mkdir(path_folder)
            id_paths = sample_paths(self.train_triples, self.entity_num, max_length, repeat_times=repeat_times,
                                    fast=True)
            print("id paths generated:", id_paths.columns.values, len(id_paths))
            paths = path_id2uri(id_paths, max_length, self.entity_id2uri, self.relation_id2uri)
            print("uri paths generated:", paths.columns.values, len(paths))
            paths.to_csv(sequence_path)
            return paths, id_paths


class TeacherKG:

    def __init__(self, data_path):

        print("\nInit teacher KG from", data_path)

        self.data_path = data_path

        self.entity_num = None
        self.relation_num = None
        self.relation_num_for_eval = None

        self.triples = None
        self.entities = None
        self.relations = None
        self.rev_relations = None

        self.entity_id = None
        self.relation_id = None

        self.entity_id2uri = None
        self.relation_id2uri = None

        self.output_path = data_path

        saved_data_path = os.path.join(self.output_path, 'kg_info_saved.pkl')

        if os.path.exists(saved_data_path):
            print('loading kg info from', saved_data_path)
            (self.entities, self.relations, self.rev_relations,
             self.entity_num, self.relation_num, self.relation_num_for_eval,
             self.triples, self.entity_id, self.relation_id, self.entity_id2uri, self.relation_id2uri) = \
                pickle.load(open(saved_data_path, 'rb'))
        else:
            self.read_triples()
            self.generate_id()
            self.add_reverse()
            self.get_rev_ids()

            saved = (self.entities, self.relations, self.rev_relations,
                     self.entity_num, self.relation_num, self.relation_num_for_eval,
                     self.triples, self.entity_id, self.relation_id, self.entity_id2uri, self.relation_id2uri)
            pickle.dump(saved, open(saved_data_path, 'wb'))
            print('kg info saved to', saved_data_path)

        print("entity num:", self.entity_num)
        print("relation num:", self.relation_num)
        print("triples:", len(self.triples))

    def read_triples(self):
        print("reading triples from", self.data_path)

        self.triples = pd.read_csv(self.data_path + 'triples.txt', header=None, sep='\t', names=['h', 'r', 't'],
                                   dtype=object)
        entities = pd.unique(pd.concat((self.triples['h'], self.triples['t']), ignore_index=True, sort=False))
        relations = pd.unique(self.triples['r'])

        self.entity_id = pd.Series(range(len(entities)), index=entities)
        self.relation_id = pd.Series(range(len(relations)), index=relations)

        self.entities = list(set(entities))
        self.relations = list(set(relations))
        self.entity_num = len(self.entities)
        self.relation_num = len(self.relations)
        print("total entities:", len(self.entities))
        print("total relations:", len(self.relations))
        print("total triples:", len(self.triples))

    def generate_id(self):
        print("converting triples to ids...")
        self.triples['h_id'] = self.entity_id[self.triples['h']].values
        self.triples['r_id'] = self.relation_id[self.triples['r']].values
        self.triples['t_id'] = self.entity_id[self.triples['t']].values

    def add_reverse(self):
        print("adding reverse triples...")

        def add_reverse_data(data):
            reversed_data = data.rename(columns={'h_id': 't_id', 't_id': 'h_id'})
            reversed_data.r_id += self.relation_num
            data = pd.concat(([data, reversed_data]), ignore_index=True, sort=False)
            return data

        self.triples = add_reverse_data(self.triples)
        self.relation_num_for_eval = self.relation_num
        self.relation_num *= 2

    def get_rev_ids(self, suffix="/reverse"):
        self.entity_id2uri = pd.Series(data=self.entity_id.index, index=self.entity_id.values)
        real_rel_num = len(self.relation_id)
        relation_id2uri = pd.Series(data=self.relation_id.index, index=self.relation_id.values)
        rev_relation_list = []
        rev_relation_id_list = []
        for idx, val in relation_id2uri.iteritems():
            rev_relation_id_list.append(idx + real_rel_num)
            rev_relation_list.append(val + suffix)
        self.rev_relations = rev_relation_list
        rev_relation_id2uri = pd.Series(data=rev_relation_list, index=rev_relation_id_list)
        self.relation_id2uri = pd.concat((relation_id2uri, rev_relation_id2uri))
        # update relation ids
        self.relation_id = pd.Series(data=self.relation_id2uri.index, index=self.relation_id2uri.values)
        assert len(self.relation_id2uri) == 2 * real_rel_num

    def get_paths(self, max_length, repeat_times):
        path_folder = self.data_path
        sequence_path = os.path.join(path_folder, "path_len" + str(max_length) + "_repeat" + str(repeat_times))
        print("teacher path saved path:", sequence_path)

        if os.path.exists(sequence_path):
            paths = pd.read_csv(sequence_path, index_col=0, dtype=object)
            print("paths loaded:", paths.columns.values, len(paths))
            id_paths = path_uri2id(paths, max_length, self.entity_id, self.relation_id)
            print("id paths:", id_paths.columns.values, len(id_paths))
            return paths, id_paths
        else:
            id_paths = sample_paths(self.triples, self.entity_num, max_length, repeat_times=repeat_times)
            print("id paths generated:", id_paths.columns.values, len(id_paths))
            paths = path_id2uri(id_paths, max_length, self.entity_id2uri, self.relation_id2uri)
            print("uri paths generated:", paths.columns.values, len(paths))
            paths.to_csv(sequence_path)
            return paths, id_paths


class TeachStuKGs4LP:

    def __init__(self, teacher_data_path, stu_data_path, entity_links_path, path_len, path_repeat_times,
                 stu_repeat_times, mode):
        print("\nInit teacher-student KGs...")

        self.stu_data_path = stu_data_path

        self.teacher_kg = TeacherKG(teacher_data_path)
        self.stu_kg = StuKG4LP(stu_data_path, mode)
        self.entity_links = pd.read_csv(entity_links_path, header=None, sep='\t', names=['ent1', 'ent2'], dtype=object)
        print("\nentity links:", len(self.entity_links))

        self.path_len = path_len
        self.path_repeat_times = path_repeat_times
        self.stu_repeat_times = stu_repeat_times

        self.entities = self.teacher_kg.entities + self.stu_kg.entities
        self.relations = self.teacher_kg.relations + self.teacher_kg.rev_relations + \
                         self.stu_kg.relations + self.stu_kg.rev_relations
        self.entity_num = len(self.entities)
        self.relation_num = len(self.relations)

        self.relation_num_for_eval = self.stu_kg.relation_num_for_eval
        self.local_entities = None

        print("total entities:", len(self.entities))
        print("total relations:", len(self.relations))

        self.train_data = None
        self.local_train_data = None

        self.entity_id = None
        self.relation_id = None

        self.triples = self.teacher_kg.triples.copy(deep=True)[['h', 'r', 't']]
        self.train_triples = self.stu_kg.train_triples.copy(deep=True)[['h', 'r', 't']]
        self.test_triples = self.stu_kg.test_triples.copy(deep=True)[['h', 'r', 't']]
        self.valid_triples = self.stu_kg.valid_triples.copy(deep=True)[['h', 'r', 't']]

        self.valid_filter_mat = None
        self.test_filter_mat = None

        self.output_path = stu_data_path + "joint_" + teacher_data_path.split('/')[-2] + "/"
        print("kg info saved path:", self.output_path)
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)

        saved_data_path = os.path.join(self.output_path, 'kg_info_saved.pkl')

        if os.path.exists(saved_data_path):
            print('loading kg info from', saved_data_path)
            (self.entities, self.relations,
             self.entity_num, self.relation_num, self.relation_num_for_eval, self.entity_id, self.relation_id) = \
                pickle.load(open(saved_data_path, 'rb'))
        else:
            self.get_id()
            saved = (self.entities, self.relations,
                     self.entity_num, self.relation_num, self.relation_num_for_eval, self.entity_id, self.relation_id)
            pickle.dump(saved, open(saved_data_path, 'wb'))
            print('kg info saved to', saved_data_path)

        self.uris2id()
        self.augment_paths()
        self.generate_tail_label()

        data = pd.concat([self.train_triples, self.test_triples, self.valid_triples], ignore_index=True, sort=False)
        local_entities = pd.concat((data['h_id'], data['t_id']), ignore_index=True, sort=False)
        local_entities = set(pd.unique(local_entities))
        self.local_entities = list(local_entities)

        test_filter_mat_path = os.path.join(self.output_path, 'test_filter_mat.npz')
        valid_filter_mat_path = os.path.join(self.output_path, 'valid_filter_mat.npz')

        if os.path.exists(test_filter_mat_path) and os.path.exists(valid_filter_mat_path):
            self.test_filter_mat = sp.load_npz(test_filter_mat_path)
            self.valid_filter_mat = sp.load_npz(valid_filter_mat_path)
            print("filter mat loaded from", valid_filter_mat_path, test_filter_mat_path)
        else:
            self.generate_filter_mat()
            sp.save_npz(test_filter_mat_path, self.test_filter_mat)
            sp.save_npz(valid_filter_mat_path, self.valid_filter_mat)
            print("filter mat saved")

        print("entity num:", self.entity_num)
        print("relation num:", self.relation_num)

    def augment_paths(self):
        path_folder = self.output_path
        sequence_path = os.path.join(path_folder, "aug_path_len" + str(self.path_len) +
                                     "_repeat" + str(self.path_repeat_times) + "_" + str(self.stu_repeat_times))
        print("joint path saved path:", sequence_path)

        local_sequence_path = None
        if "fb" in self.stu_data_path and "4fb237" not in path_folder:
            local_path_folder = path_folder.rstrip("/") + "4fb237/"
            local_sequence_path = os.path.join(local_path_folder, "aug_path_len" + str(self.path_len) +
                                               "_repeat" + str(self.path_repeat_times) + "_" +
                                               str(self.stu_repeat_times))
            print("local path saved path:", local_sequence_path)
        elif "wn" in self.stu_data_path and "4wn18srr" not in path_folder:
            local_path_folder = path_folder.rstrip("/") + "4wn18srr/"
            local_sequence_path = os.path.join(local_path_folder, "aug_path_len" + str(self.path_len) +
                                               "_repeat" + str(self.path_repeat_times) + "_" +
                                               str(self.stu_repeat_times))
            print("local path saved path:", local_sequence_path)
        elif "yago" in self.stu_data_path and "4yago3" not in path_folder:
            local_path_folder = path_folder.rstrip("/") + "4yago3/"
            local_sequence_path = os.path.join(local_path_folder, "aug_path_len" + str(self.path_len) +
                                               "_repeat" + str(self.path_repeat_times) + "_" +
                                               str(2))
            print("local path saved path:", local_sequence_path)

        if os.path.exists(sequence_path):
            paths = pd.read_csv(sequence_path, index_col=0, dtype=object)
            print("paths loaded:", paths.columns.values, len(paths))
            if local_sequence_path is not None and os.path.exists(local_sequence_path):
                self.local_train_data = pd.read_csv(local_sequence_path, index_col=0, dtype=object)
                print("local paths loaded:", self.local_train_data.columns.values, len(self.local_train_data))
        else:
            teach_paths, _ = self.teacher_kg.get_paths(self.path_len, self.path_repeat_times)
            stu_paths, _ = self.stu_kg.get_paths(self.path_len, self.path_repeat_times)
            print("augmenting paths...")
            paths, cross_paths = augment_paths(teach_paths, stu_paths, self.teacher_kg.entities, self.stu_kg.entities,
                                               self.entity_links, self.path_len, stu_repeat_times=self.stu_repeat_times)
            paths.to_csv(sequence_path)

        self.train_data = paths
        print("training paths:", len(self.train_data))
        self.train_data = path_uri2id(self.train_data, self.path_len, self.entity_id, self.relation_id)
        if self.local_train_data is not None:
            print("local training paths:", len(self.local_train_data))
            self.local_train_data = path_uri2id(self.local_train_data, self.path_len, self.entity_id, self.relation_id)

    def get_id(self):
        print("generating ids for entities and relations...")
        # entity_list = self.stu_kg.entities + list(set(self.entities) - set(self.stu_kg.entities))
        # random.shuffle(entity_list)

        all_triples = pd.concat([self.triples, self.train_triples, self.test_triples, self.valid_triples],
                                ignore_index=True, sort=False)
        entities = pd.concat([all_triples['h'], all_triples['t']], ignore_index=True, sort=False)
        ent_freq = entities.value_counts()
        entity_list = ent_freq.index.tolist()

        target_ent_set = set(self.stu_kg.entities)
        sorted_target_ent_list = []

        linked_teacher_ents = set(self.entity_links['ent1'].values.tolist())
        sorted_linked_teacher_ent_list = []
        sorted_other_teacher_ent_list = []

        for e in entity_list:
            if e in target_ent_set:
                sorted_target_ent_list.append(e)
            elif e in linked_teacher_ents:
                sorted_linked_teacher_ent_list.append(e)
            else:
                sorted_other_teacher_ent_list.append(e)

        # entity_list = sorted_target_ent_list + sorted_linked_teacher_ent_list + sorted_other_teacher_ent_list
        # entity_list = self.stu_kg.entities + sorted_linked_teacher_ent_list + sorted_other_teacher_ent_list
        random.shuffle(sorted_target_ent_list)
        random.shuffle(sorted_linked_teacher_ent_list)
        # random.shuffle(sorted_other_teacher_ent_list)
        entity_list = sorted_target_ent_list + sorted_linked_teacher_ent_list + sorted_other_teacher_ent_list

        assert len(entity_list) == self.entity_num

        relation_list = all_triples['r'].value_counts().index.tolist()
        total_relation_list = []
        for r in relation_list:
            total_relation_list.append(r)
            total_relation_list.append(r + "/reverse")
        assert len(total_relation_list) == len(self.relations)

        self.entity_id = pd.Series(range(len(self.entities)), index=entity_list)
        # self.relation_id = pd.Series(range(len(self.relations)), index=total_relation_list)
        self.relation_id = pd.Series(range(len(self.relations)), index=self.relations)

    def uris2id(self):
        print("converting triples to ids...")
        self.triples['h_id'] = self.entity_id[self.triples['h']].values
        self.triples['r_id'] = self.relation_id[self.triples['r']].values
        self.triples['t_id'] = self.entity_id[self.triples['t']].values

        self.train_triples['h_id'] = self.entity_id[self.train_triples['h']].values
        self.train_triples['r_id'] = self.relation_id[self.train_triples['r']].values
        self.train_triples['t_id'] = self.entity_id[self.train_triples['t']].values

        self.test_triples['h_id'] = self.entity_id[self.test_triples['h']].values
        self.test_triples['r_id'] = self.relation_id[self.test_triples['r']].values
        self.test_triples['t_id'] = self.entity_id[self.test_triples['t']].values

        self.valid_triples['h_id'] = self.entity_id[self.valid_triples['h']].values
        self.valid_triples['r_id'] = self.relation_id[self.valid_triples['r']].values
        self.valid_triples['t_id'] = self.entity_id[self.valid_triples['t']].values

    def generate_tail_label(self):
        print("generating tail prediction labels...")
        triples = pd.concat([self.train_triples, self.test_triples, self.valid_triples], ignore_index=True, sort=False)
        labels = triples['t_id'].groupby([triples['h_id'], triples['r_id']]).apply(lambda x: pd.unique(x.values))
        labels.name = 't_label'

        self.test_triples = self.test_triples.join(labels, on=['h_id', 'r_id'])
        self.valid_triples = self.valid_triples.join(labels, on=['h_id', 'r_id'])

    def generate_filter_mat(self):

        local_entities = set(self.local_entities)
        idx = np.array(range(self.entity_num))
        for i, ent in enumerate(self.local_entities):
            idx[ent] = i

        def get_local_ids(ents):
            return list(idx[ents])

        def gen_filter_vector(index_2be_filtered):
            selected_index = list(local_entities - set(index_2be_filtered))
            selected_local_index = get_local_ids(selected_index)
            selected_local_num = len(selected_local_index)
            values = np.array([1] * selected_local_num)
            indptr = np.array([0, selected_local_num])
            indices = np.array(selected_local_index)
            v = sp.csr_matrix((values, indices, indptr), shape=(1, len(self.local_entities)), dtype=np.float32)
            """
            csr_matrix((data, indices, indptr), [shape=(M, N)])
            is the standard CSR representation where the column indices for
            row i are stored in ``indices[indptr[i]:indptr[i+1]]`` and their
            corresponding values are stored in ``data[indptr[i]:indptr[i+1]]``.
            If the shape parameter is not supplied, the matrix dimensions
            are inferred from the index arrays.
            """
            return v

        print('generate filter mat...')
        self.valid_filter_mat = sp.vstack(self.valid_triples.t_label.apply(gen_filter_vector).values, dtype=np.float32)
        self.test_filter_mat = sp.vstack(self.test_triples.t_label.apply(gen_filter_vector).values, dtype=np.float32)
