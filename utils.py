# coding: utf-8

from functools import partial
from multiprocessing import Pool, cpu_count

import os
import logging
import json
import h5py
import numpy as np
from tqdm import tqdm

import transformers
from transformers.data.processors.squad import (
    squad_convert_example_to_features_init,
    squad_convert_example_to_features,
    SquadV2Processor,
    SquadFeatures,
    _is_whitespace,
)

import torch
from torch.utils.data import Dataset



logger = logging.getLogger(__name__)

def load_and_cache_examples(args, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    input_dir = args.data_dir if args.data_dir else "."
    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
        ),
    )

    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", input_dir)

        assert args.data_dir, "data_dir must be set."
        assert not evaluate or args.predict_file, 'When evalute == true, predict_file must be specified.'
        assert evaluate or args.train_file, "at least one of evaluate and train_file must be specified."

        processor = MySquadProcessor()
        if evaluate:
            examples = processor.get_dev_examples("", filename=args.predict_file)
        else:
            examples = processor.get_train_examples("", filename=args.train_file)

        feature_writer = CacheDataWriter(cached_features_file)
        squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
            feature_writer=feature_writer,
            threads=args.threads,
        )

    if args.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    reader = CacheDataReader(cached_features_file, is_training=not evaluate)
    return reader

def squad_convert_example_to_features_try(example, max_seq_length, doc_stride, max_query_length, is_training):
    try:
        ret = squad_convert_example_to_features(example, max_seq_length, doc_stride, max_query_length, "max_length", is_training)
    except:
        logger.warning('error on {}'.format(example.qas_id))
        ret = []

    return ret

def squad_convert_examples_to_features(examples, tokenizer, max_seq_length, doc_stride, max_query_length, is_training, feature_writer, threads=1):


    # Defining helper methods
    threads = min(threads, cpu_count())
    with Pool(threads, initializer=squad_convert_example_to_features_init, initargs=(tokenizer,)) as p:
        annotate_ = partial(
            squad_convert_example_to_features_try,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_training=is_training,
        )

        unique_id = 1000000000
        example_index = 0
        tmp_features = []
        for example_features in tqdm(p.imap(annotate_, examples, chunksize=32), total=len(examples), desc="convert squad examples to features"):
            if not example_features:
                continue

            for example_feature in example_features:
                example_feature.example_index = example_index
                example_feature.unique_id = unique_id
                unique_id += 1
                tmp_features.append(example_feature)

            example_index += 1
            if len(tmp_features) > 10000:
                feature_writer.write_features(tmp_features)
                tmp_features.clear()
        if tmp_features:
            feature_writer.write_features(tmp_features)
            tmp_features.clear()

    feature_writer.write_examples(examples)


class CacheDataWriter():
    def __init__(self, cache_file):
        self.examples_file = cache_file + ".examples"
        self.features_file = cache_file

        self.vl_int_dt = h5py.vlen_dtype("i8")
        self.str_dt = h5py.string_dtype(encoding="utf-8")
        self.tok_is_max_cntxt_dt = np.dtype([("tok_id", "i"), ("flag", "?")])
        self.vl_timc_dt = h5py.vlen_dtype(self.tok_is_max_cntxt_dt)
        self.tok_to_orig_map_dt = np.dtype([("tok_id", "i"), ("orig_txt_id", "i")])
        self.vl_ttom_dt = h5py.vlen_dtype(self.tok_to_orig_map_dt)

        self.feature_vals = {"input_ids": [], "attention_mask": [], "token_type_ids": [],
                             "cls_index": [], "p_mask": [], "example_index": [], "unique_id": [],
                             "paragraph_len": [], "token_is_max_context": [], "tokens": [],
                             "token_to_orig_map": [], "start_position": [], "end_position": [], "is_impossible": []}

        with h5py.File(self.features_file, "w") as hdf:
            g_features = hdf.create_group("features")
            g_features.create_dataset("input_ids", shape=(1,), dtype=self.vl_int_dt, maxshape=(None,))
            g_features.create_dataset("attention_mask", shape=(1,), dtype=self.vl_int_dt, maxshape=(None,))
            g_features.create_dataset("token_type_ids", shape=(1,), dtype=self.vl_int_dt, maxshape=(None,))
            g_features.create_dataset("cls_index", shape=(1,), dtype="i8", maxshape=(None,))
            g_features.create_dataset("p_mask", shape=(1,), dtype=self.vl_int_dt, maxshape=(None,))
            g_features.create_dataset("example_index", shape=(1,), dtype="i8", maxshape=(None,))
            g_features.create_dataset("unique_id", shape=(1,), dtype="i8", maxshape=(None,))
            g_features.create_dataset("paragraph_len", shape=(1,), dtype="i8", maxshape=(None,))
            g_features.create_dataset("token_is_max_context", shape=(1,), dtype=self.vl_timc_dt, maxshape=(None,))
            g_features.create_dataset("tokens", shape=(1,), dtype=self.str_dt, maxshape=(None,))
            g_features.create_dataset("token_to_orig_map", shape=(1,), dtype=self.vl_ttom_dt, maxshape=(None,))
            g_features.create_dataset("start_position", shape=(1,), dtype="i8", maxshape=(None,))
            g_features.create_dataset("end_position", shape=(1,), dtype="i8", maxshape=(None,))
            g_features.create_dataset("is_impossible", shape=(1,), dtype="?", maxshape=(None,))
            g_features.create_dataset("feature_index", shape=(1,), dtype="i8", maxshape=(None,))
            g_features.attrs["size"] = 0
            g_features.attrs["offset"] = 0

            # features が要素毎にリスト化して持っているので、それが dataset になる
            g_dataset = hdf.create_group("dataset")
            g_train_dataset = g_dataset.create_group("train")
            g_eval_dataset = g_dataset.create_group("eval")

            g_dataset.attrs["size"] = g_features.attrs["size"]
            g_train_dataset.attrs["size"] = g_dataset.attrs["size"]
            g_eval_dataset.attrs["size"] = g_dataset.attrs["size"]

            g_train_dataset["all_input_ids"] = g_features["input_ids"]
            g_train_dataset["all_attention_masks"] = g_features["attention_mask"]
            g_train_dataset["all_token_type_ids"] = g_features["token_type_ids"]
            g_train_dataset["all_start_positions"] = g_features["start_position"]
            g_train_dataset["all_end_positions"] = g_features["end_position"]
            g_train_dataset["all_cls_index"] = g_features["cls_index"]
            g_train_dataset["all_p_mask"] = g_features["p_mask"]
            g_train_dataset["all_is_impossible"] = g_features["is_impossible"]

            g_eval_dataset["all_input_ids"] = g_features["input_ids"]
            g_eval_dataset["all_attention_masks"] = g_features["attention_mask"]
            g_eval_dataset["all_token_type_ids"] = g_features["token_type_ids"]
            g_eval_dataset["all_cls_index"] = g_features["cls_index"]
            g_eval_dataset["all_p_mask"] = g_features["p_mask"]
            # all_example_index は 個々の feature の example_index の集約ではなく
            # feature の index ＝ 何番目の feature かを表す
            g_eval_dataset["all_example_index"] = g_features["feature_index"]

    def write_examples(self, examples):
        torch.save(examples, self.examples_file)

    def _set_dataset_size(self, hdf, size):
        gp_dataset = hdf["/dataset"]
        gp_eval_dataset = hdf["/dataset/eval"]
        gp_train_dataset = hdf["/dataset/train"]

        gp_dataset.attrs["size"] = size
        gp_eval_dataset.attrs["size"] = size
        gp_train_dataset.attrs["size"] = size

    def write_features(self, features):
        with h5py.File(self.features_file, "a") as hdf:
            gp_features = hdf["/features"]
            offset = gp_features.attrs["offset"]
            limit = offset + len(features)
            new_size = gp_features.attrs["size"] + len(features)

            for _k, ds in gp_features.items():
                ds.resize(size=(new_size, ))

            for f in features:
                self.feature_vals["input_ids"].append(f.input_ids)
                self.feature_vals["attention_mask"].append(f.attention_mask)
                self.feature_vals["token_type_ids"].append(f.token_type_ids)
                self.feature_vals["cls_index"].append(f.cls_index)
                self.feature_vals["p_mask"].append(f.p_mask)
                self.feature_vals["example_index"].append(f.example_index)
                self.feature_vals["unique_id"].append(f.unique_id)
                self.feature_vals["paragraph_len"].append(f.paragraph_len)
                self.feature_vals["token_is_max_context"].append(np.array([(k, v) for k,v in f.token_is_max_context.items()],
                                                                          dtype=self.tok_is_max_cntxt_dt))
                self.feature_vals["tokens"].append(json.dumps(f.tokens, ensure_ascii=False))
                self.feature_vals["token_to_orig_map"].append(np.array([(k, v) for k,v in f.token_to_orig_map.items()],
                                                                        dtype=self.tok_to_orig_map_dt))
                self.feature_vals["start_position"].append(f.start_position)
                self.feature_vals["end_position"].append(f.end_position)
                self.feature_vals["is_impossible"].append(f.is_impossible)

            for k, v in self.feature_vals.items():
                # limit = offset + len(v)
                gp_features[k][offset:limit] = v
                v.clear()

            f_idxs = [x for x in range(offset, limit)]
            gp_features["feature_index"][offset:limit] = f_idxs

            gp_features.attrs["size"] = new_size
            gp_features.attrs["offset"] = new_size
            self._set_dataset_size(hdf, new_size)


class CacheDataReader():
    def __init__(self, cache_file, is_training=False):
        self.examples_file = cache_file + ".examples"
        self.cache_data = h5py.File(cache_file, "r", swmr=True)
        if is_training:
            self.dataset_group = self.cache_data["/dataset/train"]
            self.data_keys = ["all_input_ids",
                              "all_attention_masks",
                              "all_token_type_ids",
                              "all_start_positions",
                              "all_end_positions",
                              "all_cls_index",
                              "all_p_mask",
                              "all_is_impossible"
                              ]
        else:
            self.dataset_group = self.cache_data["/dataset/eval"]
            self.data_keys = ["all_input_ids",
                              "all_attention_masks",
                              "all_token_type_ids",
                              "all_example_index",
                              "all_cls_index",
                              "all_p_mask",
                              ]

    def __del__(self):
        self.cache_data.close()

        s = super()
        if hasattr(s, "__del__"):
            s.__del__(self)

    def get_item(self, index):
        return tuple(self.dataset_group[key][index] for key in self.data_keys)

    def get_size(self):
        return self.dataset_group.attrs["size"]

    def load_examples(self):
        examples = torch.load(self.examples_file)
        return examples

    def get_features(self):
        features = [self.get_feature(index) for index in range(self.get_size())]
        return features

    def get_feature(self, index):
        token_is_max_context = dict(self.cache_data["/features/token_is_max_context"][index])
        tokens = json.loads(self.cache_data["/features/tokens"][index])
        token_to_orig_map = dict(self.cache_data["/features/token_to_orig_map"][index])

        feature = SquadFeatures(
                      self.cache_data["/features/input_ids"][index],
                      self.cache_data["/features/attention_mask"][index],
                      self.cache_data["/features/token_type_ids"][index],
                      self.cache_data["/features/cls_index"][index],
                      self.cache_data["/features/p_mask"][index],
                      example_index=self.cache_data["/features/example_index"][index],
                      unique_id=self.cache_data["/features/unique_id"][index],
                      paragraph_len=self.cache_data["/features/paragraph_len"][index],
                      token_is_max_context=token_is_max_context,
                      tokens=tokens,
                      token_to_orig_map=token_to_orig_map,
                      start_position=self.cache_data["/features/start_position"][index],
                      end_position=self.cache_data["/features/end_position"][index],
                      is_impossible=self.cache_data["/features/is_impossible"][index],
                  )

        return feature

    def get_feature_value(self, key, index):
        path = "/features/{}".format(key)
        value = self.cache_data[path][index]
        return value


class HDF5Dataset(Dataset):
    def __init__(self, cache_reader):
        self.cache_reader = cache_reader

    def __getitem__(self, index):
        return self.cache_reader.get_item(index)

    def __len__(self):
        return self.cache_reader.get_size()


class MySquadProcessor(SquadV2Processor):
    def _create_examples(self, input_data, set_type):
        is_training = set_type == "train"
        examples = []
        for entry in tqdm(input_data):
            title = entry["title"]
            for paragraph in entry["paragraphs"]:
                context_text = paragraph["context"]
                doc_tokens, char_to_word_offset = MySquadExample.divide_context_text(context_text)

                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    start_position_character = None
                    answer_text = None
                    answers = []

                    if "is_impossible" in qa:
                        is_impossible = qa["is_impossible"]
                    else:
                        is_impossible = False

                    if not is_impossible:
                        if is_training:
                            answer = qa["answers"][0]
                            answer_text = answer["text"]
                            start_position_character = answer["answer_start"]
                        else:
                            answers = qa["answers"]

                    if start_position_character is not None and not is_impossible:
                        start_position = char_to_word_offset[start_position_character]
                        end_position = char_to_word_offset[
                            min(start_position_character + len(answer_text) - 1, len(char_to_word_offset) - 1)
                        ]
                    else:
                        start_position = -1
                        end_position = -1

                    example = MySquadExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        doc_tokens=doc_tokens,
                        answer_text=answer_text,
                        start_position=start_position,
                        end_position=end_position,
                        title=title,
                        answers=answers,
                        is_impossible=is_impossible,
                    )

                    examples.append(example)
        return examples


class MySquadExample(object):
    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 answer_text,
                 start_position,
                 end_position,
                 title,
                 answers=[],
                 is_impossible=False):

        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.answer_text = answer_text
        self.title = title
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.answers = answers

    @staticmethod
    def divide_context_text(context_text):
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True

        for c in context_text:
            if _is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        return doc_tokens, char_to_word_offset
