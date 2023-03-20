from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from collections import defaultdict, Counter, namedtuple
from nltk.tokenize import WordPunctTokenizer
from nltk.stem.porter import PorterStemmer


def load_data_in_libsvm_format(
    data_path=None, file_prefix=None, feature_size=-1, topk=100
):
    features = []
    dids = []
    initial_list = []
    qids = []
    labels = []
    initial_scores = []
    initial_list_lengths = []
    feature_fin = open(data_path)
    qid_to_idx = {}
    line_num = -1
    for line in feature_fin:
        line_num += 1
        arr = line.strip().split(" ")
        qid = arr[1].split(":")[1]
        if qid not in qid_to_idx:
            qid_to_idx[qid] = len(qid_to_idx)
            qids.append(qid)
            initial_list.append([])
            labels.append([])

        # create query-document information
        qidx = qid_to_idx[qid]
        if len(initial_list[qidx]) == topk:
            continue
        initial_list[qidx].append(line_num)
        label = int(arr[0])
        labels[qidx].append(label)
        did = qid + "_" + str(line_num)
        dids.append(did)

        # read query-document feature vectors
        auto_feature_size = feature_size == -1

        if auto_feature_size:
            feature_size = 5

        features.append([0.0 for _ in range(feature_size)])
        for x in arr[2:]:
            arr2 = x.split(":")
            feature_idx = int(arr2[0]) - 1
            if feature_idx >= feature_size and auto_feature_size:
                features[-1] += [0.0 for _ in range(feature_idx - feature_size + 1)]
                feature_size = feature_idx + 1
            if feature_idx < feature_size:
                features[-1][int(feature_idx)] = float(arr2[1])

    feature_fin.close()

    initial_list_lengths = [len(initial_list[i]) for i in range(len(initial_list))]

    ds = {}
    ds["fm"] = np.array(features)
    ds["lv"] = np.concatenate([np.array(x) for x in labels], axis=0)
    ds["dlr"] = np.cumsum([0] + initial_list_lengths)
    return ds


class Preprocess:
    def __init__(
        self, sw_path, tokenizer=WordPunctTokenizer(), stemmer=PorterStemmer()
    ) -> None:
        with open(sw_path, "r") as stw_file:
            stw_lines = stw_file.readlines()
            stop_words = set([l.strip().lower() for l in stw_lines])
        self.sw = stop_words
        self.tokenizer = tokenizer
        self.stemmer = stemmer

    def pipeline(
        self, text, stem=True, remove_stopwords=True, lowercase_text=True
    ) -> list:
        tokens = []
        for token in self.tokenizer.tokenize(text):
            if remove_stopwords and token.lower() in self.sw:
                continue
            if stem:
                token = self.stemmer.stem(token)
            if lowercase_text:
                token = token.lower()
            tokens.append(token)

        return tokens

# ToDo: Complete the implemenation of process_douments method in the Documents class
class Documents:
    def __init__(self, preprocesser: Preprocess) -> None:
        self.preprocesser: Preprocess = preprocesser
        self.index = defaultdict(defaultdict)  # Index
        self.dl = defaultdict(int)  # Document Length
        self.df = defaultdict(int)  # Document Frequencies
        self.num_docs = 0  # Number of all documents

    def process_documents(self, doc_path: str):
        """_summary_
        Preprocess the collection file(document information). Your implementation should calculate
        all of the class attributes in the __init__ function.
        Parameters
        ----------
        doc_path : str
            Path of the file holding documents ID and their corresponding text
        """
        with open(doc_path, "r") as doc_file:
            for line in tqdm(doc_file, desc="Processing documents"):
                # BEGIN SOLUTION
                did, d_text = line.strip().split("\t")
                d_text = self.preprocesser.pipeline(d_text)
                self.index[did] = d_text
                self.num_docs += 1
                self.dl[did] = len(d_text)
                for term in d_text:
                    self.df[term] += 1
                # END SOLUTION


class Queries:
    def __init__(self, preprocessor: Preprocess) -> None:
        self.preprocessor = preprocessor
        self.qmap = defaultdict(list)
        self.num_queries = 0

    def preprocess_queries(self, query_path):
        with open(query_path, "r") as query_file:
            for line in query_file:
                qid, q_text = line.strip().split("\t")
                q_text = self.preprocessor.pipeline(q_text)
                self.qmap[qid] = q_text
                self.num_queries += 1


__feature_list__ = [
    "bm25",
    "query_term_coverage",
    "query_term_coverage_ratio",
    "stream_length",
    "idf",
    "sum_stream_length_normalized_tf",
    "min_stream_length_normalized_tf",
    "max_stream_length_normalized_tf",
    "mean_stream_length_normalized_tf",
    "var_stream_length_normalized_tf",
    "sum_tfidf",
    "min_tfidf",
    "max_tfidf",
    "mean_tfidf",
    "var_tfidf",
]


@dataclass
class FeatureList:
    f1 = "bm25"  # BM25 value. Parameters: k1 = 1.5, b = 0.75
    f2 = "query_term_coverage"  # number of query terms in the document
    f3 = "query_term_coverage_ratio"  # Ratio of # query terms in the document to # query terms in the query.
    f4 = "stream_length"  # length of document
    f5 = "idf"  # sum of document frequencies
    f6 = "sum_stream_length_normalized_tf"  # Sum over the ratios of each term to document length
    f7 = "min_stream_length_normalized_tf"
    f8 = "max_stream_length_normalized_tf"
    f9 = "mean_stream_length_normalized_tf"
    f10 = "var_stream_length_normalized_tf"
    f11 = "sum_tfidf"  # Sum of tfidf
    f12 = "min_tfidf"
    f13 = "max_tfidf"
    f14 = "mean_tfidf"
    f15 = "var_tfidf"

import numpy as np

class FeatureExtraction:
    def __init__(self, features: dict, documents: Documents, queries: Queries) -> None:
        self.features = features
        self.documents = documents
        self.queries = queries

    # TODO Implement this function
    def extract(self, qid: int, docid: int, **args) -> dict:
        """_summary_
        For each query and document, extract the features requested and store them
        in self.features attribute.
        
        Parameters
        ----------
        qid : int
            _description_
        docid : int
            _description_

        Returns
        -------
        dict
            _description_
        """
        # BEGIN SOLUTION

        # implementation of 15 feature extraction functions
        def bm25(self, query, document,**args): # f1
            k1 = args.get("k1")
            b = args.get("b")
            idf_smoothing = args.get("idf_smoothing")
            n = self.documents.num_docs
            avg_dl = sum(self.documents.dl.values())/n
            dl = len(document)

            bm25 = 0
            for term in query:
                tf = document.count(term) 
                df = self.documents.df[term] 
                idf = np.log((n + 1)/(df + idf_smoothing))
                f_q = (tf * (k1 + 1))/(tf + k1 * (1 - b + b*(dl/avg_dl))) 
                bm25 += idf*f_q
            return bm25

        def query_term_coverage(self, query, document, **args): # f2
            count = 0
            for term in query:
                if term in document:
                    count += 1
            return count

        def query_term_coverage_ratio(self, query, document,**args): # f3
            qtc = sum(1 for term in query if term in document)
            qtl = len(query)
            qtc_ratio = qtc / qtl if qtl != 0 else 0
            return qtc_ratio

        def stream_length(self, query, document,**args): # f4
            return len(document)
        
        def idf(self, query, document, **args): # f5
            N = self.documents.num_docs
            idf_score = 0
            idf_smoothing = 0.5
            for term in query:
                # df = sum(1 for doc in self.documents if term in doc)
                df = self.documents.df[term]
                idf_score += np.log((N+1) / (df+idf_smoothing))
            return idf_score

        def sum_stream_length_normalized_tf(self, query, document,**args): # f6
            num_terms = len(document)
            score = 0
            for term in query:
                tf = document.count(term)
                tf_normalized = tf / num_terms if num_terms != 0 else 0
                score += tf_normalized
            return score

        def min_stream_length_normalized_tf(self, query, document:str,**args): # f7
            num_terms = len(document)
            tf_normalized = []
            for term in query:
                tf = document.count(term)
                tf_normalized.append(tf / num_terms if num_terms != 0 else 0)    
            return min(tf_normalized)

        def max_stream_length_normalized_tf(self, query, document:str,**args): # f8
            num_terms = len(document)     
            tf_normalized = []
            for term in query:
                tf = document.count(term)
                tf_normalized.append(tf / num_terms if num_terms != 0 else 0)   
            return max(tf_normalized)

        def mean_stream_length_normalized_tf(self, query, document:str,**args): # f9
            num_terms = len(document)
            score = 0
            for term in query:
                tf = document.count(term)
                tf_normalized = tf / num_terms if num_terms != 0 else 0
                score += tf_normalized     
            return score/len(query)

        def var_stream_length_normalized_tf(self, query, document,**args): # f10
            num_terms = len(document)
            score = []
            for term in query:
                tf = document.count(term)
                tf_normalized = tf / num_terms if num_terms != 0 else 0
                score.append(tf_normalized) 
            return np.var(score)


        def sum_tfidf(self, query, document,**args): # f11
            score = 0
            idf_smoothing = 0.5
            N = self.documents.num_docs
            for term in query:
                tf = document.count(term)
                df = self.documents.df[term]
                # idf = self.features['idf']
                idf = np.log((N+1) / (df+idf_smoothing))
                score += tf * idf
            return score

        def min_tfidf(self, query, document:str,**args): # f12
            score = []
            idf_smoothing = 0.5
            N = self.documents.num_docs
            for term in query:
                tf = document.count(term)
                df = self.documents.df[term]
                idf = np.log((N+1) / (df+idf_smoothing))
                score.append(tf * idf)
            return min(score)

        def max_tfidf(self, query, document:str,**args): # f13
            score = []
            idf_smoothing = 0.5
            N = self.documents.num_docs
            for term in query:
                tf = document.count(term)
                df = self.documents.df[term]
                idf = np.log((N+1) / (df+idf_smoothing))
                score.append(tf * idf)
            return max(score)

        def mean_tfidf(self, query, document:str,**args): # f14
            score = 0
            idf_smoothing = 0.5
            N = self.documents.num_docs
            for term in query:
                tf = document.count(term)
                df = self.documents.df[term]
                idf = np.log((N+1) / (df+idf_smoothing))
                score += tf * idf
            return score / len(query)     

        def var_tfidf(self, query, document:str,**args): # f15
            score = []
            idf_smoothing = 0.5
            N = self.documents.num_docs
            for term in query:
                tf = document.count(term)
                df = self.documents.df[term]
                idf = np.log((N+1) / (df+idf_smoothing))
                score.append(tf * idf)
            return np.var(score)
        
        query_from_id = self.queries.qmap[str(qid)]
        document_from_id = self.documents.index[str(docid)]

        for feature_name in __feature_list__:
            feature_func = eval(feature_name)
            feature_value = feature_func(self, query=query_from_id, document=document_from_id, **args)
            self.features[feature_name] = feature_value
        
        return self.features
        # END SOLUTION


class GenerateFeatures:
    def __init__(self, feature_extractor: FeatureExtraction) -> None:
        self.fe = feature_extractor

    def run(self, qdr_path: str, qdr_feature_path: str, **fe_args):
        with open(qdr_feature_path, "w") as qdr_feature_file:
            with open(qdr_path, "r") as qdr_file:
                for line in tqdm(qdr_file):
                    qid, docid, rel = line.strip().split("\t")
                    features = self.fe.extract(qid, docid, **fe_args)
                    feature_line = "{} qid:{} {}\n".format(
                        rel,
                        qid,
                        " ".join(
                            "" if f is None else "{}:{}".format(i, f)
                            for i, f in enumerate(features.values())
                        ),
                    )

                    qdr_feature_file.write(feature_line)


class DataSet(object):
    """
    Class designed to manage meta-data for datasets.
    """

    def __init__(
        self,
        name,
        data_paths,
        num_rel_labels,
        num_features,
        num_nonzero_feat,
        feature_normalization=True,
        purge_test_set=True,
    ):
        self.name = name
        self.num_rel_labels = num_rel_labels
        self.num_features = num_features
        self.data_paths = data_paths
        self.purge_test_set = purge_test_set
        self._num_nonzero_feat = num_nonzero_feat

    def num_folds(self):
        return len(self.data_paths)

    def get_data_folds(self):
        return [DataFold(self, i, path) for i, path in enumerate(self.data_paths)]


class DataFoldSplit(object):
    def __init__(self, datafold, name, doclist_ranges, feature_matrix, label_vector):
        self.datafold = datafold
        self.name = name
        self.doclist_ranges = doclist_ranges
        self.feature_matrix = feature_matrix
        self.label_vector = label_vector

    def num_queries(self):
        return self.doclist_ranges.shape[0] - 1

    def num_docs(self):
        return self.feature_matrix.shape[0]

    def query_range(self, query_index):
        s_i = self.doclist_ranges[query_index]
        e_i = self.doclist_ranges[query_index + 1]
        return s_i, e_i

    def query_size(self, query_index):
        s_i = self.doclist_ranges[query_index]
        e_i = self.doclist_ranges[query_index + 1]
        return e_i - s_i

    def query_sizes(self):
        return self.doclist_ranges[1:] - self.doclist_ranges[:-1]

    def query_labels(self, query_index):
        s_i = self.doclist_ranges[query_index]
        e_i = self.doclist_ranges[query_index + 1]
        return self.label_vector[s_i:e_i]

    def query_feat(self, query_index):
        s_i = self.doclist_ranges[query_index]
        e_i = self.doclist_ranges[query_index + 1]
        return self.feature_matrix[s_i:e_i, :]

    def doc_feat(self, query_index, doc_index):
        s_i = self.doclist_ranges[query_index]
        e_i = self.doclist_ranges[query_index + 1]
        assert s_i + doc_index < self.doclist_ranges[query_index + 1]
        return self.feature_matrix[s_i + doc_index, :]

    def doc_str(self, query_index, doc_index):
        doc_feat = self.doc_feat(query_index, doc_index)
        feat_i = np.where(doc_feat)[0]
        doc_str = ""
        for f_i in feat_i:
            doc_str += "%f " % (doc_feat[f_i])
        return doc_str

    def subsample_by_ids(self, qids):
        feature_matrix = []
        label_vector = []
        doclist_ranges = [0]
        for qid in qids:
            feature_matrix.append(self.query_feat(qid))
            label_vector.append(self.query_labels(qid))
            doclist_ranges.append(self.query_size(qid))

        doclist_ranges = np.cumsum(np.array(doclist_ranges), axis=0)
        feature_matrix = np.concatenate(feature_matrix, axis=0)
        label_vector = np.concatenate(label_vector, axis=0)
        return doclist_ranges, feature_matrix, label_vector

    def random_subsample(self, subsample_size):
        if subsample_size > self.num_queries():
            return DataFoldSplit(
                self.datafold,
                self.name + "_*",
                self.doclist_ranges,
                self.feature_matrix,
                self.label_vector,
                self.data_raw_path,
            )
        qids = np.random.randint(0, self.num_queries(), subsample_size)

        doclist_ranges, feature_matrix, label_vector = self.subsample_by_ids(qids)

        return DataFoldSplit(
            None, self.name + str(qids), doclist_ranges, feature_matrix, label_vector
        )


class DataFold(object):
    def __init__(self, dataset, fold_num, data_path):
        self.name = dataset.name
        self.num_rel_labels = dataset.num_rel_labels
        self.num_features = dataset.num_features
        self.fold_num = fold_num
        self.data_path = data_path
        self._data_ready = False
        self._num_nonzero_feat = dataset._num_nonzero_feat

    def data_ready(self):
        return self._data_ready

    def clean_data(self):
        del self.train
        del self.validation
        del self.test
        self._data_ready = False
        gc.collect()

    def read_data(self):
        """
        Reads data from a fold folder (letor format).
        """

        output = load_data_in_libsvm_format(
            self.data_path + "train_pairs_graded.tsvg", feature_size=self.num_features
        )
        train_feature_matrix, train_label_vector, train_doclist_ranges = (
            output["fm"],
            output["lv"],
            output["dlr"],
        )

        output = load_data_in_libsvm_format(
            self.data_path + "dev_pairs_graded.tsvg", feature_size=self.num_features
        )
        valid_feature_matrix, valid_label_vector, valid_doclist_ranges = (
            output["fm"],
            output["lv"],
            output["dlr"],
        )

        output = load_data_in_libsvm_format(
            self.data_path + "test_pairs_graded.tsvg", feature_size=self.num_features
        )
        test_feature_matrix, test_label_vector, test_doclist_ranges = (
            output["fm"],
            output["lv"],
            output["dlr"],
        )

        self.train = DataFoldSplit(
            self,
            "train",
            train_doclist_ranges,
            train_feature_matrix,
            train_label_vector,
        )
        self.validation = DataFoldSplit(
            self,
            "validation",
            valid_doclist_ranges,
            valid_feature_matrix,
            valid_label_vector,
        )
        self.test = DataFoldSplit(
            self, "test", test_doclist_ranges, test_feature_matrix, test_label_vector
        )
        self._data_ready = True


# this is a useful class to create torch DataLoaders, and can be used during training
class LTRData(Dataset):
    def __init__(self, data, split):
        split = {
            "train": data.train,
            "validation": data.validation,
            "test": data.test,
        }.get(split)
        assert split is not None, "Invalid split!"
        features, labels = split.feature_matrix, split.label_vector
        self.doclist_ranges = split.doclist_ranges
        self.num_queries = split.num_queries()
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return self.features.size(0)

    def __getitem__(self, i):
        return self.features[i], self.labels[i]


class QueryGroupedLTRData(Dataset):
    def __init__(self, data, split):
        self.split = {
            "train": data.train,
            "validation": data.validation,
            "test": data.test,
        }.get(split)
        assert self.split is not None, "Invalid split!"

    def __len__(self):
        return self.split.num_queries()

    def __getitem__(self, q_i):
        feature = torch.FloatTensor(self.split.query_feat(q_i))
        labels = torch.FloatTensor(self.split.query_labels(q_i))
        return feature, labels


# the return types are different from what pytorch expects,
# so we will define a custom collate function which takes in
# a batch and returns tensors (qids, features, labels)
def qg_collate_fn(batch):

    # qids = []
    features = []
    labels = []

    for (f, l) in batch:
        # qids.append(1)
        features.append(f)
        labels.append(l)

    return features, labels
