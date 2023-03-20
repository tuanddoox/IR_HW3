import os, pickle
import torch
from ltr.loss import pointwise_loss, pairwise_loss, compute_lambda_i, listwise_loss
from ltr.dataset import Documents, FeatureExtraction, Queries, Preprocess
from ltr.model import LTRModel
from ltr.dataset import __feature_list__
import pytest


@pytest.fixture
def get_queries():
    query_path = "./data/queries.tsv"
    STOP_WORDS_PATH = "./data/common_words"
    preprocesser = Preprocess(STOP_WORDS_PATH)
    queries = Queries(preprocesser)
    queries.preprocess_queries(query_path)
    return queries


@pytest.fixture
def get_documents():
    file = "./datasets/doc.pickle"
    documents: Documents = pickle.load(open(file, "rb"))
    return documents


def test_preprocessing_queries():
    DOC_JSON = "./datasets/doc.pickle"
    assert os.path.exists(DOC_JSON)
    with open(DOC_JSON, "rb") as file:
        documents: Documents = pickle.load(file)
    assert documents.df.keys() is not None
    assert documents.dl.keys() is not None
    assert documents.index.keys() is not None


def test_feature_extraction(get_documents, get_queries):
    line = "1028080	3038588	2"
    # who guards kevin durant

    qid, docid, rel = line.strip().split("\t")
    feature_ex = FeatureExtraction({}, get_documents, get_queries)
    args = {}
    args["k1"] = 1.5
    args["b"] = 0.75
    args["idf_smoothing"] = 0.5
    ft = feature_ex.extract(qid, docid, **args)
    ft_keys = []
    for k, v in ft.items():
        assert v >= 0
        ft_keys.append(k)

    for f in __feature_list__:
        assert f in ft_keys

    assert len(ft_keys) == len(__feature_list__)


def test_neural_module():
    temp_nn = LTRModel(len(__feature_list__))
    assert temp_nn.layers.out.in_features == 10


def test_pointwise_loss():
    g = torch.manual_seed(42)
    output = [
        torch.randint(low=0, high=5, size=(5, 1), generator=g).float() for _ in range(5)
    ]
    target = torch.randint(low=0, high=5, size=(5,), generator=g).float()

    l = [pointwise_loss(o, target).item() for o in output]
    assert len(l) == 5


def test_pairwise_loss():
    temp_score = torch.FloatTensor([0.1, 0.2, 0.3]).unsqueeze(1)
    temp_label = torch.FloatTensor([0, 1, 2])

    loss = pairwise_loss(temp_score, temp_label)
    assert loss > 0


def test_compute_lambda():
    temp_score = torch.FloatTensor([0.1, 0.2, 0.3]).unsqueeze(1)
    temp_label = torch.FloatTensor([0, 1, 2])

    loss = compute_lambda_i(temp_score, temp_label)
    assert loss.shape[0] == 3
    assert loss.shape[1] == 1


def test_listwise_loss():
    temp_score = torch.FloatTensor([0.1, 0.2, 0.3]).unsqueeze(1)
    temp_label = torch.FloatTensor([0, 1, 2])

    loss = listwise_loss(temp_score, temp_label)
    assert loss.shape[0] == 3
    assert loss.shape[1] == 1
