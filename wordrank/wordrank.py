# coding: utf-8

import re
import csv
import math
import codecs
from collections import defaultdict

import networkx as nx
from tqdm import tqdm

MAX_LEN = 4

"""
《A Simple and Effective Unsupervised Word Segmentation Approach》
"""


class Docs(object):
    def __init__(self, file_path):
        self.file_path = file_path

    def __iter__(self):
        with codecs.open(self.file_path, mode='r', encoding='utf-8') as f:
            for line in f:
                yield line

    def __len__(self):
        s = 0
        with codecs.open(self.file_path, mode='r', encoding='utf-8') as f:
            for _ in f:
                s += 1
        return s


def extract_zh(txt):
    """
    抽取中文字符
    :param txt:
    :return:
    """
    return ' '.join(re.findall(r'[\u4e00-\u9fa5]+', txt))


def read_docs(docs):
    """
    遍历文本数据
    :param docs:
    """
    for line in tqdm(docs, total=len(docs)):
        line = line.strip()

        if line:
            line = extract_zh(line)
            for sline in line.split():
                if sline:
                    yield sline


def build_grams(docs, max_len):
    """
    生成ngrams
    :param docs:
    :param max_len:
    :return:
    """
    grams = defaultdict(int)
    for txt in read_docs(docs):
        for i, c in enumerate(txt):
            for j in range(1, max_len + 1):
                grams[txt[i:i + j]] += 1
                if i + j > len(txt) - 1:
                    break
    return grams


def prune_freq(grams, min_freq=2):
    """
    剪枝，过滤掉词频为1的候选词
    :param grams:
    :param min_freq:
    :return:
    """
    return {k: v for k, v in grams.items() if v >= min_freq}


def prune_overlap(grams, d_freq=1):
    """
    剪枝，使用SSR算法过滤掉短的重叠词

    频率相同的字符序列A,B，如果A是B的子序列，那么A是冗余的，需要剔除掉
    这里的实现不是相同，而是频率差值小于d_freq
    参考：
        《Statistical Substring Reduction in Linear Time》
        https://github.com/jimregan/ngramtool
    :param grams:
    :param d_freq: 差值
    :return:
    """

    words = [_ for _ in grams.keys()]

    words = sorted(words)

    for i in range(len(words) - 1):
        word1 = words[i]
        word2 = words[i + 1]
        f1 = grams[word1]
        f2 = grams[word2]

        # X[i]的词频与X[i+1]的词频差值小于阈值，并且X[i]是X[i+1]的子序列
        # 那么X[i]是冗余的
        # 对冗余的X[i]，设置它的词频为负数，作为标记
        if f1 - f2 <= d_freq and word1 in word2:
            grams[word1] = -1 * abs(grams[word1])

    # 反转字符序列
    for i, w in enumerate(words):
        words[i] = ''.join(reversed(w))

    words = sorted(words)

    for i in range(len(words) - 1):
        rword1 = words[i]
        rword2 = words[i + 1]
        word1 = ''.join(reversed(rword1))
        word2 = ''.join(reversed(rword2))

        f1 = grams[word1]
        f2 = grams[word2]

        if f1 - f2 <= d_freq and rword1 in rword2:
            grams[word1] = -1 * abs(grams[word1])

    # step 6
    for i, w in enumerate(words):
        words[i] = ''.join(reversed(w))

    return {k: v for k, v in grams.items() if v > 0}


def build_graph(candidates, docs, max_len):
    """
    使用候选词构建graph
    :param candidates:
    :param docs:
    :param max_len:
    :return:
    """
    G = nx.DiGraph()
    # G.add_nodes_from(grams)

    pre_words = None
    curr_words = None
    for next_txt in read_docs(docs):

        next_words = reverse_max_match(next_txt, candidates, max_len)

        if pre_words and curr_words and next_words:
            words = [pre_words[-1]] + curr_words + [next_words[0]]
            for i in range(1, len(words)-1):
                word = words[i]
                left_word = words[i-1]
                right_word = words[i+1]

                G.add_edge(left_word, word)
                # G.add_edge(word, right_word)

        pre_words = curr_words
        curr_words = next_words

    for node in G.nodes:
        G.nodes[node]['lbv'] = 1
        G.nodes[node]['rbv'] = 1

    return G


def reverse_max_match(text, dictionary, max_len):
    """
    分词，逆向最大匹配算法
    :param text: 待分词的文本
    :param dictionary: 词典
    :param max_len: 词的最大字数
    :rtype: list
    :return:
    """
    def get_word(txt):
        if not txt:
            return ''

        if len(txt) == 1:
            return txt

        if txt in dictionary:
            return txt
        else:
            return get_word(txt[-len(txt)+1:])

    result = []

    while text:
        word = get_word(text[-max_len:])
        result.insert(0, word)
        text = text[:-len(word):]

    return result


def cal_ebv(graph, epochs=30):
    """
    计算外部边界值
    :param graph:
    :type graph: nx.DiGraph
    :param epochs:
    :type epochs: int
    """
    for epoch in range(epochs):
        print('epoch [%s]...' % (epoch+1))

        for node in graph.nodes:
            graph.node[node]['lbv'] = sum([graph.nodes[lnode]['rbv'] for lnode in graph.predecessors(node)])

        for node in graph.nodes:
            graph.node[node]['rbv'] = sum([graph.nodes[rnode]['lbv'] for rnode in graph.successors(node)])

        lbv_norm = math.sqrt(sum(graph.node[node]['lbv'] ** 2 for node in graph.nodes))
        rbv_norm = math.sqrt(sum(graph.node[node]['rbv'] ** 2 for node in graph.nodes))

        for node in graph.nodes:
            graph.node[node]['lbv'] /= lbv_norm
            graph.node[node]['rbv'] /= rbv_norm
            graph.node[node]['ebv'] = graph.node[node]['lbv'] * graph.node[node]['rbv']


def cal_ibv(graph, grams):
    """
    计算内部边界值
    :param graph:
    :param grams:
    """
    total = sum([grams[w] for w in grams if len(w) == 1])
    print('total chars:', total)

    for node in graph.nodes:
        word = node

        if len(word) < 2:
            continue

        pmis = []
        for i in range(1, len(word)):
            word1 = word[:i]
            word2 = word[i:]

            pmi = math.log2(grams[word] * total / (grams[word1] * grams[word2]))
            pmis.append(pmi)

        graph.node[node]['ibv'] = min(pmis)


def cal_rank(graph, f='exp', alpha=3.4):
    """
    计算word rank分数，合并ebv/ibv
    f<poly> = pow(x, α)
    f<exp> = pow(α, x)
    :param graph:
    :param f: exp/poly
    :param alpha: f=exp时推荐3.4
    :type graph: nx.DiGraph
    :return:
    """
    if f == 'exp':
        def f_fun(x):
            return math.pow(alpha, x)
    elif f == 'poly':
        def f_fun(x):
            return math.pow(x, alpha)
    else:
        def f_fun(x):
            return x

    for word in graph.nodes:
        if len(word) > 1:
            graph.node[word]['rank'] = graph.node[word]['ebv'] * f_fun(graph.node[word]['ibv'])


def build(file_path, result_file_path):
    docs = Docs(file_path)

    ngrams = build_grams(docs, MAX_LEN)

    candidates = prune_freq(ngrams)

    candidates = prune_overlap(candidates)

    graph = build_graph(candidates, docs, MAX_LEN)

    cal_ebv(graph)

    cal_ibv(graph, ngrams)

    cal_rank(graph, alpha=5)

    word_ranks = [{'word': word, 'rank': graph.node[word]['rank']} for word in graph.nodes if len(word) > 1]
    word_ranks = sorted(word_ranks, key=lambda wr: wr['rank'], reverse=True)

    with codecs.open(result_file_path, mode='w', encoding='utf-8') as f:
        headers = ['word', 'rank']
        writer = csv.DictWriter(f, headers)
        writer.writeheader()
        writer.writerows(word_ranks)


if __name__ == '__main__':
    build('data.txt', 'words.csv')
