import os
import json
from transformers import PreTrainedTokenizer
from typing import Iterable, Optional


class DataBase:
    def __init__(self,
                 path: str,
                 tokenizer: Optional[PreTrainedTokenizer] = None,
                 max_passage_length: int = 512,
                 use_cache: bool = True):

        self.path = path
        self.use_cache = use_cache
        self.has_tokenizer = tokenizer is not None

        with open(os.path.join(path, "index.json"), 'r', encoding='utf-8') as fp:
            self.index = json.load(fp)

        print(f"Loaded database from {path}")
        print(f"Total {len(self.index)} items")

        if tokenizer:
            self.tokenizer = tokenizer
            self.max_passage_length = max_passage_length

        if use_cache:
            self.cache = {}

    def read_article(self, position):
        assert len(position) == 4
        path = [self.path] + position[:3]
        with open(os.path.join(*path), "r", encoding='utf-8') as fp:
            line = fp.readlines()[position[3]]
        line = json.loads(line)
        text = line['text']
        return text

    def __inner_get_passages(self, keyword):
        articles = []
        if keyword in self.index:
            positions = self.index[keyword]

            for position in positions:
                article = self.read_article(position)
                articles.append(article)

        return articles

    def get_passages_with_keyword(self, keyword):
        assert isinstance(keyword, str)

        if self.use_cache and keyword in self.cache:
            return self.cache[keyword]

        articles = self.__inner_get_passages(keyword)

        # trim over-length articles
        if self.has_tokenizer:
            articles_token_ids = self.tokenizer.convert_tokens_to_ids(articles)
            articles_token_ids = [ele[:self.max_passage_length] for ele in articles_token_ids]
            articles = self.tokenizer.convert_ids_to_tokens(articles_token_ids)

        if self.use_cache:
            self.cache[keyword] = articles

        return articles

    def get_passages_with_keywords(self, keywords):
        assert isinstance(keywords, Iterable)

        articles = []
        for keyword in keywords:
            articles.extend(self.get_passages_with_keyword(keyword))

        return articles

    def get_passages_with_sentence(self, sentence):
        # todo implement
        raise NotImplementedError


if __name__ == '__main__':
    database = DataBase("./data/database/", tokenizer=None, use_cache=True)
    passages = database.get_passages_with_keywords(['alpha decay', 'augustus'])
    print(passages)
