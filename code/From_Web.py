# !/user/bin/env python3
# -*- coding: utf-8 -*-
import os
from llama_index.core import ServiceContext, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.indices.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.indices.postprocessor import SentenceTransformerRerank
from llama_index.core import load_index_from_storage
from llama_index.readers.web import TrafilaturaWebReader
from llama_index.llms.openai import OpenAI

api_key = 'you key'
api_base = 'you url'
#清洗网上获取的数据
def clear_data_from_web(url):
    docs = TrafilaturaWebReader().load_data([url])

    # 将全角标点符号转换成半角标点符号+空格
    for d in docs:
        d.text = d.text.replace('。', '. ')
        d.text = d.text.replace('！', '! ')
        d.text = d.text.replace('？', '? ')

    print("docs",docs)
    return docs


# 定义创建向量数据库函数
def build_sentence_window_index(
        documents,
        llm,
        embed_model="local:BAAI/bge-small-zh-v1.5",
        sentence_window_size=1,
        save_dir="sentence_index",
):
    # create the sentence window node parser w/ default settings
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=sentence_window_size,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    sentence_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        node_parser=node_parser,
    )
    if not os.path.exists(save_dir):
        sentence_index = VectorStoreIndex.from_documents(
            documents, service_context=sentence_context
        )
        sentence_index.storage_context.persist(persist_dir=save_dir)
    else:
        sentence_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            service_context=sentence_context,
        )

    return sentence_index


# 定义查询引擎函数
def get_sentence_window_query_engine(
        sentence_index, similarity_top_k=6, rerank_top_n=2
):
    # define postprocessors
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )

    sentence_window_engine = sentence_index.as_query_engine(
        similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank]
    )
    return sentence_window_engine
if __name__ == '__main__':
    url = 'https://baike.baidu.com/item/%E6%A3%80%E7%B4%A2%E5%A2%9E%E5%BC%BA%E7%94%9F%E6%88%90/64380539'
    docs = clear_data_from_web(url)
    index = build_sentence_window_index(
        docs,
        llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1, api_base=api_base,api_key = api_key),
        save_dir="../output/sentence_index_web",
    )
    # 创业查询引擎
    query_engine = get_sentence_window_query_engine(index, similarity_top_k=6)
    query = 'rag的工作原理'
    response = query_engine.query(query)
    print(response)

