# !/user/bin/env python3
# -*- coding: utf-8 -*-
import os

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.core import ServiceContext, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.indices.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.indices.postprocessor import SentenceTransformerRerank
from llama_index.core import load_index_from_storage
from llama_index.core import Document
# 加载 .env 到环境变量
from dotenv import load_dotenv, find_dotenv



##定义创建向量数据库函数
def build_sentence_window_index(
        documents,
        llm,
        embed_model="local:BAAI/bge-small-zh-v1.5",
        sentence_window_size=3,
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
if __name__ == '__main__':
    _ = load_dotenv(find_dotenv())
    api_key = 'you_key'
    api_base = 'you_url'
    documents = SimpleDirectoryReader(input_files=["../data/product.pdf"]).load_data()
    # 将文本整合在一起以便于提升整体性
    document = Document(text='\n\n'.join([doc.text for doc in documents]))
    llm = OpenAI(model='gpt-3.5-turbo', temperature=0,api_key=api_key,api_base=api_base)
    sentence_index = build_sentence_window_index(
        documents,
        llm,
        embed_model='local:BAAI/bge-small-en-v1.5',
        save_dir='../output/sentence_index_product'
    )
    query_engine = sentence_index.as_query_engine()
    response = query_engine.query("文章写了什么")
    print(response)