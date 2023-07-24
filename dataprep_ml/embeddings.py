from functools import lru_cache
from typing import List, Union

import pandas as pd
import torch
from langchain.document_loaders import DataFrameLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from dataprep_ml.helpers import log

class DfLoader(DataFrameLoader):

    """
    override the load method of langchain.document_loaders.DataFrameLoaders to ignore rows with 'None' values
    """

    def __init__(self, data_frame: pd.DataFrame, page_content_column: str):
        super().__init__(data_frame=data_frame, page_content_column=page_content_column)
        self._data_frame = data_frame
        self._page_content_column = page_content_column

    def load(self) -> List[Document]:
        """Loads the dataframe as a list of documents"""
        documents = []
        for n_row, frame in self._data_frame[self._page_content_column].iteritems():
            if pd.notnull(frame):
                # ignore rows with None values
                column_name = self._page_content_column

                document_contents = frame

                documents.append(
                    Document(
                        page_content=document_contents,
                        metadata={
                            "source": "dataframe",
                            "row": n_row,
                            "column": column_name,
                        },
                    )
                )
        return documents


def df_to_documents(
    df: pd.DataFrame, page_content_columns: Union[List[str], str]
) -> List[Document]:
    """Converts a given dataframe to a list of documents"""
    documents = []

    if isinstance(page_content_columns, str):
        page_content_columns = [page_content_columns]

    for _, page_content_column in enumerate(page_content_columns):
        if page_content_column not in df.columns.tolist():
            raise ValueError(
                f"page_content_column {page_content_column} not in dataframe columns"
            )

        loader = DfLoader(data_frame=df, page_content_column=page_content_column)
        documents.extend(loader.load())

    return documents


def split_documents(df, columns):
    # Load documents and split in chunks
    log.info(f"Loading documents from input data")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = df_to_documents(df=df, page_content_columns=columns)
    texts = text_splitter.split_documents(documents)
    log.info(f"Loaded {len(documents)} documents from input data")
    log.info(f"Split into {len(texts)} chunks of text (max. 500 tokens each)")

    return texts


@lru_cache()
def load_embeddings_model(embeddings_model_name):
    try:
        model_kwargs = {"device": "gpu" if torch.cuda.is_available() else "cpu"}
        embedding_model = HuggingFaceEmbeddings(
            model_name=embeddings_model_name, model_kwargs=model_kwargs
        )
    except ValueError:
        raise ValueError(
            f"The {embeddings_model_name}  is not supported, please select a valid option from Hugging Face Hub!"
        )
    return embedding_model
