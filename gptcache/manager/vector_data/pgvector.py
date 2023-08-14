from typing import List

import numpy as np

from gptcache.manager.vector_data.base import VectorBase, VectorData
from gptcache.utils import import_sqlalchemy

import_sqlalchemy()

# pylint: disable=wrong-import-position
from sqlalchemy import create_engine, Column, Index, text  # pylint: disable=C0413
from sqlalchemy.types import (  # pylint: disable=C0413
    Integer,
    UserDefinedType
)
from sqlalchemy.orm import sessionmaker  # pylint: disable=C0413
from sqlalchemy.ext.declarative import declarative_base  # pylint: disable=C0413

Base = declarative_base()


class _VectorType(UserDefinedType):
    """
    pgvector type mapping for sqlalchemy
    """
    cache_ok = True

    def __init__(self, precision=8):
        self.precision = precision

    def get_col_spec(self, **_):
        return f"vector({self.precision})"

    # pylint: disable=unused-argument
    def bind_processor(self, dialect):
        return lambda value: value

    # pylint: disable=unused-argument
    def result_processor(self, dialect, coltype):
        return lambda value: value


def _get_model_and_index(table_prefix, vector_dimension, index_type, lists):
    class VectorStoreTable(Base):
        """
        vector store table
        """

        __tablename__ = table_prefix + "_pg_vector_store"
        __table_args__ = {"extend_existing": True}
        id = Column(Integer, primary_key=True, autoincrement=False)
        embedding = Column(_VectorType(vector_dimension), nullable=False)

    vector_store_index = Index(
        f"idx_{table_prefix}_pg_vector_store_embedding",
        text(f"embedding {index_type}"),
        postgresql_using="ivfflat",
        postgresql_with={"lists": lists}
    )

    vector_store_index.table = VectorStoreTable.__table__

    return VectorStoreTable, vector_store_index


class PGVector(VectorBase):
    """vector store: pgvector

    :param url: the connection url for PostgreSQL database, defaults to 'postgresql://postgres@localhost:5432/postgres'.
    :type url: str
    :type collection_name: str
    :param dimension: the dimension of the vector, defaults to 0.
    :type dimension: int
    :param top_k: the number of the vectors results to return, defaults to 1.
    :type top_k: int
    :param index_params: the index parameters for pgvector, defaults to 'vector_l2_ops' index:
                         {"index_type": "L2", "params": {"lists": 100, "probes": 10}.
    :type index_params: dict
    """

    INDEX_PARAM = {
        "L2": {"operator": "<->", "name": "vector_l2_ops"},  # The only one supported now
        "cosine": {"operator": "<=>", "name": "vector_cosine_ops"},
        "inner_product": {"operator": "<->", "name": "vector_ip_ops"},
    }

    def __init__(
            self,
            url: str,
            index_params: dict,
            collection_name: str = "gptcache",
            dimension: int = 0,
            top_k: int = 1,
    ):
        if dimension <= 0:
            raise ValueError(
                f"invalid `dim` param: {dimension} in the pgvector store."
            )
        self.dimension = dimension
        self.top_k = top_k
        self.index_params = index_params
        self._url = url
        self._store, self._index = _get_model_and_index(
            collection_name,
            dimension,
            index_type=self.INDEX_PARAM[index_params["index_type"]]["name"],
            lists=index_params["params"]["lists"]
        )
        self._connect(url)
        self._create_collection()

    def _connect(self, url):
        self._engine = create_engine(url, echo=False)
        self._session = sessionmaker(bind=self._engine)  # pylint: disable=invalid-name

    def _create_collection(self):
        with self._engine.connect() as con:
            con.execution_options(isolation_level="AUTOCOMMIT").execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))

        self._store.__table__.create(bind=self._engine, checkfirst=True)
        self._index.create(bind=self._engine, checkfirst=True)

    def _query(self, session):
        return session.query(self._store)

    def _format_data_for_search(self, data):
        return f"[{','.join(map(str, data))}]"

    def mul_add(self, datas: List[VectorData]):
        data_array, id_array = map(list, zip(*((data.data, data.id) for data in datas)))
        np_data = np.array(data_array).astype("float32")
        entities = [{"id": id, "embedding": embedding.tolist()} for id, embedding in zip(id_array, np_data)]

        with self._session() as session:
            session.bulk_insert_mappings(self._store, entities)
            session.commit()

    def search(self, data: np.ndarray, top_k: int = -1):
        if top_k == -1:
            top_k = self.top_k

        formatted_data = self._format_data_for_search(data.reshape(1, -1)[0].tolist())
        index_config = self.INDEX_PARAM[self.index_params["index_type"]]
        similarity = self._store.embedding.op(index_config["operator"])(formatted_data)
        with self._session() as session:
            session.execute(text(f"SET LOCAL ivfflat.probes = {self.index_params['params']['probes'] or 10};"))
            search_result = self._query(session).add_columns(
                similarity.label("distances")
            ).order_by(
                similarity
            ).limit(top_k).all()
            search_result = [(r[1], r[0].id) for r in search_result]

        return search_result

    def delete(self, ids):
        with self._session() as session:
            self._query(session).filter(self._store.id.in_(ids)).delete()
            session.commit()

    def rebuild(self, ids=None):  # pylint: disable=unused-argument
        with self._engine.connect() as con:
            con.execution_options(isolation_level="AUTOCOMMIT").execute(
                text(f"REINDEX INDEX CONCURRENTLY {self._index.name}"))

    def flush(self):
        with self._session() as session:
            session.flush()

    def close(self):
        self.flush()
