from typing import List

from datetime import datetime

from gptcache.utils import import_sqlalchemy
from gptcache.manager.scalar_data.base import CacheStorage, CacheData

import_sqlalchemy()

from sqlalchemy import func, create_engine, Column, Sequence  # pylint: disable=C0413
from sqlalchemy.types import (  # pylint: disable=C0413
    String,
    DateTime,
    LargeBinary,
    Integer,
)
from sqlalchemy.orm import sessionmaker  # pylint: disable=C0413
from sqlalchemy.ext.declarative import declarative_base  # pylint: disable=C0413

Base = declarative_base()


def get_model(table_name, db_type):
    class CacheTable(Base):
        """
        cache_table
        """

        __tablename__ = table_name
        __table_args__ = {"extend_existing": True}

        id = Column(Integer, primary_key=True, autoincrement=True)
        question = Column(String(1000), nullable=False)
        answer = Column(String(1000), nullable=False)
        create_on = Column(DateTime, default=datetime.now)
        last_access = Column(DateTime, default=datetime.now)
        embedding_data = Column(LargeBinary, nullable=True)
        state = Column(Integer, default=0)
        type = Column(Integer, default=0)

    class CacheTableSequence(Base):
        """
        cache_table sequence
        """

        __tablename__ = table_name
        __table_args__ = {"extend_existing": True}

        id = Column(
            Integer, Sequence("id_seq", start=1), primary_key=True, autoincrement=True
        )
        question = Column(String(1000), nullable=False)
        answer = Column(String(1000), nullable=False)
        create_on = Column(DateTime, default=datetime.now)
        last_access = Column(DateTime, default=datetime.now)
        embedding_data = Column(LargeBinary, nullable=True)
        state = Column(Integer, default=0)
        type = Column(Integer, default=0)

    if db_type == "oracle":
        return CacheTableSequence
    else:
        return CacheTable


class SQLDataBase(CacheStorage):
    """
    Using sqlalchemy to manage SQLite, PostgreSQL, MySQL, MariaDB, SQL Server and Oracle.

    :param name: the name of the cache storage, it is support 'sqlite', 'postgresql', 'mysql', 'mariadb', 'sqlserver' and  'oracle' now.
    :type name: str
    :param sql_url: the url of the sql database for cache, such as '<db_type>+<db_driver>://<username>:<password>@<host>:<port>/<database>',
                    and the default value is related to the `cache_store` parameter, 'sqlite:///./sqlite.db' for 'sqlite',
                    'postgresql+psycopg2://postgres:123456@127.0.0.1:5432/postgres' for 'postgresql',
                    'mysql+pymysql://root:123456@127.0.0.1:3306/mysql' for 'mysql',
                    'mariadb+pymysql://root:123456@127.0.0.1:3307/mysql' for 'mariadb',
                    'mssql+pyodbc://sa:Strongpsw_123@127.0.0.1:1434/msdb?driver=ODBC+Driver+17+for+SQL+Server' for 'sqlserver',
                    'oracle+cx_oracle://oracle:123456@127.0.0.1:1521/?service_name=helowin&encoding=UTF-8&nencoding=UTF-8' for 'oracle'.
    :type sql_url: str
    :param table_name: the table name for sql database, defaults to 'gptcache'.
    :type table_name: str
    """

    def __init__(
        self,
        db_type: str = "sqlite",
        url: str = "sqlite:///./sqlite.db",
        table_name: str = "gptcache",
    ):
        self._url = url
        self._model = get_model(table_name, db_type)
        self._engine = create_engine(self._url)
        Session = sessionmaker(bind=self._engine)  # pylint: disable=invalid-name
        self._session = Session()
        self.create()

    def create(self):
        self._model.__table__.create(bind=self._engine, checkfirst=True)

    def batch_insert(self, datas: List[CacheData]):
        model_objs = []
        for data in datas:
            model_obj = self._model(
                question=data.question,
                answer=data.answer,
                embedding_data=data.embedding_data.tobytes()
                if data.embedding_data is not None
                else None,
            )
            model_objs.append(model_obj)

        self._session.add_all(model_objs)
        self._session.commit()
        return [model_obj.id for model_obj in model_objs]

    def get_data_by_id(self, key):
        res = (
            self._session.query(self._model.question, self._model.answer, self._model.embedding_data)
            .filter(self._model.id == key)
            .filter(self._model.state == 0)
            .first()
        )
        return res

    def get_ids_by_state(self, state):
        res = (
            self._session.query(self._model.id).filter(self._model.state == state).all()
        )
        return res

    def get_embedding_data(self, offset, size):
        res = (
            self._session.query(self._model.id, self._model.embedding_data)
            .order_by(self._model.id.asc())
            .limit(size)
            .offset(offset)
            .all()
        )
        return res

    def update_access_time(self, key):
        self._session.query(self._model).filter(self._model.id == key).update(
            {"last_access": datetime.now()}
        )
        self._session.commit()

    def get_old_access(self, count):
        res = (
            self._session.query(self._model.id)
            .order_by(self._model.last_access.asc())
            .filter(self._model.state == 0)
            .limit(count)
            .all()
        )
        return res

    def get_old_create(self, count):
        res = (
            self._session.query(self._model.id)
            .order_by(self._model.create_on.asc())
            .filter(self._model.state == 0)
            .limit(count)
            .all()
        )
        return res

    def update_state(self, keys):
        self._session.query(self._model).filter(self._model.id.in_(keys)).update(
            {"state": -1}
        )
        self._session.commit()

    def remove_by_state(self):
        self._session.query(self._model).filter(self._model.state == -1).delete()
        self._session.commit()

    def count(self, state: int = 0, is_all: bool = False):
        if is_all:
            return self._session.query(func.count(self._model.id)).scalar()
        return (
            self._session.query(func.count(self._model.id))
            .filter(self._model.state == state)
            .scalar()
        )

    def close(self):
        self._session.close()
