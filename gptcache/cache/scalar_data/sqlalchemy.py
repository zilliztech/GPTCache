from gptcache.util import import_sqlalchemy
import_sqlalchemy()

import numpy as np
from datetime import datetime
from sqlalchemy import func, create_engine, Column
from sqlalchemy.types import String, DateTime, VARBINARY, Integer
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

from .base import ScalarStorage, TABLE_NAME

Base = declarative_base()


class CacheTable(Base):
    """
    cache_table
    """
    __tablename__ = TABLE_NAME

    uid = Column(Integer, primary_key=True, index=True, autoincrement=True)
    id = Column(String, nullable=False)
    data = Column(String, nullable=False)
    reply = Column(String, nullable=False)
    create_on = Column(DateTime, default=datetime.now)
    last_access = Column(DateTime, default=datetime.now)
    embedding_data = Column(VARBINARY, nullable=True)
    state = Column(Integer, default=0)


class SQLDataBase(ScalarStorage):
    url_list = {
        'sqlite': 'sqlite:///./gpt_cache.db',
        'postgresql': 'postgresql+psycopg2://user:password@hostname:port/database_name',
        'mysql': 'mysql+pymysql://user:password@hostname:port/database_name',
        'mariadb': 'mariadb+pymysql://user:password@hostname:port/database_name',
        'sqlserver': 'mssql+pyodbc://user:password@database_name',
        'oracle': 'oracle+zxjdbc://user:password@hostname:port/database_name',
    }
    """
    Using sqlalchemy to manage SQLite, PostgreSQL, MySQL, MariaDB, SQL Server and Oracle.
    """
    def __init__(self, url: str = 'sqlite:///./gpt_cache.db'):
        self._url = url
        self._engine = None
        self._session = None
        self._model = CacheTable
        self.init()

    def init(self):
        self._engine = create_engine(self._url)
        Session = sessionmaker(bind=self._engine)
        self._session = Session()
        self.create()

    def create(self):
        self._model.__table__.create(bind=self._engine, checkfirst=True)

    def insert(self, key, data, reply, embedding_data: np.ndarray = None):
        model_obj = self._model(id=key, data=data, reply=reply, embedding_data=embedding_data)
        self._session.add(model_obj)
        self._session.commit()

    def get_data_by_ids(self, keys):
        res = self._session.query(self._model.data, self._model.reply).filter(self._model.id.in_(keys)).filter(self._model.state == 0).all()
        return res

    def get_data_by_id(self, key):
        res = self._session.query(self._model.data, self._model.reply).filter(self._model.id == key).filter(self._model.state == 0).first()
        return res

    def get_ids_by_state(self, state):
        res = self._session.query(self._model.id).filter(self._model.state == state).all()
        return res

    def get_embedding_data(self, offset, size):
        res = self._session.query(self._model.embedding_data).order_by(self._model.uid.asc()).limit(size).offset(offset).all()
        return res

    def update_access_time(self, key):
        self._session.query(self._model).filter(self._model.id == key).update({'last_access': datetime.now()})
        self._session.commit()

    def get_old_access(self, count):
        res = self._session.query(self._model.id).order_by(self._model.last_access.asc()).limit(count).all()
        return res

    def update_state(self, keys, state: int = -1):
        self._session.query(self._model).filter(self._model.id.in_(keys)).update({'state': state})
        self._session.commit()

    def remove_by_state(self):
        res = self._session.query(self._model).filter(self._model.state == -1).delete()
        self._session.commit()
        return res

    def count(self, state: int = 0, is_all: bool = False):
        if is_all:
            return self._session.query(func.count(self._model.id)).scalar()
        return self._session.query(func.count(self._model.id)).filter(self._model.state == state).scalar()

    def close(self):
        self._session.close()
