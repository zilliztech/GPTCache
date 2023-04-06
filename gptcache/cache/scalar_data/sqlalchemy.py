from gptcache.utils import import_sqlalchemy
import_sqlalchemy()

import numpy as np
from datetime import datetime
from sqlalchemy import func, create_engine, Column, Sequence
from sqlalchemy.types import String, DateTime, LargeBinary, Integer
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

from .base import CacheStorage

Base = declarative_base()


def get_model(table_name, db_type):
    class CacheTable(Base):
        """
        cache_table
        """
        __tablename__ = table_name
        __table_args__ = {'extend_existing': True}

        uid = Column(Integer, primary_key=True, autoincrement=True)
        id = Column(String(500), nullable=False)
        data = Column(String(1000), nullable=False)
        reply = Column(String(1000), nullable=False)
        create_on = Column(DateTime, default=datetime.now)
        last_access = Column(DateTime, default=datetime.now)
        embedding_data = Column(LargeBinary, nullable=True)
        state = Column(Integer, default=0)

    class CacheTableSequence(Base):
        """
        cache_table sequence
        """
        __tablename__ = table_name
        __table_args__ = {'extend_existing': True}

        uid = Column(Integer, Sequence('id_seq', start=1), primary_key=True, autoincrement=True)
        id = Column(String(500), nullable=False)
        data = Column(String(1000), nullable=False)
        reply = Column(String(1000), nullable=False)
        create_on = Column(DateTime, default=datetime.now)
        last_access = Column(DateTime, default=datetime.now)
        embedding_data = Column(LargeBinary, nullable=True)
        state = Column(Integer, default=0)

    if db_type == 'oracle':
        return CacheTableSequence
    else:
        return CacheTable


class SQLDataBase(CacheStorage):
    """
    Using sqlalchemy to manage SQLite, PostgreSQL, MySQL, MariaDB, SQL Server and Oracle.
    """
    def __init__(self, db_type: str = 'sqlite', url: str = 'sqlite:///./gptcache.db', table_name: str = 'gptcache'):
        self._url = url
        self._engine = None
        self._session = None
        self._model = get_model(table_name, db_type)
        self.init()

    def init(self):
        self._engine = create_engine(self._url)
        Session = sessionmaker(bind=self._engine)
        self._session = Session()
        self.create()

    def create(self):
        self._model.__table__.create(bind=self._engine, checkfirst=True)

    def insert(self, key, data, reply, embedding_data: np.ndarray = None):
        embedding_data = embedding_data.tobytes()
        model_obj = self._model(id=key, data=data, reply=reply, embedding_data=embedding_data)
        self._session.add(model_obj)
        self._session.commit()

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
        res = self._session.query(self._model.id).order_by(self._model.last_access.asc()).filter(self._model.state == 0).limit(count).all()
        return res

    def get_old_create(self, count):
        res = self._session.query(self._model.id).order_by(self._model.create_on.asc()).filter(self._model.state == 0).limit(count).all()
        return res

    def update_state(self, keys):
        self._session.query(self._model).filter(self._model.id.in_(keys)).update({'state': -1})
        self._session.commit()

    def remove_by_state(self):
        self._session.query(self._model).filter(self._model.state == -1).delete()
        self._session.commit()

    def count(self, state: int = 0, is_all: bool = False):
        if is_all:
            return self._session.query(func.count(self._model.id)).scalar()
        return self._session.query(func.count(self._model.id)).filter(self._model.state == state).scalar()

    def close(self):
        self._session.close()
