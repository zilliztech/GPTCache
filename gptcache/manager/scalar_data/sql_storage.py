from typing import List

from datetime import datetime

from gptcache.utils import import_sqlalchemy
from gptcache.manager.scalar_data.base import CacheStorage, CacheData

import_sqlalchemy()

import sqlalchemy  # pylint: disable=wrong-import-position
from sqlalchemy import func, create_engine, Column, Sequence # pylint: disable=C0413
from sqlalchemy.types import (  # pylint: disable=C0413
    String,
    DateTime,
    LargeBinary,
    Integer
    )
from sqlalchemy.orm import sessionmaker  # pylint: disable=C0413
from sqlalchemy.ext.declarative import declarative_base  # pylint: disable=C0413

Base = declarative_base()


def get_models(table_prefix, db_type):
    class QuestionTable(Base):
        """
        question table
        """

        __tablename__ = table_prefix + "_question"
        __table_args__ = {"extend_existing": True}

        if db_type == "oracle":
            id = Column(
                Integer, Sequence("id_seq", start=1), primary_key=True, autoincrement=True
            )
        else:
            id = Column(Integer, primary_key=True, autoincrement=True)
        question = Column(String(1000), nullable=False)
        create_on = Column(DateTime, default=datetime.now)
        last_access = Column(DateTime, default=datetime.now)
        embedding_data = Column(LargeBinary, nullable=True)
        deleted = Column(Integer, default=0)

    class AnswerTable(Base):
        """
        answer table
        """
        __tablename__ = table_prefix + "_answer"
        __table_args__ = {"extend_existing": True}

        if db_type == "oracle":
            id = Column(
                Integer, Sequence("id_seq", start=1), primary_key=True, autoincrement=True
            )
        else:
            id = Column(Integer, primary_key=True, autoincrement=True)
        question_id = Column(Integer, nullable=False)
        answer = Column(String(1000), nullable=False)
        answer_type = Column(Integer, nullable=False)

    return QuestionTable, AnswerTable


class SQLStorage(CacheStorage):
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
        self._ques, self._answer = get_models(table_name, db_type)
        self._engine = create_engine(self._url)
        self.Session = sessionmaker(bind=self._engine)  # pylint: disable=invalid-name
        self.create()

    def create(self):
        self._ques.__table__.create(bind=self._engine, checkfirst=True)
        self._answer.__table__.create(bind=self._engine, checkfirst=True)

    def _insert(self, data: CacheData, session: sqlalchemy.orm.Session):
        ques_data = self._ques(
            question=data.question,
            embedding_data=data.embedding_data.tobytes()
            if data.embedding_data is not None
            else None,
        )
        session.add(ques_data)
        session.flush()
        answers = data.answer if isinstance(data.answer, list) else [data.answer]
        all_data = []
        for answer in answers:
            answer_data = self._answer(
                question_id=ques_data.id,
                answer=answer,
                answer_type=data.answer_type
            )
            all_data.append(answer_data)
        session.add_all(all_data)
        return ques_data.id

    def batch_insert(self, all_data: List[CacheData]):
        ids = []
        with self.Session() as session:
            for data in all_data:
                ids.append(self._insert(data, session))
            session.commit()
            return ids

    def get_data_by_id(self, key: int):
        # return 'question', 'answer', 'embedding'
        with self.Session() as session:
            qs = (
                session.query(self._ques.id, self._ques.question, self._ques.embedding_data)
                .filter(self._ques.id == key)
                .filter(self._ques.deleted == 0)
                .first()
            )
            if qs is None:
                return None
            ans = (
                session.query(self._answer.answer)
                .filter(self._answer.question_id == qs.id)
                .all()
            )
            res = list(qs[1:])
            res_ans = ans[0][0] if len(ans) == 1 else [item[0] for item in ans]
            res.insert(1, res_ans)
            return res

    def update_access_time(self, key: int):
        with self.Session() as session:
            session.query(self._ques).filter(self._ques.id == key).update(
                {"last_access": datetime.now()}
            )
            session.commit()

    def get_ids(self, deleted=True):
        state = -1 if deleted else 0
        with self.Session() as session:
            res = (
                session.query(self._ques.id).filter(self._ques.deleted == state).all()
            )
            return [item.id for item in res]

    def get_old_access(self, count):
        with self.Session() as session:
            res = (
                session.query(self._ques.id)
                .order_by(self._ques.last_access.asc())
                .filter(self._ques.deleted == 0)
                .limit(count)
                .all()
            )
            return [item.id for item in res]

    def get_old_create(self, count):
        with self.Session() as session:
            res = (
                session.query(self._ques.id)
                .order_by(self._ques.create_on.asc())
                .filter(self._ques.deleted == 0)
                .limit(count)
                .all()
            )
            return [item.id for item in res]

    def mark_deleted(self, keys):
        with self.Session() as session:
            session.query(self._ques).filter(self._ques.id.in_(keys)).update(
                {"deleted": -1}
            )
            session.commit()

    def clear_deleted_data(self):
        with self.Session() as session:
            objs = session.query(self._ques).filter(self._ques.deleted == -1)
            q_ids = [obj.id for obj in objs]
            session.query(self._answer).filter(self._answer.question_id.in_(q_ids)).delete()
            objs.delete()
            session.commit()

    def count(self, state: int = 0, is_all: bool = False):
        with self.Session() as session:
            if is_all:
                return session.query(func.count(self._ques.id)).scalar()
            return (
                session.query(func.count(self._ques.id))
                .filter(self._ques.deleted == state)
                .scalar()
            )

    def close(self):
        pass

    def count_answers(self):
        # for UT
        with self.Session() as session:
            return session.query(func.count(self._answer.id)).scalar()
