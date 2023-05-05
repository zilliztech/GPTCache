from typing import List, Optional
from datetime import datetime
import numpy as np
from gptcache.utils import import_sqlalchemy
from gptcache.manager.scalar_data.base import (
    CacheStorage,
    CacheData,
    Question,
    QuestionDep,
)

import_sqlalchemy()

import sqlalchemy  # pylint: disable=wrong-import-position
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


def get_models(table_prefix, db_type):
    class QuestionTable(Base):
        """
        question table
        """

        __tablename__ = table_prefix + "_question"
        __table_args__ = {"extend_existing": True}

        if db_type in ("oracle", "duckdb"):
            question_id_seq = Sequence(f"{__tablename__}_id_seq", start=1)
            id = Column(Integer, question_id_seq, primary_key=True, autoincrement=True)
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

        if db_type in ("oracle", "duckdb"):
            answer_id_seq = Sequence(f"{__tablename__}_id_seq")
            id = Column(Integer, answer_id_seq, primary_key=True, autoincrement=True)
        else:
            id = Column(Integer, primary_key=True, autoincrement=True)
        question_id = Column(Integer, nullable=False)
        answer = Column(String(2000), nullable=False)
        answer_type = Column(Integer, nullable=False)

    class SessionTable(Base):
        """
        session table
        """

        __tablename__ = table_prefix + "_session"
        __table_args__ = {"extend_existing": True}

        if db_type in ("oracle", "duckdb"):
            session_id_seq = Sequence(f"{__tablename__}_id_seq", start=1)
            id = Column(
                Integer,
                session_id_seq,
                primary_key=True,
                autoincrement=True,
            )
        else:
            id = Column(Integer, primary_key=True, autoincrement=True)
        question_id = Column(Integer, nullable=False)
        session_id = Column(String(500), nullable=False)
        session_question = Column(String(2000), nullable=False)

    class QuestionDepTable(Base):
        """
        answer table
        """

        __tablename__ = table_prefix + "_question_dep"
        __table_args__ = {"extend_existing": True}

        if db_type in ("oracle", "duckdb"):
            question_dep_id_seq = Sequence(f"{__tablename__}_id_seq", start=1)
            id = Column(
                Integer, question_dep_id_seq, primary_key=True, autoincrement=True
            )
        else:
            id = Column(Integer, primary_key=True, autoincrement=True)
        question_id = Column(Integer, nullable=False)
        dep_name = Column(String(255), nullable=False)
        dep_data = Column(String(1000), nullable=False)
        dep_type = Column(Integer, nullable=False)

    return QuestionTable, AnswerTable, QuestionDepTable, SessionTable


class SQLStorage(CacheStorage):
    """
    Using sqlalchemy to manage SQLite, PostgreSQL, MySQL, MariaDB, SQL Server and Oracle.

    :param name: the name of the cache storage, it is support 'sqlite', 'postgresql', 'mysql', 'mariadb', 'sqlserver' and  'oracle' now.
    :type name: str
    :param sql_url: the url of the sql database for cache, such as '<db_type>+<db_driver>://<username>:<password>@<host>:<port>/<database>',
                    and the default value is related to the `cache_store` parameter,
                    'sqlite:///./sqlite.db' for 'sqlite',
                    'duckdb:///./duck.db' for 'duckdb',
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
        self._ques, self._answer, self._ques_dep, self._session = get_models(
            table_name, db_type
        )
        self._engine = create_engine(self._url)
        self.Session = sessionmaker(bind=self._engine)  # pylint: disable=invalid-name
        self.create()

    def create(self):
        self._ques.__table__.create(bind=self._engine, checkfirst=True)
        self._answer.__table__.create(bind=self._engine, checkfirst=True)
        self._ques_dep.__table__.create(bind=self._engine, checkfirst=True)
        self._session.__table__.create(bind=self._engine, checkfirst=True)

    def _insert(self, data: CacheData, session: sqlalchemy.orm.Session) -> Column:
        ques_data = self._ques(
            question=data.question
            if isinstance(data.question, str)
            else data.question.content,
            embedding_data=data.embedding_data.tobytes()
            if data.embedding_data is not None
            else None,
        )
        session.add(ques_data)
        session.flush()
        if isinstance(data.question, Question) and data.question.deps is not None:
            all_deps = []
            for dep in data.question.deps:
                all_deps.append(
                    self._ques_dep(
                        question_id=ques_data.id,
                        dep_name=dep.name,
                        dep_data=dep.data,
                        dep_type=dep.dep_type,
                    )
                )
            session.add_all(all_deps)
        answers = data.answers if isinstance(data.answers, list) else [data.answers]
        all_data = []
        for answer in answers:
            answer_data = self._answer(
                question_id=ques_data.id,
                answer=answer.answer,
                answer_type=int(answer.answer_type),
            )
            all_data.append(answer_data)
        session.add_all(all_data)
        if data.session_id:
            session_data = self._session(
                question_id=ques_data.id,
                session_id=data.session_id,
                session_question=data.question
                if isinstance(data.question, str)
                else data.question.content,
            )
            session.add(session_data)
        return ques_data.id

    def batch_insert(self, all_data: List[CacheData]):
        ids = []
        with self.Session() as session:
            for data in all_data:
                ids.append(self._insert(data, session))
            session.commit()
        return ids

    def get_data_by_id(self, key: int) -> Optional[CacheData]:
        with self.Session() as session:
            qs = (
                session.query(
                    self._ques.id, self._ques.question, self._ques.embedding_data
                )
                .filter(self._ques.id == key)
                .filter(self._ques.deleted == 0)
                .first()
            )
            if qs is None:
                return None
            ans = (
                session.query(self._answer.answer, self._answer.answer_type)
                .filter(self._answer.question_id == qs.id)
                .all()
            )
            deps = (
                session.query(
                    self._ques_dep.dep_name,
                    self._ques_dep.dep_data,
                    self._ques_dep.dep_type,
                )
                .filter(self._ques_dep.question_id == qs.id)
                .all()
            )
            session_ids = (
                session.query(self._session.session_id)
                .filter(self._session.question_id == qs.id)
                .all()
            )
            res_ans = [(item.answer, item.answer_type) for item in ans]
            res_deps = [
                QuestionDep(item.dep_name, item.dep_data, item.dep_type)
                for item in deps
            ]
            return CacheData(
                question=qs[1] if not deps else Question(qs[1], res_deps),
                answers=res_ans,
                embedding_data=np.frombuffer(qs[2], dtype=np.float32),
                session_id=session_ids,
            )

    def get_ids(self, deleted=True):
        state = -1 if deleted else 0
        with self.Session() as session:
            res = session.query(self._ques.id).filter(self._ques.deleted == state).all()
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
            session.query(self._answer).filter(
                self._answer.question_id.in_(q_ids)
            ).delete()
            session.query(self._ques_dep).filter(
                self._ques_dep.question_id.in_(q_ids)
            ).delete()
            session.query(self._session).filter(
                self._session.question_id.in_(q_ids)
            ).delete()
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

    def add_session(self, question_id, session_id, session_question):
        with self.Session() as session:
            session_data = self._session(
                question_id=question_id,
                session_id=session_id,
                session_question=session_question,
            )
            session.add(session_data)
            session.commit()

    def delete_session(self, keys):
        with self.Session() as session:
            session.query(self._session).filter(self._session.id.in_(keys)).delete()
            session.commit()

    def list_sessions(self, session_id=None, key=None):
        with self.Session() as session:
            query = session.query(self._session)
            if session_id:
                query = query.filter(self._session.session_id == session_id)
            elif key:
                query = query.filter(self._session.question_id == key)
            return query.all()

    def close(self):
        pass

    def count_answers(self):
        # for UT
        with self.Session() as session:
            return session.query(func.count(self._answer.id)).scalar()
