from datetime import datetime
from typing import List, Optional

import numpy as np

from gptcache.manager.scalar_data.base import (
    CacheStorage,
    CacheData,
    Question,
    QuestionDep,
)
from gptcache.utils import import_mongodb

import_mongodb()

# pylint: disable=C0413
from mongoengine import Document
from mongoengine import fields
import mongoengine as me


def get_models():
    class Questions(Document):
        """
        questions collection
        """

        meta = {"collection": "questions", "indexes": ["deleted"]}
        _id = fields.SequenceField()
        question = fields.StringField()
        create_on = fields.DateTimeField(default=datetime.now())
        last_access = fields.DateTimeField(default=datetime.now())
        embedding_data = fields.BinaryField()
        deleted = fields.IntField(default=0)

        @property
        def oid(self):
            return self._id

    class Answers(Document):
        """
        answer collection
        """

        _id = fields.SequenceField()
        meta = {"collection": "answers", "indexes": ["question_id"]}
        answer = fields.StringField()
        answer_type = fields.IntField()
        question_id = fields.IntField()

        @property
        def oid(self):
            return self._id

    class Sessions(Document):
        """
        session collection
        """

        meta = {"collection": "sessions", "indexes": ["question_id"]}
        _id = fields.SequenceField()
        session_id = fields.StringField()
        session_question = fields.StringField()
        question_id = fields.IntField()

        @property
        def oid(self):
            return self._id

    class QuestionDeps(Document):
        """
        Question Dep collection
        """

        meta = {"collection": "question_deps", "indexes": ["question_id"]}
        _id = fields.SequenceField()
        question_id = fields.IntField()
        dep_name = fields.StringField()
        dep_data = fields.StringField()
        dep_type = fields.IntField()

        @property
        def oid(self):
            return self._id

    class Report(Document):
        """
        Report
        """

        meta = {
            "collection": "report",
            "indexes": ["cache_question_id", "similarity", "cache_delta_time"],
        }
        _id = fields.SequenceField()
        user_question = fields.StringField()
        cache_question_id = fields.IntField()
        cache_question = fields.StringField()
        cache_answer = fields.StringField()
        similarity = fields.FloatField()
        cache_delta_time = fields.FloatField()
        cache_time = fields.DateTimeField(default=datetime.now())
        extra = fields.StringField()

        @property
        def oid(self):
            return self._id

    return Questions, Answers, QuestionDeps, Sessions, Report


class MongoStorage(CacheStorage):
    """
    Using mongoengine as ORM to manage mongodb documents.
    By default, data is stored 'gptcache' database and following collections are created to store the data
        1. 'sessions'
        2. 'answers'
        3. 'questions'
        4. 'question_deps'

    :param host: mongodb host, default value 'localhost'
    :type host: str
    :param port: mongodb port, default value 27017
    :type host: int
    :param dbname: database name, default value 'gptcache'
    :type host: str
    :param : Mongo database name, default value 'gptcache'
    :type host: str
    :param username: username for authentication, default value None
    :type host: str
    :param password: password for authentication, default value None
    :type host: str

    Example:
        .. code-block:: python

            from gptcache.manager import CacheBase, manager_factory

            cache_store = CacheBase('mongo',
                mongo_host="localhost",
                mongo_port=27017,
                dbname="gptcache",
                username=None,
                password=None,
            )
            # or
            data_manager = manager_factory("mongo,faiss", data_dir="./workspace",
                scalar_params={
                    "mongo_host": "localhost",
                    "mongo_port": 27017,
                    "dbname"="gptcache",
                    "username"="",
                    "password"="",
                },
                vector_params={"dimension": 128},
            )
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 27017,
        dbname: str = "gptcache",
        username: str = None,
        password: str = None,
        **kwargs
    ):
        self.con = me.connect(
            host=host,
            port=port,
            db=dbname,
            username=username,
            password=password,
            **kwargs
        )
        (
            self._ques,
            self._answer,
            self._ques_dep,
            self._session,
            self._report,
        ) = get_models()

    def create(self):
        pass

    def _insert(self, data: CacheData):
        ques_data = self._ques(
            question=data.question
            if isinstance(data.question, str)
            else data.question.content,
            embedding_data=data.embedding_data.tobytes()
            if data.embedding_data is not None
            else None,
        )
        ques_data.save()
        if isinstance(data.question, Question) and data.question.deps is not None:
            all_deps = []
            for dep in data.question.deps:
                all_deps.append(
                    self._ques_dep(
                        question_id=ques_data.oid,
                        dep_name=dep.name,
                        dep_data=dep.data,
                        dep_type=dep.dep_type,
                    )
                )
            self._ques_dep.objects.insert(all_deps)

        answers = data.answers if isinstance(data.answers, list) else [data.answers]
        all_data = []
        for answer in answers:
            answer_data = self._answer(
                question_id=ques_data.oid,
                answer=answer.answer,
                answer_type=int(answer.answer_type),
            )
            all_data.append(answer_data)
        self._answer.objects.insert(all_data)

        if data.session_id:
            session_data = self._session(
                question_id=ques_data.oid,
                session_id=data.session_id,
                session_question=data.question
                if isinstance(data.question, str)
                else data.question.content,
            )
            self._session.objects.insert(session_data)

        return ques_data.oid

    def batch_insert(self, all_data: List[CacheData]):
        ids = []
        for data in all_data:
            ids.append(self._insert(data))
        return ids

    def get_data_by_id(self, key) -> Optional[CacheData]:
        qs = self._ques.objects.get(_id=key, deleted=0)
        if qs is None:
            return None
        last_access = qs.last_access
        qs.last_access = datetime.now()
        qs.save()
        answers = self._answer.objects(question_id=qs.oid)
        deps = self._ques_dep.objects(question_id=qs.oid)
        session_ids = self._session.objects(question_id=qs.oid)

        res_ans = [(item.answer, item.answer_type) for item in answers]
        res_deps = [
            QuestionDep(item.dep_name, item.dep_data, item.dep_type) for item in deps
        ]
        return CacheData(
            question=qs.question if not deps else Question(qs.question, res_deps),
            answers=res_ans,
            embedding_data=np.frombuffer(qs.embedding_data, dtype=np.float32),
            session_id=session_ids,
            create_on=qs.create_on,
            last_access=last_access,
        )

    def mark_deleted(self, keys):
        self._ques.objects(_id__in=keys).update(deleted=-1)

    def clear_deleted_data(self):
        questions = self._ques.objects(deleted=-1).only("_id")
        q_ids = [obj.oid for obj in questions]
        self._answer.objects(question_id__in=q_ids).delete()
        self._ques_dep.objects(question_id__in=q_ids).delete()
        self._session.objects(question_id__in=q_ids).delete()
        questions.delete()

    def get_ids(self, deleted: bool = True):
        state = -1 if deleted else 0
        res = [obj.oid for obj in self._ques.objects(deleted=state).only("_id")]
        return res

    def count(self, state: int = 0, is_all: bool = False):
        if is_all:
            return self._ques.objects.count()
        return self._ques.objects(deleted=state).count()

    def add_session(self, question_id, session_id, session_question):
        self._session(
            question_id=question_id,
            session_id=session_id,
            session_question=session_question,
        ).save()

    def list_sessions(self, session_id=None, key=None):
        query = {}
        if session_id:
            query["session_id"] = session_id
        if key:
            query["question_id"] = key

        return self._session.objects(__raw__=query)

    def delete_session(self, keys):
        self._session.objects(question_id__in=keys).delete()

    def count_answers(self):
        return self._answer.objects.count()

    def report_cache(
        self,
        user_question,
        cache_question,
        cache_question_id,
        cache_answer,
        similarity_value,
        cache_delta_time,
    ):
        report_data = self._report(
            user_question=user_question,
            cache_question=cache_question,
            cache_question_id=cache_question_id,
            cache_answer=cache_answer,
            similarity=similarity_value,
            cache_delta_time=cache_delta_time,
        )
        report_data.save()

    def close(self):
        me.disconnect()
        self.con.close()
