import datetime
from typing import List, Optional

import numpy as np

from gptcache.manager.scalar_data.base import (
    CacheStorage,
    CacheData,
    Question,
    QuestionDep,
)
from gptcache.utils import import_redis

import_redis()

# pylint: disable=C0413
from redis import Redis
from redis.client import Pipeline
from redis_om import get_redis_connection
from redis_om import JsonModel, EmbeddedJsonModel, NotFoundError, Field, Migrator


def get_models(global_key: str, redis_connection: Redis):
    """
    Get all the models for the given global key and redis connection.
    :param global_key: Global key will be used as a prefix for all the keys
    :type global_key: str

    :param redis_connection: Redis connection to use for all the models.
    Note: This needs to be explicitly mentioned in `Meta` class for each Object Model,
    otherwise it will use the default connection from the pool.
    :type redis_connection: Redis
    """

    class Counter:
        """
        counter collection
        """
        key_name = global_key + ":counter"
        database = redis_connection

        @classmethod
        def incr(cls):
            cls.database.incr(cls.key_name)

        @classmethod
        def get(cls):
            return cls.database.get(cls.key_name)

    class Embedding:
        """
        Custom class for storing embedding result.
        An embedding of type ``bytes`` is stored against Hash record type for the provided key.
        :param pk: Primary key against which hash data for embedding would be stored
        :type pk: str
        :param embedding: Embedding information to store
        :type embedding: bytes

        Note:
            As of this implementation, redis-om doesn't have a good compatibility to store bytes data
            and successfully retrieve it without corruption.
            In addition to that, decoding while getting the response is disabled as well.
        """

        prefix = global_key + ":embedding"

        def __init__(self, pk: str, embedding: bytes):
            self.pk = pk
            self.embedding = embedding

        def save(self, pipeline: Pipeline):
            pipeline.hset(self.prefix + ":" + str(self.pk), "embedding", self.embedding)

        @classmethod
        def get(cls, key: int, db: Redis):
            """
            Returns embedding stored against the ``key``.
            Decode only key value while creating a response
            :param key: redis key to fetch embedding
            :type key: str
            """
            result = db.hgetall(cls.prefix + ":" + str(key))
            return {k.decode("utf-8"): v for k, v in result.items()}

    class Answers(EmbeddedJsonModel):
        """
        answer collection
        """

        answer: str
        answer_type: int

        class Meta:
            database = redis_connection

    class Questions(JsonModel):
        """
        questions collection
        """

        question: str = Field(index=True)
        create_on: datetime.datetime
        last_access: datetime.datetime
        deleted: int = Field(index=True)
        answers: List[Answers]

        class Meta:
            global_key_prefix = global_key
            model_key_prefix = "questions"
            database = redis_connection

    class Sessions(JsonModel):
        """
        session collection
        """

        class Meta:
            global_key_prefix = global_key
            model_key_prefix = "sessions"
            database = redis_connection

        session_id: str = Field(index=True)
        session_question: str
        question_id: str = Field(index=True)

    class QuestionDeps(JsonModel):
        """
        Question Dep collection
        """

        class Meta:
            global_key_prefix = global_key
            model_key_prefix = "ques_deps"
            database = redis_connection

        question_id: str = Field(index=True)
        dep_name: str
        dep_data: str
        dep_type: int

    class Report(JsonModel):
        """
        Report collection
        """

        class Meta:
            global_key_prefix = global_key
            model_key_prefix = "report"
            database = redis_connection

        user_question: str
        cache_question_id: int = Field(index=True)
        cache_question: str
        cache_answer: str
        similarity: float = Field(index=True)
        cache_delta_time: float = Field(index=True)
        cache_time: datetime.datetime = Field(index=True)
        extra: Optional[str]

    return Questions, Embedding, Answers, QuestionDeps, Sessions, Counter, Report


class RedisCacheStorage(CacheStorage):
    """
     Using redis-om as OM to store data in redis cache storage

    :param host: redis host, default value 'localhost'
     :type host: str
     :param port: redis port, default value 27017
     :type port: int
     :param global_key_prefix: A global prefix for keys against which data is stored.
     For example, for a global_key_prefix ='gptcache', keys would be constructed would look like this:
     gptcache:questions:abc123
     :type global_key_prefix: str
     :param kwargs: Additional parameters to provide in order to create redis om connection

    Example:
        .. code-block:: python

            from gptcache.manager import CacheBase, manager_factory

            cache_store = CacheBase('redis',
                redis_host="localhost",
                redis_port=6379,
                global_key_prefix="gptcache",
            )
            # or
            data_manager = manager_factory("mongo,faiss", data_dir="./workspace",
                scalar_params={
                    "redis_host"="localhost",
                    "redis_port"=6379,
                    "global_key_prefix"="gptcache",
                },
                vector_params={"dimension": 128},
            )
    """

    def __init__(
            self,
            global_key_prefix="gptcache",
            host: str = "localhost",
            port: int = 6379,
            **kwargs
    ):
        self.con = get_redis_connection(host=host, port=port, **kwargs)

        self.con_encoded = get_redis_connection(
            host=host, port=port, decode_responses=False, **kwargs
        )

        (
            self._ques,
            self._embedding,
            self._answer,
            self._ques_dep,
            self._session,
            self._counter,
            self._report,
        ) = get_models(global_key_prefix, redis_connection=self.con)

        Migrator().run()

    def create(self):
        pass

    def _insert(self, data: CacheData, pipeline: Pipeline = None):
        self._counter.incr()
        pk = str(self._counter.get())
        answers = data.answers if isinstance(data.answers, list) else [data.answers]
        all_data = []
        for answer in answers:
            answer_data = self._answer(
                answer=answer.answer,
                answer_type=int(answer.answer_type),
            )
            all_data.append(answer_data)

        ques_data = self._ques(
            pk=pk,
            question=data.question
            if isinstance(data.question, str)
            else data.question.content,
            create_on=datetime.datetime.utcnow(),
            last_access=datetime.datetime.utcnow(),
            deleted=0,
            answers=answers,
        )

        ques_data.save(pipeline)

        embedding_data = (
            data.embedding_data.astype(np.float32).tobytes()
            if data.embedding_data is not None
            else None
        )
        self._embedding(pk=ques_data.pk, embedding=embedding_data).save(pipeline)

        if isinstance(data.question, Question) and data.question.deps is not None:
            all_deps = []
            for dep in data.question.deps:
                all_deps.append(
                    self._ques_dep(
                        question_id=ques_data.pk,
                        dep_name=dep.name,
                        dep_data=dep.data,
                        dep_type=dep.dep_type,
                    )
                )
            self._ques_dep.add(all_deps, pipeline=pipeline)

        if data.session_id:
            session_data = self._session(
                question_id=ques_data.pk,
                session_id=data.session_id,
                session_question=data.question
                if isinstance(data.question, str)
                else data.question.content,
            )
            session_data.save(pipeline)

        return int(ques_data.pk)

    def batch_insert(self, all_data: List[CacheData]):
        ids = []
        with self.con.pipeline() as pipeline:
            for data in all_data:
                ids.append(self._insert(data, pipeline=pipeline))
            pipeline.execute()
        return ids

    def get_data_by_id(self, key: str):
        key = str(key)
        try:
            qs = self._ques.get(pk=key)
        except NotFoundError:
            return None

        qs.update(last_access=datetime.datetime.utcnow())
        res_ans = [(item.answer, item.answer_type) for item in qs.answers]

        deps = self._ques_dep.find(self._ques_dep.question_id == key).all()
        res_deps = [
            QuestionDep(item.dep_name, item.dep_data, item.dep_type) for item in deps
        ]

        session_ids = [
            obj.session_id
            for obj in self._session.find(self._session.question_id == key).all()
        ]

        res_embedding = self._embedding.get(qs.pk, self.con_encoded)["embedding"]
        return CacheData(
            question=qs.question if not deps else Question(qs.question, res_deps),
            answers=res_ans,
            embedding_data=np.frombuffer(res_embedding, dtype=np.float32),
            session_id=session_ids,
            create_on=qs.create_on,
            last_access=qs.last_access,
        )

    def mark_deleted(self, keys):
        result = self._ques.find(self._ques.pk << keys).all()
        for qs in result:
            qs.update(deleted=-1)

    def clear_deleted_data(self):
        with self.con.pipeline() as pipeline:
            qs_to_delete = self._ques.find(self._ques.deleted == -1).all()
            self._ques.delete_many(qs_to_delete, pipeline)

            q_ids = [qs.pk for qs in qs_to_delete]
            sessions_to_delete = self._session.find(
                self._session.question_id << q_ids
            ).all()
            self._session.delete_many(sessions_to_delete, pipeline)

            deps_to_delete = self._ques_dep.find(
                self._ques_dep.question_id << q_ids
            ).all()
            self._ques_dep.delete_many(deps_to_delete, pipeline)

            pipeline.execute()

    def get_ids(self, deleted=True):
        state = -1 if deleted else 0
        res = [
            int(obj.pk) for obj in self._ques.find(self._ques.deleted == state).all()
        ]
        return res

    def count(self, state: int = 0, is_all: bool = False):
        if is_all:
            return self._ques.find().count()
        return self._ques.find(self._ques.deleted == state).count()

    def add_session(self, question_id, session_id, session_question):
        self._session(
            question_id=question_id,
            session_id=session_id,
            session_question=session_question,
        ).save()

    def list_sessions(self, session_id=None, key=None):
        if session_id and key:
            self._session.find(
                self._session.session_id == session_id
                and self._session.question_id == key
            ).all()
        if key:
            key = str(key)
            return self._session.find(self._session.question_id == key).all()
        if session_id:
            return self._session.find(self._session.session_id == session_id).all()
        return self._session.find().all()

    def delete_session(self, keys: List[str]):
        keys = [str(key) for key in keys]
        with self.con.pipeline() as pipeline:
            sessions_to_delete = self._session.find(
                self._session.question_id << keys
            ).all()
            self._session.delete_many(sessions_to_delete, pipeline)
            pipeline.execute()

    def report_cache(self, user_question, cache_question, cache_question_id, cache_answer, similarity_value,
                     cache_delta_time):
        self._report(
            user_question=user_question,
            cache_question=cache_question,
            cache_question_id=cache_question_id,
            cache_answer=cache_answer,
            similarity=similarity_value,
            cache_delta_time=cache_delta_time,
            cache_time=datetime.datetime.utcnow(),
        ).save()

    def close(self):
        self.con.close()
