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

    class EmbeddingType:
        """
        Directly using bytes for embedding data is not supported by redis-om as of now.
        Custom type for embedding data. This will be stored as bytes in redis.
        Latin-1 encoding is used to convert the bytes to string and vice versa.
        """

        def __init__(self, data: bytes):
            self.data = data

        @classmethod
        def __get_validators__(cls):
            yield cls.validate

        @classmethod
        def validate(cls, v: [np.array, bytes]):
            if isinstance(v, np.ndarray):
                return cls(v.astype(np.float32).tobytes())
            elif isinstance(v, bytes):
                return cls(v)

            return cls(v)

        def to_numpy(self) -> np.ndarray:
            return np.frombuffer(self.data.encode("latin-1"), dtype=np.float32)

        def __repr__(self):
            return f"{self.data}"

    class Answers(EmbeddedJsonModel):
        """
        answer collection
        """

        answer: str
        answer_type: int

        class Meta:
            database = redis_connection

    class QuestionDeps(EmbeddedJsonModel):
        """
        Question Dep collection
        """

        dep_name: str
        dep_data: str
        dep_type: int

    class Questions(JsonModel):
        """
        questions collection
        """

        question: str = Field(index=True)
        create_on: datetime.datetime
        last_access: datetime.datetime
        deleted: int = Field(index=True)
        answers: List[Answers]
        deps: List[QuestionDeps]
        embedding: EmbeddingType

        class Meta:
            global_key_prefix = global_key
            model_key_prefix = "questions"
            database = redis_connection

        class Config:
            json_encoders = {
                EmbeddingType: lambda n: n.data.decode("latin-1")
                if isinstance(n.data, bytes) else n.data
            }

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

    return Questions, Answers, QuestionDeps, Sessions, Counter, Report


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
    :param maxmemory: Maximum memory to use for redis cache storage
    :type maxmemory: str
    :param policy: Policy to use for eviction, default value 'allkeys-lru'
    :type policy: str
    :param ttl: Time to live for keys in milliseconds, default value None
    :type ttl: int
    :param maxmemory_samples: Number of keys to sample when evicting keys
    :type maxmemory_samples: int
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
            global_key_prefix: str = "gptcache",
            host: str = "localhost",
            port: int = 6379,
            maxmemory: str = None,
            policy: str = None,
            ttl: int = None,
            maxmemory_samples: int = None,
            **kwargs
    ):
        self.con = get_redis_connection(host=host, port=port, **kwargs)
        self.default_ttl = ttl
        self.init_eviction_params(policy=policy, maxmemory=maxmemory, maxmemory_samples=maxmemory_samples, ttl=ttl)
        (
            self._ques,
            self._answer,
            self._ques_dep,
            self._session,
            self._counter,
            self._report,
        ) = get_models(global_key_prefix, redis_connection=self.con)

        Migrator().run()

    def init_eviction_params(self, policy, maxmemory, maxmemory_samples, ttl):
        self.default_ttl = ttl
        if maxmemory:
            self.con.config_set("maxmemory", maxmemory)
        if policy:
            self.con.config_set("maxmemory-policy", policy)
        if maxmemory_samples:
            self.con.config_set("maxmemory-samples", maxmemory_samples)

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

        all_deps = []
        if isinstance(data.question, Question) and data.question.deps is not None:
            for dep in data.question.deps:
                all_deps.append(
                    self._ques_dep(
                        dep_name=dep.name,
                        dep_data=dep.data,
                        dep_type=dep.dep_type,
                    )
                )
        embedding_data = (
            data.embedding_data
            if data.embedding_data is not None
            else None
        )
        ques_data = self._ques(
            pk=pk,
            question=data.question
            if isinstance(data.question, str)
            else data.question.content,
            create_on=datetime.datetime.utcnow(),
            last_access=datetime.datetime.utcnow(),
            deleted=0,
            answers=answers,
            deps=all_deps,
            embedding=embedding_data
        )

        ques_data.save(pipeline)

        if data.session_id:
            session_data = self._session(
                question_id=ques_data.pk,
                session_id=data.session_id,
                session_question=data.question
                if isinstance(data.question, str)
                else data.question.content,
            )
            session_data.save(pipeline)
        if self.default_ttl:
            ques_data.expire(self.default_ttl, pipeline=pipeline)
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
        res_deps = [
            QuestionDep(item.dep_name, item.dep_data, item.dep_type) for item in qs.deps
        ]

        session_ids = [
            obj.session_id
            for obj in self._session.find(self._session.question_id == key).all()
        ]
        if self.default_ttl:
            qs.expire(self.default_ttl)
        return CacheData(
            question=qs.question if not res_deps else Question(qs.question, res_deps),
            answers=res_ans,
            embedding_data=qs.embedding.to_numpy(),
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
