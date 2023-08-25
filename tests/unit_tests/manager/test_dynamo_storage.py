import numpy as np
import pytest
import unittest

from random import randint
from uuid import uuid4
from datetime import datetime
from decimal import Decimal
from gptcache.manager.scalar_data.base import CacheStorage, CacheData, DataType, Question, QuestionDep
from gptcache.utils import import_boto3
from gptcache.manager.scalar_data.dynamo_storage import DynamoStorage 

import_boto3()

from boto3.dynamodb.conditions import Attr as DynamoAttr, Key as DynamoKey
from boto3 import client as awsclient, resource as awsresource
import boto3

class TestDynamoCacheStorage(unittest.TestCase):
    _dynamodb_local_endpoint_url = "http://localhost:9999"
    _region_name = "us-east-1"

    def setUp(self):
        # First find all the tables and blow them away to keep a clean slate before starting tests
        dynamo_client = awsclient(
            "dynamodb",
            endpoint_url = self._dynamodb_local_endpoint_url,
            aws_access_key_id = "test",
            aws_secret_access_key = "test",
            region_name = self._region_name,
        )
        table_names = dynamo_client.list_tables()["TableNames"]

        for table_name in table_names:
            dynamo_client.delete_table(TableName=table_name)

        self.dynamo_cache_storage = DynamoStorage(
            aws_endpoint_url = self._dynamodb_local_endpoint_url,
            aws_access_key_id = "test",
            aws_secret_access_key = "test",
            aws_region_name = self._region_name,
        )

    def test_create_is_idempotent(self):
        try:
            self.dynamo_cache_storage.create()
            self.dynamo_cache_storage.create()
            self.dynamo_cache_storage.create()
            self.dynamo_cache_storage.create()
        except Exception as e:
            pytest.fail("create() method should be idempotent. It failed with exception: " + str(e))

        # now manually query via boto3 to make sure the table is there
        table = self._dynamo_resource().Table("gptcache_questions")
        assert table is not None

        table = self._dynamo_resource().Table("gptcache_reports")
        assert table is not None

    def test_creation_and_batch_insert(self):
        time_before_insertion = datetime.utcnow()
        cache_entries_to_insert = [
            self._random_cachedata_without_dependencies(session_id = "1"),
            self._random_cachedata_without_dependencies(),
            self._random_cachedata_with_dependencies(session_id = "3"),
        ]
        self.dynamo_cache_storage.batch_insert(cache_entries_to_insert)

        # now manually query via boto3 to make sure the data is there
        table = self._dynamo_resource().Table("gptcache_questions")
        items = table.scan()['Items']

        # Since there are session_ids for each item except one, there should be a corresponding session entries in this
        # table. So there should be (3 regular question rows + 2 session rows) = 5 items in the table.
        assert len(items) == 5

        for item in items:
            original_data_inserted = next(
                    (cache_data for cache_data in cache_entries_to_insert if (cache_data.question == item["question"] or 
                        isinstance(cache_data.question, Question) and cache_data.question.content == item["question"])),
                    None
            )
            expected_question_text = (
                original_data_inserted.question
                if isinstance(original_data_inserted.question, str)
                else original_data_inserted.question.content
            )

            if item["id"].startswith("sessions#"):
                assert original_data_inserted.session_id == item["id"].split("#")[1]
                assert expected_question_text == item["question"]

            else:
                assert expected_question_text == item["question"]

                # ensure that a 'create_on' timestamp is always set for new items.
                assert (datetime.fromisoformat(item["create_on"]) > time_before_insertion and 
                        datetime.fromisoformat(item["create_on"]) < datetime.utcnow())

                # The last access value should be blank because the entry is new and nothing has accessed it yet.
                assert "last_access" not in item 

                # It shouldn't be deleted. Since boolean values have low cardinality, the class should be suffixing the value
                # with a random integer to avoid hot partitions as per AWS's recommendations.
                assert item["deleted"].startswith("False_")

                assert (original_data_inserted.embedding_data == np.frombuffer(item["embedding_data"].value, dtype=np.float32)).all()

                # if the question has dependencies, make sure they're all there
                if isinstance(original_data_inserted.question, Question) and original_data_inserted.question.deps is not None:
                    assert len(original_data_inserted.question.deps) == len(item["deps"])
                    for dep in original_data_inserted.question.deps:
                        matching_cache_entry = next(
                                (dep_dict for dep_dict in item["deps"] if dep.name == dep_dict["name"]),
                                None
                        )
                        assert matching_cache_entry is not None
                        assert matching_cache_entry["data"] == dep.data
                        assert matching_cache_entry["dep_type"] == dep.dep_type

                for answer in original_data_inserted.answers:
                    matching_cache_entry = next((answer_dict for answer_dict in item["answers"] if answer.answer == answer_dict["answer"]), None)
                    assert matching_cache_entry is not None
                    assert matching_cache_entry["answer_type"] == answer.answer_type

    def test_mark_deleted(self):
        item_to_insert = self._random_cachedata_without_dependencies(session_id = "1")
        persisted_id = self.dynamo_cache_storage.batch_insert([item_to_insert])[0]

        # now mark it as deleted
        self.dynamo_cache_storage.mark_deleted([persisted_id])

        # it should have been soft deleted. Boto3 API shouldn't even return an "Item" obj in the response if its deleted
        table = self._dynamo_resource().Table("gptcache_questions")
        resp = table.get_item(
            Key={"pk": f"questions#{persisted_id}", "id": f"questions#{persisted_id}"},
        )
        assert resp["Item"]["deleted"].startswith("True_")

    def test_clear_deleted_data(self):
        # first let's insert some data, mark them as deleted
        items_to_insert = [
            self._random_cachedata_without_dependencies(session_id = "1"),
            self._random_cachedata_without_dependencies(),
            self._random_cachedata_without_dependencies(),
        ]
        persisted_ids = self.dynamo_cache_storage.batch_insert(items_to_insert)

        # only delete the first 2 (just to ensure we can later assert the method in test doesn't actually 
        # delete ALL data from the table)
        self.dynamo_cache_storage.mark_deleted(persisted_ids[:2])

        # now clear the deleted data. This should remove it completely from the table (aka. hard delete)
        self.dynamo_cache_storage.clear_deleted_data()

        # now query the table and make sure the first 2 items are gone, but the last one is still there
        table = self._dynamo_resource().Table("gptcache_questions")
        resp = table.scan(FilterExpression = DynamoAttr("id").begins_with("questions#"))

        assert len(resp["Items"]) == 1
        assert int(resp["Items"][0]["id"].replace("questions#", "")) == persisted_ids[2]

    def test_get_ids(self):
        # first let's insert some data, mark them as deleted
        items_to_insert = [
            self._random_cachedata_without_dependencies(session_id = "1"),
            self._random_cachedata_without_dependencies(),
            self._random_cachedata_without_dependencies(),
        ]
        persisted_ids = self.dynamo_cache_storage.batch_insert(items_to_insert)

        # only delete the first 2 (just to ensure we can later assert the method in test doesn't actually 
        # delete ALL data from the table)
        self.dynamo_cache_storage.mark_deleted(persisted_ids[:2])

        # by default, the get_ids method should return only deleted ids. all ids, including deleted ones
        deleted_ids = self.dynamo_cache_storage.get_ids()
        assert sorted(deleted_ids) == sorted(persisted_ids[:2])

        # now if we pass in deleted = True, it should, again, only return the deleted ids
        deleted_ids = self.dynamo_cache_storage.get_ids(deleted = True)
        assert sorted(deleted_ids) == sorted(persisted_ids[:2])

        # now if we pass in deleted = False, it should return the undeleted items 
        undeleted_ids = self.dynamo_cache_storage.get_ids(deleted = False)
        assert sorted(undeleted_ids) == sorted([persisted_ids[2]])


    def test_count(self):
        # first let's insert some data, mark them as deleted
        items_to_insert = [
            self._random_cachedata_without_dependencies(session_id = "1"),
            self._random_cachedata_without_dependencies(session_id = "1"),
            self._random_cachedata_without_dependencies(),
        ]
        persisted_ids = self.dynamo_cache_storage.batch_insert(items_to_insert)

        # delete one of the items
        self.dynamo_cache_storage.mark_deleted([persisted_ids[0]])

        # by default, it retuns only the undeleted ids
        assert self.dynamo_cache_storage.count() == 2

        # now if we pass in state = 0, it should only return the undeleted ids
        # Doing the opposite will return only deleted ids.
        assert self.dynamo_cache_storage.count(state = 0) == 2
        assert self.dynamo_cache_storage.count(state = -1) == 1

        assert self.dynamo_cache_storage.count(is_all = True) == 3

    def test_add_session(self):
        item_to_insert = self._random_cachedata_with_dependencies()
        persisted_id = self.dynamo_cache_storage.batch_insert([item_to_insert])[0]

        # now add a question to some sessions 
        session_id1 = str(uuid4())
        session_id2 = str(uuid4())
        self.dynamo_cache_storage.add_session(persisted_id, session_id1, item_to_insert.question.content)
        self.dynamo_cache_storage.add_session(persisted_id, session_id2, item_to_insert.question.content)

        # now query the table and make sure the sessions are there
        table = self._dynamo_resource().Table("gptcache_questions")
        session1_questions = table.query(
            IndexName = "gsi_items_by_type",
            KeyConditionExpression = DynamoKey("id").eq(f"sessions#{session_id1}")
        )["Items"]
        session2_questions = table.query(
            IndexName = "gsi_items_by_type",
            KeyConditionExpression = (DynamoKey("id").eq(f"sessions#{session_id2}"))
        )["Items"]

        assert len(session1_questions) == 1
        assert session1_questions[0]["question"] == item_to_insert.question.content
        assert int(session1_questions[0]["pk"].replace("questions#", "")) == persisted_id

        assert len(session2_questions) == 1
        assert session2_questions[0]["question"] == item_to_insert.question.content
        assert int(session2_questions[0]["pk"].replace("questions#", "")) == persisted_id

    def test_list_sessions(self):
        items_to_insert = [
            self._random_cachedata_with_dependencies(),
            self._random_cachedata_with_dependencies(),
        ]
        persisted_ids = self.dynamo_cache_storage.batch_insert(items_to_insert)

        # now add a question to some sessions 
        session_id1 = str(uuid4())
        session_id2 = str(uuid4())
        session_id3 = str(uuid4())
        self.dynamo_cache_storage.add_session(persisted_ids[0], session_id1, items_to_insert[0].question.content)
        self.dynamo_cache_storage.add_session(persisted_ids[0], session_id2, items_to_insert[0].question.content)
        self.dynamo_cache_storage.add_session(persisted_ids[1], session_id3, items_to_insert[1].question.content)

        ids_without_prefix = lambda ids: [id_with_prefix.replace("sessions#", "") for id_with_prefix in ids]

        # now we should be able to see those sessions
        session_ids_with_prefix = self.dynamo_cache_storage.list_sessions()
        assert sorted(ids_without_prefix(session_ids_with_prefix)) == sorted([session_id1, session_id2, session_id3])

        # now filter by one of the session ids
        session_ids_with_prefix = self.dynamo_cache_storage.list_sessions(session_id = session_id1)
        assert ids_without_prefix(session_ids_with_prefix) == [session_id1]

        # now filter by the question id
        session_ids_with_prefix = self.dynamo_cache_storage.list_sessions(key = persisted_ids[1])
        assert sorted(ids_without_prefix(session_ids_with_prefix)) == sorted([session_id3])

        # now filter by both
        session_ids_with_prefix = self.dynamo_cache_storage.list_sessions(session_id = session_id2, key = persisted_ids[0])
        assert ids_without_prefix(session_ids_with_prefix) == [session_id2]

    def test_delete_session(self):
        session_id1 = str(uuid4())
        session_id2 = str(uuid4())
        items_to_insert = [
            self._random_cachedata_with_dependencies(session_id = session_id1),
            self._random_cachedata_with_dependencies(session_id = session_id1),
            self._random_cachedata_with_dependencies(session_id = session_id2),
            self._random_cachedata_with_dependencies(),
        ]
        persisted_ids = self.dynamo_cache_storage.batch_insert(items_to_insert)

        # now delete the session
        self.dynamo_cache_storage.delete_session([session_id1])

        # There shouldn't be a session entry for session_id1 anymore but the other sessions should still exist.
        session_ids_with_prefix = self.dynamo_cache_storage.list_sessions()
        assert session_ids_with_prefix == [f"sessions#{session_id2}"]


    def test_get_data_by_id(self):
        time_before_insertion = datetime.utcnow()

        item_to_insert = self._random_cachedata_with_dependencies(session_id = "1")
        persisted_id = self.dynamo_cache_storage.batch_insert([item_to_insert])[0]

        entry = self.dynamo_cache_storage.get_data_by_id(persisted_id)
        assert entry.question.content == item_to_insert.question.content
        assert entry.session_id == [item_to_insert.session_id]
        assert (entry.embedding_data == item_to_insert.embedding_data).all()
        assert entry.create_on > time_before_insertion

        for answer in entry.answers:
            matching_answer = next(
                (answer_to_insert for answer_to_insert in item_to_insert.answers if answer.answer == answer_to_insert.answer and answer.answer_type == answer_to_insert.answer_type),
                None
            )
            assert matching_answer is not None

        # The first time you access it, the last_access should be None (since we're accessing it for the first time)
        assert entry.last_access is None

        # ensure that each time we access it, the last_access value is updated
        entry_accessed_again = self.dynamo_cache_storage.get_data_by_id(persisted_id)
        entry_accessed_again.last_access > time_before_insertion

        entry_accessed_thrice = self.dynamo_cache_storage.get_data_by_id(persisted_id)
        entry_accessed_thrice.last_access > entry_accessed_again.last_access

        # next test with a random id that doesn't exist. It should return None
        entry = self.dynamo_cache_storage.get_data_by_id(str(uuid4()))
        assert entry is None

        # now soft delete the existing row in dynamo and make sure it's not returned
        table = self._dynamo_resource().Table("gptcache_questions")
        table.update_item(
            Key = {
                "pk": f"questions#{persisted_id}",
                "id": f"questions#{persisted_id}",
            },
            UpdateExpression = "SET deleted = :deleted",
            ExpressionAttributeValues = {
                ":deleted": "True_1"
            }
        )

        entry = self.dynamo_cache_storage.get_data_by_id(persisted_id)
        assert entry is None

    def test_report_cache(self):
        item_to_insert = {
            "user_question": "how many people in this picture?",
            "cache_question": "how many people are in this picture?",
            "cache_question_id": f"questions#{str(uuid4())}",
            "cache_answer": "5",
            "similarity_value": 0.9,
            "cache_delta_time": 0.5,
        }

        self.dynamo_cache_storage.report_cache(
            user_question = item_to_insert["user_question"],
            cache_question = item_to_insert["cache_question"],
            cache_question_id = item_to_insert["cache_question_id"],
            cache_answer = item_to_insert["cache_answer"],
            similarity_value = item_to_insert["similarity_value"],
            cache_delta_time = item_to_insert["cache_delta_time"],
        )

        # now manually query via boto3 to make sure the data is there
        table = self._dynamo_resource().Table("gptcache_reports")
        items = table.scan()['Items']

        # Since there are session_ids for each item except one, there should be a corresponding session entries in this
        # table. So there should be (3 regular question rows + 2 session rows) = 5 items in the table.
        assert len(items) == 1

        assert items[0]["user_question"] == item_to_insert["user_question"]
        assert items[0]["cache_question"] == item_to_insert["cache_question"]
        assert items[0]["cache_question_id"] == item_to_insert["cache_question_id"]
        assert items[0]["cache_answer"] == item_to_insert["cache_answer"]
        assert items[0]["similarity"] == Decimal(str(item_to_insert["similarity_value"]))
        assert items[0]["cache_delta_time"] == Decimal(str(item_to_insert["cache_delta_time"]))

    def _random_cachedata_without_dependencies(self, session_id = None):
        question_id = uuid4()
        return CacheData(
            question = f"question_{question_id}",
            answers = [f"answer1_for_{question_id}", f"answer2_for_{question_id}"],
            embedding_data = np.random.rand(8).astype(np.float32),
            session_id = session_id,
        )

    def _random_cachedata_with_dependencies(self, session_id = None):
        question_id = uuid4()
        return CacheData(
            question = Question(
                content = f"question_{question_id}",
                deps = [
                    QuestionDep(name = "text", data = "how many people in this picture", dep_type = DataType.STR),
                    QuestionDep(name = "image", data = "object_name", dep_type = DataType.IMAGE_BASE64),
                ],
            ),
            answers = [f"answer1_for_{question_id}"],
            embedding_data = np.random.rand(8).astype(np.float32),
            session_id = session_id,
        )

    def _dynamo_resource(self):
        return awsresource(
            "dynamodb",
            endpoint_url = self._dynamodb_local_endpoint_url,
            aws_access_key_id = "test",
            aws_secret_access_key = "test",
            region_name = self._region_name,
        )

