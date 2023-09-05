from functools import reduce
from random import randint, SystemRandom
from datetime import datetime
from typing import List, Optional, Dict
from decimal import Decimal
from gptcache.manager.scalar_data.base import CacheStorage, CacheData, Question, QuestionDep, Answer
from gptcache.utils import import_boto3
from gptcache.utils.log import gptcache_log
from concurrent.futures import ThreadPoolExecutor, wait
import numpy as np
import math

import_boto3()
from boto3.session import Session as AwsSession
from boto3.dynamodb.conditions import Key as DynamoKey, Attr as DynamoAttr
from boto3.dynamodb.types import Binary as DynamoBinary

class DynamoStorage(CacheStorage):
    """
    DynamoDB storage using AWS's boto3 library.

    :param aws_access_key_id: AWS access key ID. If not specified, boto3 will use the default resolution behavior.
    :type host: str

    :param aws_secret_access_key: AWS secret access key. If not specified, boto3 will use the default resolution behavior.
    :type aws_secret_access_key str

    :param aws_region_name: AWS region name. If not specified, boto3 will use the default resolution behavior.
    :type aws_region_name: str

    :param aws_profile_name: AWS profile name. If not specified, boto3 will use the default resolution behavior.
    :type aws_profile_name: str

    :param aws_endpoint_url: AWS endpoint URL. This is normally handled automatically but is exposed to allow overriding for testing purposes
                            (using something like LocalStack or dynamoDB-local for example).
    :type aws_endpoint_url: str
    """

    max_cardinality_suffix = 10

    def __init__(
        self,
        aws_access_key_id: str = None,
        aws_secret_access_key: str = None,
        aws_region_name: str = None,
        aws_profile_name: str = None,
        aws_endpoint_url: str = None,
    ):
        self._aws_session = AwsSession(
            aws_access_key_id = aws_access_key_id,
            aws_secret_access_key = aws_secret_access_key,
            region_name = aws_region_name,
            profile_name = aws_profile_name,
        )
        self._dynamo = self._aws_session.resource(
            "dynamodb",
            endpoint_url=aws_endpoint_url,
        )
        self.create()

    def create(self):
        # NOTE: We use PAY_PER_REQUEST billing mode as per AWS's recommendations for unpredicatble workloads since it's
        #       being used as a cache and so we won't know how many reads/writes we'll need to perform ahead of time.
        recommended_billing_mode = "PAY_PER_REQUEST"

        if not self._does_table_already_exist_and_is_active("gptcache_questions"):
            self._dynamo.create_table(
                TableName = "gptcache_questions",
                KeySchema = [
                    {"AttributeName": "pk", "KeyType": "HASH"},
                    {"AttributeName": "id", "KeyType": "RANGE"},
                ],
                AttributeDefinitions = [
                    {"AttributeName": "pk", "AttributeType": "S"},
                    {"AttributeName": "id", "AttributeType": "S"},

                    # You might be wondering why the 'deleted' attribute is a string value, not a BOOL.
                    # This is because we need to efficiently be able to query items that are soft-deleted. This is normally
                    # done with a GSI (GlobalSecondaryIndex). However, GSIs don't support the BOOL type due to low cardinality.
                    # So we use a string instead.
                    {"AttributeName": "deleted", "AttributeType": "S"},
                ],
                GlobalSecondaryIndexes = [
                    {
                        "IndexName": "gsi_items_by_type",
                        "KeySchema": [{"AttributeName": "id", "KeyType": "HASH"}],
                        "Projection": {"ProjectionType": "ALL"},
                    },
                    {
                        "IndexName": "gsi_questions_by_deletion_status",
                        "KeySchema": [{"AttributeName": "deleted", "KeyType": "HASH"}],
                        "Projection": {"ProjectionType": "ALL"},
                    },
                ],
                BillingMode = recommended_billing_mode,
            )

        if not self._does_table_already_exist_and_is_active("gptcache_reports"):
            self._dynamo.create_table(
                TableName = "gptcache_reports",
                KeySchema = [
                    {"AttributeName": "id", "KeyType": "HASH"},
                ],
                AttributeDefinitions = [
                    {"AttributeName": "id", "AttributeType": "S"},
                ],
                BillingMode = recommended_billing_mode,
            )

    def batch_insert(self, all_data: List[CacheData]) -> List[str]:
        """
        Inserts a list of CacheData objects into the DynamoDB table and returns the ids of the inserted rows
        """
        table = self._dynamo.Table("gptcache_questions")
        table.wait_until_exists()
        ids = []

        with table.batch_writer() as batch:
            for item in all_data:
                item_deps = item.question.deps if isinstance(item.question, Question) and item.question.deps is not None else []
                new_id_without_prefix = self._generate_id()
                new_id = f"questions#{new_id_without_prefix}"
                ids.append(new_id_without_prefix)
                creation_time = item.create_on if item.create_on is not None else datetime.utcnow()

                batch.put_item(
                    Item = self._strip_all_props_with_none_values({
                        "pk": new_id,
                        "id": new_id,
                        "question": self._question_text(item.question),
                        "answers": [self._serialize_answer(answer) for answer in item.answers],
                        "deps": [self._serialize_question_deps(dep) for dep in item_deps],
                        #"session_id": item.session_id if item.session_id is not None else None,
                        "create_on": creation_time.isoformat(timespec="microseconds"),
                        "last_access": item.last_access.isoformat(timespec="microseconds") if item.last_access is not None else None,
                        "embedding_data": DynamoBinary(item.embedding_data.tobytes()) if item.embedding_data is not None else None,
                        "deleted": self._serialize_deleted_value(False),
                    })
                )

                if item.session_id:
                    batch.put_item(
                        Item = self._strip_all_props_with_none_values({
                            "pk": new_id,
                            "id": f"sessions#{item.session_id}",
                            "question": self._question_text(item.question),
                        })
                    )
        return ids

    def get_data_by_id(self, key: str) -> Optional[CacheData]:
        table = self._dynamo.Table("gptcache_questions")
        table.wait_until_exists()

        key_with_prefix = f"questions#{key}"

        # get all sessions that contain this question
        sessions_containing_question = table.query(
            KeyConditionExpression = DynamoKey("pk").eq(key_with_prefix) & DynamoKey("id").begins_with("sessions#"),
        )["Items"]

        # then get the question info itself
        response = table.get_item(
            Key={"pk": key_with_prefix, "id": key_with_prefix},
        )

        if "Item" not in response or self._deserialize_deleted_value(response["Item"]["deleted"]):
            return None

        # after it's accessed, we need to update that particular item's last_access timestamp
        table.update_item(
            Key={"pk": key_with_prefix, "id": key_with_prefix},
            UpdateExpression="SET last_access = :last_access",
            ExpressionAttributeValues={
                ":last_access": datetime.utcnow().isoformat(timespec="microseconds"),
            },
        )

        cache_data = self._response_item_to_cache_data(response["Item"])
        cache_data.session_id = [session["id"].split("#")[1] for session in sessions_containing_question]
        return cache_data

    def mark_deleted(self, keys: str):
        table = self._dynamo.Table("gptcache_questions")
        table.wait_until_exists()

        with table.batch_writer() as batch:
            for key in keys:
                batch.put_item(
                    Item={
                        "pk": f"questions#{key}",
                        "id": f"questions#{key}",
                        "deleted": self._serialize_deleted_value(True),
                    }
                )

    def clear_deleted_data(self):
        # Since the 'deleted' attribute is a string, we need to query for all items that have a 'deleted' value of
        # 'True_*'. Then we can call delete on them.
        table = self._dynamo.Table("gptcache_questions")
        table.wait_until_exists()

        # Build out all possible values for the 'deleted' attribute and query it in 1 go. It cannot be done as one
        # network call unfortunately due to limitations of DynamoDB. The DB only allows running queries on the GSI
        # with a SINGLE, concrete value for the partition key.
        #
        # To address this, we're gonna spin up a thread pool and query concurrently
        with ThreadPoolExecutor(max_workers = 10) as executor:
            futures = []
            for i in range(1, self.max_cardinality_suffix + 1):
                futures.append(executor.submit(
                    table.query,
                    IndexName = "gsi_questions_by_deletion_status",
                    KeyConditionExpression = DynamoKey("deleted").eq(f"True_{i}")
                ))

            completed_responses, incomplete_responses = wait(futures, timeout = 10)

            if len(incomplete_responses) > 0:
                gptcache_log.error(
                    """
                    Unable to complete deletion of all soft-deleted items due to request timeout.
                    Some items may remain in the cache. %s",
                    """,
                    incomplete_responses,
                )

        soft_deleted_entries_per_partition = [response.result()["Items"] for response in completed_responses]
        soft_deleted_entries = reduce(lambda x, y: x + y, soft_deleted_entries_per_partition)

        keys = [entry["id"] for entry in soft_deleted_entries]

        with table.batch_writer() as batch:
            for key in keys:
                batch.delete_item(
                    Key={"pk": key, "id": key},
                )

    def get_ids(self, deleted: bool = True) -> List[str]:
        table = self._dynamo.Table("gptcache_questions")
        table.wait_until_exists()

        def run_scan_operation(last_evaluated_key):
            filter_expression = (
                DynamoAttr("id").begins_with("questions#") & DynamoAttr("deleted").begins_with(f"{deleted}_")
            )

            if last_evaluated_key is None:
                return table.scan(
                    Select = "SPECIFIC_ATTRIBUTES",
                    ProjectionExpression = "pk",
                    FilterExpression = filter_expression,
                )
            return table.scan(
                Select = "SPECIFIC_ATTRIBUTES",
                ProjectionExpression = "pk",
                FilterExpression = filter_expression,
                ExclusiveStartKey = last_evaluated_key,
            )

        all_responses = self._fetch_all_pages(run_scan_operation)
        all_items = reduce(lambda x, y: x + y, [response["Items"] for response in all_responses])
        all_ids = [int(item["pk"].replace("questions#", "")) for item in all_items]
        return all_ids

    def count(self, state: int = 0, is_all: bool = False) -> int:
        table = self._dynamo.Table("gptcache_questions")
        table.wait_until_exists()
        key_condition_expression = DynamoAttr("id").begins_with("questions#")

        if not is_all and state == 0:
            key_condition_expression &= DynamoAttr("deleted").begins_with("False_")
        elif not is_all and state != 0:
            key_condition_expression &= DynamoAttr("deleted").begins_with("True_")

        # TODO: find out if specifying a "COUNT" select type results in a single page or not. For now, assume the worst
        #       and attempt pagination anyway.
        def run_scan_operation(last_evaluated_key):
            if last_evaluated_key is None:
                return table.scan(
                    IndexName = "gsi_questions_by_deletion_status",
                    FilterExpression = key_condition_expression,
                    Select = "COUNT",
                )

            return table.scan(
                IndexName = "gsi_questions_by_deletion_status",
                FilterExpression = key_condition_expression,
                ExclusiveStartKey = last_evaluated_key,
                Select = "COUNT",
            )

        all_responses = self._fetch_all_pages(run_scan_operation)
        return sum([response["Count"] for response in all_responses])

    def add_session(self, question_id: str, session_id: str, session_question: str):
        table = self._dynamo.Table("gptcache_questions")
        table.wait_until_exists()

        table.put_item(
            Item = {
                "pk": f"questions#{question_id}",
                "id": f"sessions#{session_id}",
                "question": session_question,
            },
        )

    def list_sessions(self, session_id = None, key = None) -> List[str]:
        table = self._dynamo.Table("gptcache_questions")
        table.wait_until_exists()

        if session_id and key:
            response = table.query(
                IndexName = "gsi_items_by_type",
                ProjectionExpression = "id",
                KeyConditionExpression = DynamoKey("id").eq(f"sessions#{session_id}"),
                FilterExpression = DynamoAttr("pk").eq(f"questions#{key}"),
            )
        elif session_id:
            response = table.query(
                IndexName = "gsi_items_by_type",
                ProjectionExpression = "id",
                KeyConditionExpression = DynamoKey("id").eq(f"sessions#{session_id}"),
            )
        elif key:
            response = table.query(
                ProjectionExpression = "id",
                KeyConditionExpression = DynamoKey("pk").eq(f"questions#{key}") & DynamoKey("id").begins_with("sessions#"),
            )
        else:
            def run_scan_operation(last_evaluated_key):
                filter_expression = DynamoAttr("id").begins_with("sessions#")

                if last_evaluated_key is None:
                    return table.scan(
                        ProjectionExpression = "id",
                        FilterExpression = filter_expression,
                    )
                return table.scan(
                    ProjectionExpression = "id",
                    FilterExpression = filter_expression,
                    ExclusiveStartKey = last_evaluated_key,
                )

            response = self._fetch_all_pages(run_scan_operation)

            # since these are paginated results, merge all items in each response to a new dict
            response = reduce(lambda x, y: { "Items": x["Items"] + y["Items"] }, response)

        # since sessions can be the shared across multiple items, we need to dedupe the results
        return list({ item["id"].replace("questions#", "") for item in response["Items"] })

    def delete_session(self, keys: List[str]):
        table = self._dynamo.Table("gptcache_questions")
        table.wait_until_exists()

        # first find all items with that session_id
        # unfortunately, there is no "batch get" operation on a GSI. So we need to spin up a thread pool and query
        # concurrently
        with ThreadPoolExecutor(max_workers = 10) as executor:
            futures = []
            for key in keys:
                futures.append(executor.submit(
                    table.query,
                    IndexName = "gsi_items_by_type",
                    Select = "SPECIFIC_ATTRIBUTES",
                    ProjectionExpression = "pk, id",
                    KeyConditionExpression = DynamoKey("id").eq(f"sessions#{key}")
                ))

            completed_responses, incomplete_responses = wait(futures, timeout = 10)

            if len(incomplete_responses) > 0:
                gptcache_log.error(
                    "Unable to query all questions in session due to request timeout. %s",
                    incomplete_responses,
                )

        items_per_session = [response.result()["Items"] for response in completed_responses]
        all_relevant_items = reduce(lambda x, y: x + y, items_per_session)

        # now we need to delete all items with those keys to clear out all session data.
        with table.batch_writer() as batch:
            for item in all_relevant_items:
                batch.delete_item(
                    Key={"pk": item["pk"], "id": item["id"]},
                )

    def report_cache(
        self,
        user_question,
        cache_question,
        cache_question_id,
        cache_answer,
        similarity_value,
        cache_delta_time,
    ):
        table = self._dynamo.Table("gptcache_reports")
        table.wait_until_exists()

        table.put_item(
            Item={
                "id": str(self._generate_id()),
                "user_question": user_question,
                "cache_question": cache_question,
                "cache_question_id": cache_question_id,
                "cache_answer": cache_answer,
                "similarity": (
                    # DynamoDB doesn't support floats; only decimals
                    Decimal(str(similarity_value))
                    if isinstance(similarity_value, float)
                    else similarity_value
                ),
                "cache_delta_time": Decimal(str(cache_delta_time)),
            }
        )

    def close(self):
        pass

    def _does_table_already_exist_and_is_active(self, table_name: str) -> bool:
        try:
            self._dynamo.Table(table_name).table_status in ["ACTIVE", "UPDATING"]
        except self._dynamo.meta.client.exceptions.ResourceNotFoundException:
            return False
        return True

    def _wait_until_table_exists(self, table_name: str):
        """
        When tables are created in DynamoDB, they are not immediately available as table creation is asynchronous.
        This function will block until the table is available in order to ensure we don't attempt to perform any
        operations on a table that doesn't exist yet.
        """
        self._dynamo.Table(table_name).wait_until_exists()


    def _response_item_to_cache_data(self, question_resp: Dict) -> Optional[CacheData]:
        deps = question_resp["deps"] if "deps" in question_resp else []
        return CacheData(
            question = Question(
                content = question_resp["question"],
                deps = [self._deserialize_question_dep(dep) for dep in deps]
            ),
            answers = [self._deserialize_answer(answer) for answer in question_resp["answers"]],
            embedding_data = np.frombuffer(question_resp["embedding_data"].value, dtype=np.float32),
            create_on = datetime.fromisoformat(question_resp["create_on"]),
            last_access = datetime.fromisoformat(question_resp["last_access"]) if "last_access" in question_resp else None,
        )

    def _fetch_all_pages(self, scan_fn) -> List[Dict]:
        """
        We often have to resort to scan operations in Dynamo which could result in a lot of data being returned. To ensure,
        we get all the data, we need to paginate through the results. This function handles that for us.

        :param scan_fn: The function to call to perform the scan operation. It must take in a last_evaluated_key to pass
                        along to Dynamo and return a Dynamo response object

        see: https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/dynamodb/table/scan.html
        """
        all_responses = []
        last_evaluated_key = None

        while True:
            response = scan_fn(last_evaluated_key)
            all_responses.append(response)

            if "LastEvaluatedKey" in response and len(response["LastEvaluatedKey"].keys()) > 0:
                last_evaluated_key = response["LastEvaluatedKey"]
            else:
                break

        return all_responses

    def _question_text(self, question) -> str:
        return question if isinstance(question, str) else question.content

    def _generate_id(self):
        """
        Generates a unique ID for a row. Unfortunately, we cannot use something like uuid4() as scalar storage
        implmentations need to return an integer based id.

        It seems the underlying adapter that uses this class converts the ids to C longs which are 64-bit signed
        so we'll use that as our upper bound.

        We use systemrandom for better randomness.
        """
        return SystemRandom().randint(1, math.pow(2, 63) - 1)

    def _strip_all_props_with_none_values(self, obj: Dict) -> Dict:
        sanitized = {}
        for k, v in obj.items():
            if v is not None:
                sanitized[k] = v
        return sanitized

    def _serialize_deleted_value(self, value: bool, suffix_value = randint(1, max_cardinality_suffix)) -> str:
        """
        We need to be able to query and filter on the 'deleted' attribute. However, in order for us to use this value
        as an indexable value, it cannot be a BOOL (DynamoDB doesn't support it due to low cardinaility).

        To increase the cardinality, we first convert the boolean to a string and append a random integer.

        see: https://aws.amazon.com/blogs/database/how-to-design-amazon-dynamodb-global-secondary-indexes/
        """
        return f"{value}_{suffix_value}"

    def _serialize_answer(self, answer: Answer) -> Dict:
        return { "answer": answer.answer, "answer_type": answer.answer_type.value }

    def _serialize_question_deps(self, dep: QuestionDep) -> Dict:
        return { "name": dep.name, "data": dep.data, "dep_type": dep.dep_type.value }

    def _deserialize_deleted_value(self, value: str) -> bool:
        return value.startswith("True_")

    def _deserialize_question_dep(self, dep: Dict) -> QuestionDep:
        return QuestionDep(name = dep["name"], data = dep["data"], dep_type = int(dep["dep_type"]))

    def _deserialize_answer(self, answer: Dict) -> Answer:
        return Answer(answer = answer["answer"], answer_type = int(answer["answer_type"]))
