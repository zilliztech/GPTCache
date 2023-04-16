import unittest
import mock
import os
import requests
from pathlib import Path
import numpy as np
from tempfile import TemporaryDirectory

from gptcache.manager.object_data.local_storage import LocalObjectStorage
from gptcache.manager.object_data.s3_storage import S3Storage
from gptcache.manager import ObjectBase


class TestLocal(unittest.TestCase):
    def test_normal(self):
        with TemporaryDirectory(dir="./") as root:
            o = LocalObjectStorage(root)
            data = b'My test'
            fp = o.put(data)
            self.assertTrue(Path(fp).is_file())
            self.assertEqual(o.get(fp), data)
            self.assertEqual(o.get_access_link(fp), fp)
            o.delete([fp])
            self.assertFalse(Path(fp).is_file())


class TestS3(unittest.TestCase):
    def test_normal(self):
        access_key = os.environ.get('AWS_ACCESS_KEY_ID')
        secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
        bucket = os.environ.get('BUCKET')
        endpoint = os.environ.get('ENDPOINT')        
        if access_key is None or secret_key is None or bucket is None:
            return
        o = S3Storage(bucket, 'gptcache', access_key, secret_key, endpoint)
        data = b'My test'
        fp = o.put(data)
        self.assertEqual(o.get(fp), data)
        link = o.get_access_link(fp)
        self.assertEqual(requests.get(link, verify=False).content, data)
        o.delete([fp])
        self.assertIsNone(o.get(fp))

class TestBase(unittest.TestCase):
    def test_local(self):
        with TemporaryDirectory(dir="./") as root:
            o = ObjectBase("local", path = root)
            data = b'My test'
            fp = o.put(data)
            self.assertTrue(Path(fp).is_file())
            self.assertEqual(o.get(fp), data)
            self.assertEqual(o.get_access_link(fp), fp)
            o.delete([fp])
            self.assertFalse(Path(fp).is_file())

    def test_s3(self):
        with mock.patch("boto3.Session") as mock_session:
            o = ObjectBase("s3", bucket="", path_prefix="",
                           access_key="", secret_key="")
            data = b"My test"
            fp = o.put(data)
            o.get(fp)
            o.get_access_link(fp)
