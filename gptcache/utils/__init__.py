__all__ = ['import_pymilvus', 'import_huggingface_hub',
           'import_faiss', 'import_chromadb',
           'import_sqlalchemy', 'import_sql_client',
           'import_huggingface', 'import_torch',
           'import_sbert', 'import_onnxruntime',
           'import_cohere', 'import_fasttext'
           ]
from .dependency_control import prompt_install


# pylint: disable=unused-import
# pylint: disable=ungrouped-imports
# pragma: no cover
def import_pymilvus():
    try:
        import pymilvus
    except ModuleNotFoundError as e:
        prompt_install('pymilvus')
        import pymilvus

def import_sbert():
    try:
        import sentence_transformers
    except ModuleNotFoundError:
        prompt_install('sentence-transformers')
        import sentence_transformers

def import_cohere():
    try:
        import cohere
    except ModuleNotFoundError:
        prompt_install('cohere')
        import cohere

def import_fasttext():
    try:
        import fasttext
    except ModuleNotFoundError:
        prompt_install('fasttext')
        import fasttext

def import_huggingface():
    try:
        import transformers
    except ModuleNotFoundError as e:
        prompt_install('transformers')
        import transformers

def import_torch():
    try:
        import torch
    except ModuleNotFoundError:
        prompt_install('torch')
        import torch

def import_huggingface_hub(): 
    try:
        import huggingface_hub
    except ModuleNotFoundError as e:
        prompt_install('huggingface-hub')
        import huggingface_hub

def import_onnxruntime():
    try:
        import onnxruntime
    except ModuleNotFoundError as e:
        prompt_install('onnxruntime')
        import onnxruntime

def import_faiss():
    try:
        import faiss
    except ModuleNotFoundError as e:
        prompt_install('faiss-cpu==1.6.5')
        import faiss


def import_chromadb():
    try:
        import chromadb
    except ModuleNotFoundError as e:
        prompt_install('chromadb')
        import chromadb


def import_sqlalchemy():
    try:
        import sqlalchemy
    except ModuleNotFoundError as e:
        prompt_install('sqlalchemy')
        import sqlalchemy


def import_postgresql():
    try:
        import psycopg2
    except ModuleNotFoundError as e:
        prompt_install('psycopg2-binary')
        import psycopg2


def import_pymysql():
    try:
        import pymysql
    except ModuleNotFoundError as e:
        prompt_install('pymysql')
        import pymysql


# `brew install unixodbc` in mac
# and install PyODBC driver.
def import_pyodbc():
    try:
        import pyodbc
    except ModuleNotFoundError as e:
        prompt_install('pyodbc')
        import pyodbc


# install cx-Oracle driver.
def import_cxoracle():
    try:
        import cx_Oracle
    except ModuleNotFoundError as e:
        prompt_install('cx_Oracle')
        import cx_Oracle


def import_sql_client(db_name):
    if db_name == 'postgresql':
        import_postgresql()
    elif db_name in ['mysql', 'mariadb']:
        import_pymysql()
    elif db_name == 'sqlserver':
        import_pyodbc()
    elif db_name == 'oracle':
        import_cxoracle()
