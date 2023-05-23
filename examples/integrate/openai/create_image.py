import time
import base64
from io import BytesIO
from PIL import Image

from gptcache.adapter import openai
from gptcache.processor.pre import get_prompt
from gptcache import cache

from gptcache.embedding import Onnx
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
from gptcache.manager import get_data_manager, CacheBase, VectorBase, ObjectBase


onnx = Onnx()
cache_base = CacheBase('sqlite')
vector_base = VectorBase('milvus', host='localhost', port='19530', collection_name='gptcache_image', dimension=onnx.dimension)
object_base = ObjectBase('local', path='./images')
data_manager = get_data_manager(cache_base, vector_base, object_base)

cache.init(
    pre_embedding_func=get_prompt,
    embedding_func=onnx.to_embeddings,
    data_manager=data_manager,
    similarity_evaluation=SearchDistanceEvaluation(),
    )
cache.set_openai_key()

##################### Create an image with prompt1 ###################
prompt1 = 'a cat sitting besides a dog'
size1 = '256x256'

start = time.time()
response1 = openai.Image.create(
    prompt=prompt1,
    size=size1,
    # response_format='b64_json'
    response_format='b64_json'
    )
end = time.time()
print('Time elapsed:', end - start)

# img = Image.open(response['data'][0]['url'])
# print(img.size)

img_b64_1 = response1['data'][0]['b64_json']
img_bytes_1 = base64.b64decode((img_b64_1))
img_file_1 = BytesIO(img_bytes_1)  # convert image to file-like object
img_1 = Image.open(img_file_1) # convert image to PIL
assert img_1.size == tuple([int(x) for x in size1.split('x')]), \
    'Expected to generate an image of size {size1} but got {img_1.size}.'


#####################  Create an image with prompt2 ##################### 
prompt2 = 'a dog sitting besides a cat'
size2 = '512x512'

start = time.time()
response2 = openai.Image.create(
    prompt=prompt2,
    size=size2,
    # response_format='b64_json'
    response_format='b64_json'
    )
end = time.time()
print('Time elapsed:', end - start)

# img = Image.open(response['data'][0]['url'])
# print(img.size)

img_b64_2 = response2['data'][0]['b64_json']
img_bytes_2 = base64.b64decode((img_b64_2))
img_file_2 = BytesIO(img_bytes_2)  # convert image to file-like object
img_2 = Image.open(img_file_2) # convert image to PIL
assert img_2.size == tuple([int(x) for x in size2.split('x')]), \
    f'Expected to generate an image of size {size2} but got {img_2.size}.'
