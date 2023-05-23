import time

from gptcache import cache
from gptcache.adapter import replicate
from gptcache.embedding import Timm, Onnx
from gptcache.manager import get_data_manager, CacheBase, VectorBase, ObjectBase
from gptcache.processor.pre import get_input_image_file_name
from gptcache.similarity_evaluation.np import NumpyNormEvaluation

timm = Timm('resnet18')
onnx = Onnx()
cache_base = CacheBase('sqlite')
vector_base = VectorBase('faiss', dimension=timm.dimension)
object_base = ObjectBase('local', path='./objects')
data_manager = get_data_manager(cache_base, vector_base, object_base)

cache.init(
    pre_embedding_func=get_input_image_file_name,
    data_manager=data_manager,
    embedding_func=timm.to_embeddings,
    similarity_evaluation=NumpyNormEvaluation(enable_normal=True, question_embedding_function=onnx.to_embeddings)
    )
    

image_path = '../../../docs/GPTCache.png'


# run replicate clinet with gptcache
start = time.time()
question1 = "what is in the image?"
question2 = "What can you see in the image?"

output = replicate.run(
    "andreasjansson/blip-2:4b32258c42e9efd4288bb9910bc532a69727f9acd26aa08e175713a0a857a608",
    input={
        "image": open(image_path, 'rb'),
        "question": question1}
    )
end = time.time()
print('Answer:', output)
print('Time elapsed 1:', end - start)

start = time.time()
output = replicate.run(
    "andreasjansson/blip-2:4b32258c42e9efd4288bb9910bc532a69727f9acd26aa08e175713a0a857a608",
    input={
        "image": open(image_path, 'rb'),
        "question": question2}
    )
end = time.time()
print('Answer:', output)
print('Time elapsed 2:', end - start)