import streamlit as st
from PIL import Image
import os
import io
import base64
from io import BytesIO
import requests

from gptcache import cache
from gptcache.manager import get_data_manager, CacheBase, VectorBase, ObjectBase
from gptcache.adapter import openai
from gptcache.processor.pre import get_prompt
from gptcache.embedding import Onnx
from gptcache.similarity_evaluation import ExactMatchEvaluation

st.title('GPTCache for Image Demo')

@st.cache_resource
def initialize_configuration():
    onnx = Onnx()
    data_manager = get_data_manager(CacheBase('sqlite', sql_url='sqlite:///./local/gptcache10.db'),
                                    VectorBase('faiss', dimension=onnx.dimension, index_path='./local/faiss10.index'),
                                    ObjectBase('local', path='./local'))
    cache.init(
        pre_embedding_func=get_prompt,
        embedding_func=onnx.to_embeddings,
        data_manager=data_manager,
        similarity_evaluation=ExactMatchEvaluation(),
        )
    return data_manager

data_manager = initialize_configuration()

def api_call(text_input, open_ai_key):
    os.environ['CURL_CA_BUNDLE'] = ''
    response = openai.Image.create(
      prompt=text_input,
      n=1,
      size='256x256',
      api_key=open_ai_key
    )
    image_url = response['data'][0]['url']

    is_cached = response.get('gptcache', False)
    if is_cached is False:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
    else:
        img = Image.open(image_url)
    return img, is_cached

def main():
    open_ai_key = st.text_input('OpenAI key:')
    text_input = st.text_input('prompt:')

    if st.button('generate', key='button'):
        try:
            image, is_cached = api_call(text_input, open_ai_key)
            width, height = image.size
            desired_width = 500
            desired_height = int(height * desired_width / width)
            resized_image = image.resize((desired_width, desired_height))
            img_bytes = io.BytesIO()
            resized_image.save(img_bytes, format='PNG')
            img_str = base64.b64encode(img_bytes.getvalue()).decode()

            st.markdown(
                f'<div style="display: flex; justify-content: center;"><img src="data:image/png;base64,{img_str}" \
                        alt="Uploaded Image" width="{desired_width}"></div>',
                unsafe_allow_html=True
            )

            if is_cached:
                st.markdown('<div style="display: flex; align-items: center; justify-content: center; background-color: \
                        green; padding: 10px; color: white; font-weight: bold; border-radius: 5px; margin: 10px auto; \
                        max-width: 100px;">cache</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error('invalid OpenAI API key or inappropriate prompt rejected by OpenAI.')

if __name__ == '__main__':
    main()
