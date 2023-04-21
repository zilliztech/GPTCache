import streamlit as st
import os
import uuid

from gptcache import cache
from gptcache.manager import get_data_manager, CacheBase, VectorBase, ObjectBase
from gptcache.adapter import openai
from gptcache.processor.pre import get_file_name
from gptcache.embedding import Data2VecAudio
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation


@st.cache_resource
def initialize_configuration():
    data2vec = Data2VecAudio()
    data_manager = get_data_manager(CacheBase('sqlite', sql_url='sqlite:///./local/gptcache20.db'),
                                    VectorBase('faiss', dimension=data2vec.dimension, index_path='./local/faiss20.index'),
                                    ObjectBase('local', path='./local'))
    cache.init(
        pre_embedding_func=get_file_name,
        embedding_func=data2vec.to_embeddings,
        data_manager=data_manager,
        similarity_evaluation=SearchDistanceEvaluation(),
        )
    return data_manager

data_manager = initialize_configuration()

def api_call(audio_bytes, open_ai_key):
    os.environ['OPENAI_API_KEY'] =  open_ai_key
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
    cache.set_openai_key()
    transcript = openai.Audio.transcribe('whisper-1', audio_bytes, api_key=open_ai_key)
    is_cached = transcript.get('gptcache', False)
    return transcript['text'], is_cached

def main():

    st.title('GPTCache for Audio Demo')
    open_ai_key = st.text_input('OpenAI key')
    audio_file = st.file_uploader('Choose an audio file (.mp3, .wav, or .ogg)', type=['mp3', 'wav', 'ogg'])

    if st.button('generate', key='button'):
        file_extension = os.path.splitext(audio_file.name)[1]  # Get the extension of the uploaded file
        random_filename = str(uuid.uuid4()) + file_extension
        with open(random_filename, 'wb') as f:
            f.write(audio_file.getbuffer())
        audio_file_handler = open(random_filename, 'rb')
        text, is_cached = api_call(audio_file_handler, open_ai_key)
        st.write(text)
        os.remove(random_filename)

        if is_cached:
            st.markdown('<div style="display: flex; align-items: center; justify-content: center; \
                    background-color: green; padding: 10px; color: white; font-weight: bold; border-radius: \
                    5px; margin: 10px auto; max-width: 100px;">cache</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
