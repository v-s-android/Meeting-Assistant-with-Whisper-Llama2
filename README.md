# Meeting-Assistant-with-Whisper-Llama2
The Project uses Open Ai's. Whisper and Meta Llama 2 for capturing coversations in Meetings

Consider you're attending a business meeting where all conversations are being captured by an advanced AI application. This application not only transcribes the discussions with high accuracy but also provides a concise summary of the meeting, emphasizing the key points and decisions made.

In our project, we'll use OpenAI's Whisper to transform speech to text. Next, we'll use IBM Watson's AI to summarize and find key points. We'll make an app with Hugging Face Gradio as the user interface.

Preparing the environment and install required libraries:
```
pip3 install virtualenv 
virtualenv my_env # create a virtual environment my_env
source my_env/bin/activate # activate my_env
```
```
# installing required libraries in my_env
pip install --force-reinstall "setuptools<70" transformers==4.36.0 torch==2.1.1 gradio==5.23.2 langchain==0.0.343 ibm_watson_machine_learning==1.0.335 huggingface-hub==0.28.1
```
