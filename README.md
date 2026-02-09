# Python-Speech-Recognition-Tutorial-

- https://raw.githubusercontent.com/RodrigoMvs123/Python-Speech-Recognition-Tutorial-/main/README.md
- https://github.com/RodrigoMvs123/Python-Speech-Recognition-Tutorial-/blame/main/README.md
- https://www.youtube.com/watch?v=mYUyaKmvu6Y

## Python Speech Recognition Tutorial

Python Speech Recognition Tutorial – Full Course for Beginners

- https://app.assemblyai.com/
- https://www.listennotes.com/podcasts/99-invisible/503-repeat-cIp5H6uJiYO/
- https://www.listennotes.com/podcast-api/docs/?s=side_bottom&id=236f314d3aaf4a9a8025bbabc5183654#get-api-v2-episodes-id

Websockets

OpenAI API

**01Main.py**

```python
import sys
from api_communication import * 


# upload 
# upload_endpoint = "https://api.assemblyai.com/v2/upload"
# transcript_endpoint = "https://api.assemblyai.com/v2/transcript"

# headers = {'authorization': "API_KEY_ASSEMBLYAI"}

filename = sys.argv[1]


audio_url = upload(filename)
save transcript(audio_url, filename)
```

```bash
(base) misraturp@msras-MacBook-Pro-2 Simple_Speech_Recognition % python main py output.wav 
(base) misraturp@msras-MacBook-Pro-2 Simple_Speech_Recognition %

(base) misraturp@msras-MacBook-Pro-2 Simple_Speech_Recognition % python main py output.wav 
(base) misraturp@msras-MacBook-Pro-2 Simple_Speech_Recognition %
```

Waiting for 30 seconds...

**02Main.py**
```python
import json 
from yt_extractor import get_audio_url, get_video_infos
from api import save_transcript 

def save_video_sentiments(url):
    video_infos = get_video_infos(url)
    audio_url = get_audio_url(video_infos)
    title = video_infos["title"]
    title = title.strip().replace(" ", "_")
    title = "data/" + title
    save_transcript(audio_url, title, sentiment_analysis=True)

    if __name__ == "__main__":
       # save_video_sentiments(https://www.youtube.com/watch?v=g5ymJNLURRI)
```

```bash
        (base) misraturp@msras-MacBook-Pro-2 Simple_Speech_Recognition % python main py
        (base) misraturp@msras-MacBook-Pro-2 Simple_Speech_Recognition %
```

```python
        with open ("data/iphone_13_Review:_Pros_and_Cons_sentiments.json", "r") as f:
            data = json.load(f)

            positives = []
            negatives = []
            neutrals = []
            for result in data: 
                text = result["text"]
                if result ["sentiment"] == "POSITIVE":
                   positives.append(text)
                elif result ["sentiment"] == "NEGATIVE":
                   negatives.append(text)
                else: neutrals.append(text)


        n_pos = len(positives)
        n_neg = len(negatives)
        n_neut = len(neutrals)

        print("Num positives:", n_pos)
        print("Num negatives:", n neg)
        print("Num neutrals:", n neut)


        r = n_pos / (n_pos + n_neg)
        print(f"Positive radio: {r:.3f}")
```

```bash
        (base) misraturp@msras-MacBook-Pro-2 Simple_Speech_Recognition % python main py
        (base) misraturp@msras-MacBook-Pro-2 Simple_Speech_Recognition %
```

**03Main.py**
```python
from api_communication import *         
import streamlit as st
import json 

st.title(' Welcome to my application that creates podcast summaries')
episode_id = st.sidebar.text_imput('Please input an episode id')
button = st.sidebar.button('Get podcast summary!', on_click=save_transcript, args=(episode_id,))

def get_clean_time(start_ms):
    seconds = int(start_ms / 1000) %60)
    minutes = int(start_ms / (1000 * 60)) % 60)
    hours = int(start_ms / (1000 * 60 * 60)) % 24
    if hours > 0:
        start_t = f'{hours:02d}:{minutes:02d}:{seconds:02d}'
    else
        start_t = f'{minutes:02d}:{seconds:02d}'

        return start_t 

if button: 
    text_filename = episode_id + '_chapters.json'
    with open(filename, 'r') as f:
        data = json.load(f)

        chapters = data['chapters']
        podcast_title = data[' podcast_title']
        episode_title = data['episode_title']
        thumbnail = data['episode_thumbnail']

    st.header(f'{podcast_title}-{episode_title}')
    st.image(thumbnail)
    for chp in chapters: 
        with st.expander(chp['gist'] + '-' + get_clean_time(chp['start'])):
        chp['summary']


# save transcript(...)
```

```bash
(base)misraturp@msras-MacBook-Pro-2 Podcast sumarizarition % python 03main.py

(base)misraturp@msras-MacBook-Pro-2 Podcast sumarizarition % python 03main.py
(base)misraturp@msras-MacBook-Pro-2 Podcast sumarizarition %

(base)misraturp@msras-MacBook-Pro-2 Podcast sumarizarition % cd Desktop/Podcast\sumarizarition  
(base)misraturp@msras-MacBook-Pro-2 Podcast\sumarizarition  % streamlit run 03main.py 
(base)misraturp@msras-MacBook-Pro-2 
```

**04Main.py**
```python
 import pyaudio 
    import websockets 
    import asyncio 
    import base64 
    import json 
    from api_secrets import API_KEY_ASSEMBLYAI 
    from openai_helper import ask_computer
     
     FRAMES_PER-BUFFER = 3200 
     FORMAT = pyaudio.paInt16
     CHANNELS = 1 
     RATE = 16000

     p = pyaudio.Py.Audio()

     stream = p.open(
         format=FORMAT,
         channels=CHANNELS,
         rate=RATE,
         input=True,
         frames_per_buffer=FRAMES_PER_BUFFER
     )

     URL = "wss://api.assemblyai.com/v2/realtime/ws?sample_rate=16000"

     async def send_receive():
         async with websockets.connect(
             URL,
             ping_timeout=20,
             ping_interval=5,
             extra_headers={"Authorization":API_KEY_ASSEMBLYAI }
         ) as _ws:
            await asyncio.sleep(0.1)
            session_begins = await _ws.recv()
            print(session_begins)
            print("Sending messages") 

            async def send(): 
                while True:
                   try:
                       data = stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
                       data = base.64.b64encode(data).decode("utf-8")
                       json_data = json_dumps({"audio_data": data})
                       await _ws.send(json_data)
                    except websockets.execptions.ConnectionCloseError as e:
                        print(e)
                        assert e.code == 4008
                        break
                    except Exception as e:
                        assert False, "Not a websocket 4008 error"
                    await asyncio.sleep(0.01)   

            async def receive():
                while True: 
                    try:
                      result_str = await _ws.recv()
                      result = json.loads(result_str)
                      prompt = result["text"]
                      if prompt and result["message_type"] == "FinalTranscript":
                          print("Me:", prompt)
                          response = ask_computer(prompt)
                          print("Bot:", response)

                    except websockets.execptions.ConnectionCloseError as e:
                        print(e)
                        assert e.code == 4008
                        break
                    except Exception as e:
                        assert False, "Not a websocket 4008 error"

            send_result, receive_result = await asyncio.gather(send(), receive())

           asyncio.run(send_receive())
```

```bash
           ~/p/python-speec-rec/05-realtime-openai python 04main.py
```

**api_communication.py**
```python
           import requests
from api_secrets import API_KEY_ASSEMBLYAI, API_KEY_LISTENNOTES
import time
import json 
import print

# upload 
transcript_endpoint = "https://api.assemblyai.com/v2/transcript"
assemblyai_headers = {'authorization': "API_KEY_ASSEMBLYAI"}

listennotes_episode_endpoint = "https://listen-api.listennotes.com/api/v2/episodes"
listennotes_headers = {'X-ListenAPI-Key': "API_KEY_LISTENNOTES "}
```

```bash
(base) misraturp@msras-MacBook-Pro-2 Simple_Speech_Recognition % python main py output.wav
# Hi, I'am Rodrigo. This is a test one, two, three.
(base) misraturp@msras-MacBook-Pro-2 Simple_Speech_Recognition %
```

```python
def get_episode_audio_url(episode_id):
   url = listennotes_episode_endpoint + '/' + episode_id 
   response = requests.request('GET', url, headers = listennotes_headers)

   data = response.json()
   #pprint.pprint(data)

   audio_url = data['audio']
   episode_thumbnail = data['thumbnail']
   podcast_title = data['podcast']['title']
   episode_title = data['title']
  
   
   return audio_url, episode_thumbnail, episode_title, podcast_title


# Transcribe 
def transcribe(audio_url, auto_chapters): 
    transcript_request = { 
        'audio_url': audio_url
        'auto_chapters' : auto_chapters 
         }
    transcript_response = requests.post(transcript_endpoint, json=transcript_request, headers=assemblyai_headers)
    job_id = transcript_response.json()['id']
    return job_id
```

```bash
(base) misraturp@msras-MacBook-Pro-2 Simple_Speech_Recognition % python main py output.wav
(base) misraturp@msras-MacBook-Pro-2 Simple_Speech_Recognition % python main py 
(base) misraturp@msras-MacBook-Pro-2 Simple_Speech_Recognition %
```

```python
# Poll 
def poll(transcript_id): 
    polling_endpoint = transcript_endpoint + '/' + transcript_id 
    polling_response = requests.get(transcript_endpoint, headers=headers)
    return polling_response.json()
```

```bash
(base) misraturp@msras-MacBook-Pro-2 Simple_Speech_Recognition % python main py 
(base) misraturp@msras-MacBook-Pro-2 Simple_Speech_Recognition %
```

```python
def get_transcription_result_url(url, auto_chapters)
    transcript_id = transcribe(url, auto_chapters)
    while True: 
        data = poll(transcript_id)
        if data['status'] == 'completed': 
            return data, None 
        elif data['status'] == 'error': 
            return data, data['error']

    print('Waiting 60 seconds...')
    time.sleep(60)
```

```bash
(base) misraturp@msras-MacBook-Pro-2 Simple_Speech_Recognition % python main py output.wav 
(base) misraturp@msras-MacBook-Pro-2 Simple_Speech_Recognition %
```

```python
# Save transcript 

def save transcript(episode_id): 
    audio_url, episode_thumbnail, episode_title, podcast_title = get_episode_audio_url(episode_id)
    data, error = get_transcription_result_url(audio_url, auto_chapters=True) 

    pprint.pprint(data)

    if data: 
       text_filename = episode_id + ".txt"
        with open('text_filename', "w") as f:
           f.write(data)['text'] 

        chapters_filename = episode_id + '_chapters.json'
        with open(chapters_filename, 'w') as f:
            chapters = data['chapters']
            episode_data = {'chapters':chapters}
            episode_data['episode_thumbnail'] = episode_thumbnail
            episode_data['episode_title'] = episode_thumbnail
            episode_data['podcast_title'] = episode_thumbnail

             json.dump(episode_data, f )
             print('Transcript saved')
             return True 
    elif error 
        print("Error!!", error)
       return False
```

```bash
(base) misraturp@msras-MacBook-Pro-2 Simple_Speech_Recognition % python main py What_is_NLP .mp4 
(base) misraturp@msras-MacBook-Pro-2 Simple_Speech_Recognition %
```

**Api_Secrets.py**
```python
API_KEY_ASSEMBLYAI = "..."
API_KEY_LISTENNOTES = "..."
```

**Openai_Helper.py**

```bash
~/p/python-speec-rec/05-realtime-openai pip install openai
```

```python
import openai
from api_secrets import API_KEY_OPENAI

openai.api_key = API_KEY_OPENAI

def ask_computer(prompt):
    response = openai.Completion.create( 
        model="text-davinci-002",
        prompt=prompt,
        max_tokens=100) 
    return response["choises"][0]["text"]
```

**Simple_Speech_Recognition.py**

```bash
   (base) misraturp@msras-MacBook-Pro-2 Simple_Speech_Recognition % python main py output.wav
# Hi, I'am Rodrigo. This is a test one, two, three. 

(base) misraturp@msras-MacBook-Pro-2 Simple_Speech_Recognition %
```

- https://app.assemblyai.com/

```
...
```

**api.py**
```python
dev save_transcript(url, title, sentiment_analysis=false):
data, error = get_transcript_result_url(url, sentiment_analysis)
```

**Load_mp3.py**

```bash
~/p/python-speech-rec/01-basics  brew install ffmpeg
~/p/python-speech-rec/01-basics  pip install pydub
```

```python
from pydub import AudioSegment 

audio = AudioSegment.from_wav("Rodrigo.wav")

# Increase the volume by 6dB
audio = audio + 6 

audio = audio * 2 

audio = audio.fade_in (2000)

audio.export("mashup.mp3", format="mp3")

audio2 = AudioSegment.from_mp3("mashup.mp3")
print("done")
```

```bash
~/p/python-speech-rec/01-basics   python load_mp3.py
```

**output.wav**
```
encoding                            pcm_s16le            decode:  Stop, Play 
format                              s16                  volume
number_of_channel                   1(mono)              seekbar 
sample_rate                         16000
file_size                           160044 byte 
duration                            5s 
analyze                                                  settings
```

**Plot_Audio.py**
```python
import wave
import matplotlib.pyplot as plt 
import numpy as np 

obj = wave.open("Rodrigo.wave", "rb")

sample_freq = obj.getframerate()
n_samples = obj.getnframes()
signal_wave = obj.readframes(-1)

obj.close()

t_audio = n_samples / samples_freq 

print(t_audio)
```

```bash
~/p/python-speech-rec/01-basics    python plot_audio.py
5.0 ...
```

```python
signal_array = np.frombuffer(signal_wave, dtype=np.int16)

times = np.linspace(0, t_audio, num=n_samples)

plt.figure(figsize=(15,5))
plt.plot(times, signal_array)
plt.title("Audio Signal")
plt.ylabel("Signal wave") 
plt.xlabel("Time(s)")
plt.xlim(0, t_audio)
plt.show()
```

```bash
~/p/python-speech-rec/01-basics    python plot_audio.py
```

**Record_Mic.py**
```python
import pyaudio
import wave 

FRAMES_PER_BUFFER = 3200 
FORMAT = pyaudio.paInt16
CHANNELS = 1 
RATE = 16000

p = pyaudio.PyAudio() 

stream = p.open(
    format=FORMAT,
    channels=CHANNELS, 
    rate=RATE,
    input=True,
    frames_per_buffer=FRAMES_PER_BUFFER
)

print("Start Recording")
seconds = 5
frames = []
for i in range (0, int(RATE/FRAMES_PER_BUFFER*seconds)):
    data =  stream.read(FRAMES_PER_BUFFER)
    frames.append(data)

stream.stop_stream()
stream.close()
p.terminate()

obj = wave.open("output.wav", "wb")
obj.setnchannels(CHANNELS)
obj.setsampwidth(p.get_sample_size(FORMAT))
obj.setframerate(RATE)
obj.writeframes(b"".join(frames))
obj.close()
```

```bash
~/p/python-speech-rec/01-basics  python record_mic.py 
// Hi I am rodrigo this is a test 123.
// output.wav
```

**Rodrigo_new_wav.wav**
```
encoding                            pcm_s16le            decode:  Stop, Play 
format                              s16                  volume
number_of_channel                   1(mono)              seekbar 
sample_rate                         16000
file_size                           160044 byte 
duration                            5s 
analyze                                                  settings
```

**Rodrigo.wav**
```
encoding                            pcm_s16le            decode:  Stop, Play 
format                              s16                  volume
number_of_channel                   1(mono)              seekbar 
sample_rate                         16000
file_size                           160044 byte 
duration                            5s 
analyze                                                  settings
```

**Wave_exemple.py**

Audio File Formats

- .mp3
- .flac
- .wav

```python
import wave
```

Audio Signal Parameters

- number of channels
- sample width
- framerate/sample_rate: 44,100 hz
- number of frames
- value of a frame

```python
obj = wave.open("Rodrigo.wav", "rb")

print("Number_of_channels", obj.getnchannels())
print("sample width", obj.getsampwidth())
print("frame rate", obj.getframerate())
print("Number of frames", obj.getnframes())
pritn("parameters", obj.getparams())

t_audio = obj.getnframes() / obj.getframerate()
print_(t_audio)
```

```bash
~/p/python-speech-rec/01-basics    python wave_exemple.py 
Number of Channels ... 

~/p/python-speech-rec/01-basics    python wave_exemple.py 
Number of Channels ... 
5s
```

```python
frames = obj.readframes(-1)
print(type(frames), type(frames[0]))
print (len(frames) / 2)

obj.close()
```

```bash
~/p/python-speech-rec/01-basics    python wave_exemple.py 
Number of Channels ... 
80000
```

```python
obj_new = wave.open("Rodrigo_new.wav", "wb")
obj_new.setnchannels(1)
obj_new.setsampwidth(2)
obj_new.setframerate(16000.0)

obj.new.writeframes(frames)

obj.new.close()
```

```bash
~/p/python-speech-rec/01-basics    python wave_exemple.py 
Number of Channels ...
```

**yt_Extractor.py**
```python
import youtube_dl 

ydl = youtube_dl.YoutubeDL()

def get_videos_infos(url):
    with ydl:
        result = ydl.extract_info(
            url, 
            downloand=false
        )
    if "entries" in result: 
        return result["entries"][0]
    return result 

def get_audio_url(video_info):
    for f in print["formats"]:
        if f ["ext"] == "m4a":
           return f["url"]  

if __name__ == "__main__":
    video_info = get_video_infos(https://www.youtube.com/watch?v=g5ymJNLURRI)
    audio_url = get_audio_url(video_info)
    print(audio_url)
```

```bash
/p/python-speech-python-rec/03-sentiment-analyses  yt_extract.py 
/p/python-speech-python-rec/03-sentiment-analyses
```
