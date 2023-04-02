import whisper
import gradio as gr
import warnings
import openai

warnings.filterwarnings("ignore")

# Use your API key to authenticate
openai.api_key = "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"

# model = whisper.load_model("base")
model = whisper.load_model("tiny")
model_size = "tiny"

model.device

def transcribe(audio, lang="Korean"):
    # load audio and pad/trim it to fit 30 seconds
    x=whisper.load_audio(audio)
    audio_x=whisper.pad_or_trim(x)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio_x).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)

    # decode the audio
    #options = whisper.DecodingOptions()
    # options = whisper.DecodingOptions(fp16=False)
    options = whisper.DecodingOptions(task="translation")
    result = whisper.decode(model, mel, options)
    result_text = result.text

    # Pass the generated text to Audio
    # Use the openai API to generate a response
    response = openai.Completion.create(
        engine="ada",
        prompt=result_text,
        max_tokens=1024,
        n=1,
        temperature=0.5
    ).choices[0].text

    out_result = response
    print(out_result)

    return [result_text, out_result]

delay_slider = gr.inputs.Slider(minimum=1, maximum=5, default=1.2, label="Rate of transcription")
output_1 = gr.Textbox(label="Speech to Text")
output_2 = gr.Textbox(label="ChatGPT Output")

gr.Interface(
    title = 'OpenAI Whisper and ChatGPT ASR Gradio Web UI',
    fn=transcribe,
    inputs=[
        gr.inputs.Audio(source="microphone", type="filepath"),
        delay_slider,
    ],
    outputs=[output_1, output_2],
    # server_url="http://localhost:8080",  # Change this to your local address
    live=True).launch()

Python formatter를 설정하는 방법은 다음과 같습니다.

1. IDE나 텍스트 에디터에서 Python 파일을 엽니다.
2. 파일 상단에 있는 `import` 문 다음에 `from __future__ import print_function, division`을 추가합니다. 이는 Python 2에서 Python 3 스타일의 print 함수와 나눗셈 연산을 사용할 수 있도록 합니다.
3. 코드를 작성할 때 PEP 8 스타일 가이드를 따르도록 합니다. 이는 코드의 가독성을 높이기 위한 규칙이며, 대부분의 Python 개발자들이 따르는 표준입니다.
4. 코드 포맷터를 사용하여 코드를 자동으로 포맷팅하도록 합니다. 다양한 코드 포맷터가 있지만, 예를 들어 Black, YAPF, autopep8 등을 사용할 수 있습니다. 이들은 코드에 일관성을 유지하고, 효율적으로 포맷팅해주는 기능을 제공합니다.
5. 코드를 저장하고, 팀 내에서 공유할 때, 일관된 스타일을 유지할 수 있도록 코드 리뷰를 수행합니다. 이는 코드의 가독성과 유지보수성에 큰 영향을 미치며, 팀 전체적으로 일관된 코드 스타일을 유지할 수 있도록 합니다.

위와 같은 방법으로 Python formatter를 설정하면, 코드 작성의 효율성과 일관성을 유지할 수 있습니다.