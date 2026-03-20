import gradio as gr
import os
from huggingface_hub import InferenceClient

# ============================================================
# Lion Translate - Push-to-Talk SEA Language Translation
# Powered by SEA-LION x IMDA
# Built with SEA-LION by AI Singapore | Deployed on HuggingFace
# ============================================================

# Load HF token from environment (set as Space secret: HF_TOKEN)
HF_TOKEN = os.environ.get("HF_TOKEN")

# Whisper client (hf-inference provider)
asr_client = InferenceClient(
    provider="hf-inference",
    api_key=HF_TOKEN,
)

# SEA-LION translation client (featherless-ai provider)
llm_client = InferenceClient(
    provider="featherless-ai",
    api_key=HF_TOKEN,
)

# Supported languages (display name -> Whisper language code)
LANGUAGE_TO_WHISPER = {
    "Malay": "ms",
    "Mandarin (Simplified)": "zh",
    "Tamil": "ta",
    "Bahasa Indonesia": "id",
    "Thai": "th",
    "Vietnamese": "vi",
    "English": "en",
}

LANGUAGES = list(LANGUAGE_TO_WHISPER.keys())


def translate_audio(audio_file, source_language, target_language):
    if audio_file is None:
        return "No audio detected, please try again.", ""

    try:
        # Step 1: Speech-to-Text using Whisper via InferenceClient
        # Note: language hint not passed as it is not supported by the pipeline;
        # Whisper auto-detects the language reliably from the audio.
        asr_result = asr_client.automatic_speech_recognition(
            audio_file,
            model="openai/whisper-large-v3-turbo",
        )

        # Result is an ASROutput object with a .text attribute
        if hasattr(asr_result, "text"):
            transcribed_text = asr_result.text.strip()
        else:
            transcribed_text = str(asr_result).strip()

        if not transcribed_text:
            return "Could not detect speech, please try again.", ""

        # Step 2: Translation using Gemma-SEA-LION-v3 via featherless-ai
        prompt = (
            f"Translate the following {source_language} text to {target_language}. "
            f"Return ONLY the translated text, nothing else.\n"
            f"Text: {transcribed_text}"
        )

        response = llm_client.chat.completions.create(
            model="aisingapore/Gemma-SEA-LION-v3-9B-IT",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.3,
        )

        translated_text = response.choices[0].message.content.strip()

        if not translated_text:
            return transcribed_text, "Translation unavailable, please retry"

        return transcribed_text, translated_text

    except Exception as e:
        err = str(e)
        if "503" in err:
            return "Model is loading, please wait 30 seconds and try again.", ""
        if "429" in err:
            return "Model busy, please try again in 30 seconds.", ""
        return f"Error: {err}", ""


# ============================================================
# Gradio UI
# ============================================================

with gr.Blocks(theme=gr.themes.Soft(), title="Lion Translate") as demo:
    gr.Markdown(
        "# Lion Translate - Powered by SEA-LION x IMDA\n"
        "**Push-to-talk translation for Southeast Asian languages.**"
    )

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Hold to Speak")
            source_selector = gr.Dropdown(
                choices=LANGUAGES,
                value="Mandarin (Simplified)",
                label="Speaking language (FROM):",
            )
            target_selector = gr.Dropdown(
                choices=LANGUAGES,
                value="English",
                label="Translate to (TO):",
            )
            translate_btn = gr.Button("Translate", variant="primary")

        with gr.Column():
            transcription_output = gr.Textbox(label="You said:", interactive=False)
            translation_output = gr.Textbox(label="Translation:", interactive=False)

    translate_btn.click(
        fn=translate_audio,
        inputs=[audio_input, source_selector, target_selector],
        outputs=[transcription_output, translation_output],
    )

    gr.Markdown("---\n*Built with SEA-LION by AI Singapore | Deployed on HuggingFace*")

demo.launch()
