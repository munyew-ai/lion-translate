# Lion Translate - Powered by SEA-LION x IMDA

Push-to-talk translation app for Southeast Asian languages, built with SEA-LION by AI Singapore.

## Live Demo

Try it on HuggingFace Spaces: https://huggingface.co/spaces/munyew/lion-translate

## What It Does

1. User records audio via microphone (push-to-talk)
2. Speech is transcribed using OpenAI Whisper Large v3 Turbo (via HuggingFace Inference API)
3. Transcribed text is translated using Gemma-SEA-LION-v3-9B-IT (via Featherless AI)
4. Translation is displayed instantly

## Supported Languages

- Malay
- Mandarin (Simplified)
- Tamil
- Bahasa Indonesia
- Thai
- Vietnamese
- English

## Tech Stack

- Frontend/UI: Gradio 5
- Speech-to-Text: OpenAI Whisper Large v3 Turbo (HuggingFace hf-inference provider)
- Translation: aisingapore/Gemma-SEA-LION-v3-9B-IT (Featherless AI provider)
- Deployment: HuggingFace Spaces (CPU Basic, free tier)

## Setup (Local)

```bash
pip install -r requirements.txt
export HF_TOKEN=your_hf_token_here
python app.py
```

Set HF_TOKEN as an environment variable with your HuggingFace API token that has Inference API access.

## Project Structure

```
lion-translate/
├── app.py              # Main Gradio app
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Built By

Darren Loh (IMDA) - National Multimodal LLM Programme (NMLP)

Built with SEA-LION by AI Singapore | Deployed on HuggingFace
