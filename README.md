Negotiation Assistant Frontend

This project provides a local GUI frontend to the `it_works.py` negotiation demo. It records audio from your microphone, transcribes it using Whisper, sends the buyer message to a fine-tuned Qwen model (LoRA adapter), applies a simple guardrail, and speaks the seller reply via Piper TTS.

Files added:
- `gui_frontend.py` - A PySimpleGUI-based frontend to set scenario fields and run the voice-driven conversation.
- `requirements.txt` - Packages likely needed to run the demo.

Running

1. (Optional) Create a virtual environment and activate it.

Windows PowerShell example:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Ensure your model files are available:
- LoRA adapter path referenced in `it_works.py` (`MODEL_DIR`) should point to your adapter directory.
- Ensure `PIPER_MODEL` path is valid (the Piper .onnx model file).

4. Run the GUI:

```powershell
python gui_frontend.py
```

