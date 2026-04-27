"""My-LM — local LLM, image generation, and QLoRA training playground.

Public sub-packages:

- ``mylm.imaging`` — SDXL pipeline helpers, ESRGAN upscaler, ADetailer-style face fix
- ``mylm.rag``     — BookMind retrieval-augmented generation (Atlas Vector Search)
- ``mylm.io``      — Newline-delimited JSON helpers for the bridge protocol

The bridge entry points (``llm_bridge``, ``image_bridge``, ``train_bridge``,
``book_bridge``) live under ``scripts/`` so they can be spawned by the Electron
main process via ``python scripts/<name>.py``.
"""

__version__ = "0.1.0"
