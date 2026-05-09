# My-LM System Architecture

This diagram is generated from [architecture.yaml](architecture.yaml) and rendered with Mermaid for instant viewing on GitHub. The same YAML is the input to the [VSDX skill](../../.github/skills/VISIO.md) for producing an editable Visio file.

```mermaid
flowchart LR
    user["User"]
    renderer["Renderer<br/>(TS + webpack)"]
    preload["Preload<br/>(window.My)"]
    main["Electron Main<br/>(Node)"]

    subgraph bridges["Bridge managers (Node)"]
        llmBridge["llmBridge"]
        imageBridge["imageBridge"]
        trainBridge["trainBridge"]
        bookBridge["bookBridge"]
    end

    subgraph subprocs["Python subprocesses"]
        llmPy["scripts/llm_bridge.py"]
        imagePy["scripts/image_bridge.py"]
        trainPy["scripts/train_bridge.py"]
        bookPy["scripts/book_bridge.py"]
    end

    mylm["src/mylm<br/>(Python lib)"]
    torch["PyTorch + CUDA<br/>(GPU)"]
    hf["Hugging Face Hub"]
    atlas[("MongoDB Atlas<br/>vector search")]

    user -->|interacts| renderer
    renderer -->|window.My| preload
    preload -->|IPC| main
    main --> llmBridge
    main --> imageBridge
    main --> trainBridge
    main --> bookBridge

    llmBridge -->|spawn / NDJSON| llmPy
    imageBridge -->|spawn / NDJSON| imagePy
    trainBridge -->|spawn / NDJSON| trainPy
    bookBridge -->|spawn / NDJSON| bookPy

    llmPy --> mylm
    imagePy --> mylm
    trainPy --> mylm
    bookPy --> mylm

    mylm -->|torch.cuda| torch
    mylm -->|downloads| hf
    bookPy -->|"$vectorSearch"| atlas
```

## Notes

- **Bridges are long-lived**: the Node-side `*Bridge` modules each manage one Python subprocess for the lifetime of the app. Screen switches do not interrupt running ops.
- **Wire format**: newline-delimited JSON over stdin/stdout.
- **GPU contention**: chat / image / training all want the whole 6 GB. Don't run them concurrently.
- **BookMind path**: bookPy goes directly to MongoDB Atlas via `pymongo`'s `$vectorSearch`; mylm provides the embedding helpers.
