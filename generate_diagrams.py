"""Generate Mermaid diagrams as PNG files using the mermaid.ink service."""

import base64
import requests
from pathlib import Path

DIAGRAMS = {
    "evaluation_pipeline": """flowchart TD
    A["Load LAMBADA Dataset"] --> B["Sample N Passages"]
    B --> C["Extract Target Word - last word of passage"]
    C --> D["Prepare Prompt - passage without last word"]
    D --> E{"Select Model"}
    E --> F["Phi-3 Mini 3.8B"]
    E --> G["Gemma 2 9B"]
    E --> H["Llama 3.1 8B"]
    F --> I["OpenRouter API"]
    G --> I
    H --> I
    I --> J["Get Prediction"]
    J --> K{"Compare with Target Word"}
    K -->|Match| L["Correct"]
    K -->|No Match| M["Incorrect"]
    L --> N["Aggregate Metrics"]
    M --> N
    N --> O["Save Results JSON"]
    O --> P["Dashboard Visualization"]""",

    "lambada_task": """flowchart LR
    A["Full Passage"] --> B["Split"]
    B --> C["Context: all words except last"]
    B --> D["Target: last word"]
    C --> E["Send to LLM"]
    E --> F["LLM Predicts word"]
    F --> G{"Match?"}
    G -->|"prediction == target"| H["Correct"]
    G -->|"prediction != target"| I["Incorrect"]""",

    "architecture_comparison": """flowchart TB
    subgraph PHI["Phi-3 Mini 3.8B"]
        P1["32 Transformer Layers"]
        P2["GQA 32H / 8KV"]
        P3["SwiGLU Activation"]
        P4["128K Context RoPE"]
        P5["Textbook-Quality Data"]
        P1 --> P2 --> P3 --> P4 --> P5
    end
    subgraph GEMMA["Gemma 2 9B"]
        G1["42 Transformer Layers"]
        G2["Local + Global Attention"]
        G3["GeGLU Activation"]
        G4["8K Context RoPE"]
        G5["Knowledge Distillation"]
        G1 --> G2 --> G3 --> G4 --> G5
    end
    subgraph LLAMA["Llama 3.1 8B"]
        L1["32 Transformer Layers"]
        L2["GQA 32H / 8KV"]
        L3["SwiGLU Activation"]
        L4["128K Context ext. RoPE"]
        L5["15T Token Pretraining"]
        L1 --> L2 --> L3 --> L4 --> L5
    end""",

    "project_workflow": """flowchart TD
    START(["Project Start"]) --> DS["LAMBADA Dataset"]
    DS --> PREP["Data Preparation"]
    PREP --> CONFIG["Configure Models"]
    CONFIG --> API["OpenRouter API"]
    API --> EVAL["Evaluation and Metrics"]
    EVAL --> VIZ["Visualization Dashboard"]
    VIZ --> REPORT["Report Generation"]
    REPORT --> END(["Complete"])""",
}


def save_mermaid_png(mermaid_code: str, output_path: str) -> bool:
    try:
        encoded = base64.urlsafe_b64encode(mermaid_code.encode("utf-8")).decode("utf-8")
        url = f"https://mermaid.ink/img/{encoded}"
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200 and len(resp.content) > 100:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(resp.content)
            return True
        print(f"  FAIL ({resp.status_code}) for {output_path}")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main():
    out_dir = Path("diagrams")
    out_dir.mkdir(exist_ok=True)

    for name, code in DIAGRAMS.items():
        fpath = out_dir / f"{name}.png"
        print(f"Generating {fpath} ...")
        ok = save_mermaid_png(code, str(fpath))
        print(f"  {'OK' if ok else 'FAILED'}")


if __name__ == "__main__":
    main()
