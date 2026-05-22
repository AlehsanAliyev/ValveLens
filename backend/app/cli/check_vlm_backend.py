from __future__ import annotations

import os

from app.pipeline import load_config
from app.vlm_assistant import _assistant_config, vlm_available


def main() -> None:
    config = load_config()
    assistant = _assistant_config(config)
    try:
        from openai import OpenAI  # noqa: F401

        openai_import = True
    except Exception:
        openai_import = False

    ready, reason = vlm_available(config)
    print("VLM backend check:")
    print(f"  assistant.enable_vlm: {assistant['enable_vlm']}")
    print(f"  provider: {assistant['provider']}")
    print(f"  model: {assistant['model'] or '<not configured>'}")
    print(f"  DEEPINFRA_ENDPOINT present: {'yes' if os.environ.get('DEEPINFRA_ENDPOINT') else 'no'}")
    print(f"  OPENAI_API_KEY present: {'yes' if os.environ.get('OPENAI_API_KEY') else 'no'}")
    print(f"  DEEPINFRA_TOKEN present: {'yes' if os.environ.get('DEEPINFRA_TOKEN') else 'no'}")
    print(f"  DEEPINFRA_API_KEY present: {'yes' if os.environ.get('DEEPINFRA_API_KEY') else 'no'}")
    print(f"  openai package import: {'yes' if openai_import else 'no'}")
    print(f"  provider ready: {'yes' if ready else 'no'}")
    print(f"  reason: {reason}")


if __name__ == "__main__":
    main()
