"""Hugging Face Space entrypoint."""

from harbor_hf_space.ui import SPACE_CSS, create_app

demo = create_app()

if __name__ == "__main__":
    demo.launch(show_error=True, css=SPACE_CSS)
