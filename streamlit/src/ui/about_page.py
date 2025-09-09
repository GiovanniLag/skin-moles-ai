from pathlib import Path
import base64
import mimetypes
import re
import streamlit as st

def _md_image_to_html(match: re.Match, base_dir: Path) -> str:
    alt_text = match.group("alt") or ""
    url      = match.group("url") or ""
    title    = match.group("title") or ""
    title_attr = f' title="{title}"' if title else ""

    # If already a URL or data URI, keep as-is
    if url.startswith(("http://", "https://", "data:")):
        return (
            f'<img src="{url}" alt="{alt_text}"{title_attr} '
            'style="display:block;margin:1rem auto;max-width:100%;height:auto;" />'
        )

    # Resolve local file relative to the README folder
    img_path = (base_dir / url).resolve()
    if not img_path.exists():
        # Fall back to original Markdown if file missing (don’t hard-crash)
        return f'![{alt_text}]({url}{f" \"{title}\"" if title else ""})'

    # Guess a MIME type; default to octet-stream if unknown
    mime, _ = mimetypes.guess_type(str(img_path))
    mime = mime or "application/octet-stream"

    data = base64.b64encode(img_path.read_bytes()).decode("ascii")
    data_uri = f"data:{mime};base64,{data}"

    return (
        f'<img src="{data_uri}" alt="{alt_text}"{title_attr} '
        'style="display:block;margin:1rem auto;max-width:100%;height:auto;" />'
    )

def _replace_image_with_html(markdown: str, base_dir: Path) -> str:
    """
    Replace Markdown image syntax with <img> tags.
    - Local relative paths are inlined as base64 data URIs.
    - Remote URLs are kept as URLs.
    """
    # Matches: ![alt](url "title")
    pattern = re.compile(
        r'!\[(?P<alt>[^\]]*)\]\('
        r'(?P<url>[^)\s]+)'               # no spaces until ) to keep it simple
        r'(?:\s+"(?P<title>[^"]*)")?'     # optional "title"
        r'\)',
        flags=re.UNICODE,
    )
    return pattern.sub(lambda m: _md_image_to_html(m, base_dir), markdown)

def render() -> None:
    st.title("About")

    readme_path = Path(__file__).resolve().parent.parent.parent.parent / "README.md"
    if readme_path.exists():
        readme_content = readme_path.read_text(encoding="utf-8")
        # Convert MD images -> HTML <img>, base64-inlined if local
        readme_content = _replace_image_with_html(readme_content, readme_path.parent)
        st.markdown(readme_content, unsafe_allow_html=True)
    else:
        st.markdown(
            """
            **DermaNet** is a skin-lesion classification Streamlit app using a ResNet-style CNN with SE blocks and GeM pooling.
            The repo also includes BYOL self-supervised pretraining code (course project, University of Milano-Bicocca, 2025).
            """
        )
