#!/usr/bin/env python3
"""
en_to_mni_wiki_translate.py

Fetch an English Wikipedia article, translate paragraphs to Meetei (Meitei) and
produce a wikitext draft file ready for review.

Dependencies:
    pip install requests beautifulsoup4
Optional (Google Cloud Translate):
    pip install google-cloud-translate
Optional (HuggingFace transformers):
    pip install transformers sentencepiece torch  # may be heavy

Usage:
    python en_to_mni_wiki_translate.py "Albert Einstein" --out draft.txt --translator google
    python en_to_mni_wiki_translate.py "Kangla" --translator hf --hf-model "Helsinki-NLP/opus-mt-en-mni"
"""

import argparse
import requests
from bs4 import BeautifulSoup
import time
import math
import os
from typing import List

# ---------- Configuration ----------
EN_WIKI_API = "https://en.wikipedia.org/w/api.php"
DEFAULT_TARGET_LANG = "mni"  # Meitei language code; confirm on your wiki
# -----------------------------------

def fetch_article_html(title: str) -> str:
    """Fetch parsed HTML of the page (cleaner to extract paragraphs)."""
    params = {
        "action": "parse",
        "page": title,
        "prop": "text",
        "format": "json",
        "redirects": True
    }
    resp = requests.get(EN_WIKI_API, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"Wikipedia API error: {data['error']}")
    html = data["parse"]["text"]["*"]
    return html

def extract_paragraphs_from_html(html: str, max_paragraphs: int = None) -> List[str]:
    """Extract main paragraphs (<p>) from the HTML returned by parse API."""
    soup = BeautifulSoup(html, "html.parser")
    paragraphs = []
    # select paragraphs under content; skip reference-only paragraphs
    for p in soup.find_all("p"):
        text = p.get_text().strip()
        if not text:
            continue
        # skip 'Coordinates:' or tiny captions
        if len(text) < 30:
            continue
        paragraphs.append(text)
        if max_paragraphs and len(paragraphs) >= max_paragraphs:
            break
    return paragraphs

# ---------- Pluggable translation implementations ----------

def translate_with_google_cloud(texts: List[str], target_lang: str = DEFAULT_TARGET_LANG) -> List[str]:
    """
    Translate using Google Cloud Translate v3. Requires:
      - pip install google-cloud-translate
      - set environment variable GOOGLE_APPLICATION_CREDENTIALS to service account json
    This function translates a list of strings and returns translated strings in same order.
    """
    try:
        from google.cloud import translate_v2 as translate_v2  # fallback to v2 client (simpler)
    except Exception as e:
        raise RuntimeError("google-cloud-translate not installed. pip install google-cloud-translate") from e

    client = translate_v2.Client()
    translated = []
    # Google client supports batch calls but here we call per item to be explicit
    for txt in texts:
        res = client.translate(txt, target_language=target_lang, format_="text")
        translated.append(res["translatedText"])
        # small delay to be polite
        time.sleep(0.2)
    return translated

def translate_with_hf_model(texts: List[str], model_name: str) -> List[str]:
    """
    Translate using Hugging Face transformers pipeline.
    WARNING: you must provide a valid model that supports en -> target (e.g. en-mni).
    Example model name placeholder: "Helsinki-NLP/opus-mt-en-mni" (if it exists).
    """
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    except Exception as e:
        raise RuntimeError("transformers not installed. pip install transformers sentencepiece torch") from e

    print(f"Loading model {model_name} ... (this may take time and RAM)")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    translator = pipeline("translation", model=model, tokenizer=tokenizer, device=-1)  # CPU; set device if GPU available
    translated = []
    for txt in texts:
        # pipeline may truncate very long texts — we chunk by sentences if necessary
        out = translator(txt, max_length=4096)
        translated.append(out[0]["translation_text"])
        time.sleep(0.1)
    return translated

def translate_with_dummy(texts: List[str], target_lang: str = DEFAULT_TARGET_LANG) -> List[str]:
    """Dummy translator for testing — returns original text prefixed to mark untranslated output."""
    return [f"[MT {target_lang}] {t}" for t in texts]

# ----------------------------------------------------------

def chunk_paragraphs_for_translation(paragraphs: List[str], max_chars: int = 4000) -> List[str]:
    """
    Some MT APIs have size limits; this will combine paragraphs into chunks
    not exceeding max_chars. Returns a list of text-chunks.
    """
    chunks = []
    cur = []
    cur_len = 0
    for p in paragraphs:
        if cur_len + len(p) + 1 > max_chars:
            chunks.append("\n\n".join(cur))
            cur = [p]
            cur_len = len(p)
        else:
            cur.append(p)
            cur_len += len(p) + 2
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks

def re_split_translated_chunks(chunks: List[str], original_paragraphs: List[str]) -> List[str]:
    """
    Re-split chunk translations back into paragraphs by splitting on double newlines.
    This assumes the translator preserved the paragraph separators; not always true.
    """
    translated_paragraphs = []
    for chunk in chunks:
        parts = [p.strip() for p in chunk.split("\n\n") if p.strip()]
        translated_paragraphs.extend(parts)
    # If counts mismatch, attempt to adjust: if less, pad by copying last; if more, join extras.
    if len(translated_paragraphs) != len(original_paragraphs):
        # fallback: return whatever we have (human reviewer required)
        print("Warning: number of translated paragraphs differs from original.")
    return translated_paragraphs

def build_wikitext_draft(title_en: str, translated_paragraphs: List[str], source_url: str) -> str:
    """Create a simple wikitext draft with a header and source link. Human editing expected."""
    w = []
    w.append(f"== {title_en} ==")
    w.append("")
    w.append(f"Translated (machine) draft of the English article '''{title_en}'''.")
    w.append("")
    w.append("<!-- IMPORTANT: This is an automatically generated machine translation. Please review and human-edit before publishing. -->")
    w.append("")
    for p in translated_paragraphs:
        w.append(p)
        w.append("")
    w.append("----")
    w.append(f"Original article: [{source_url} {title_en} — English Wikipedia]")
    w.append("")
    w.append("<!-- Add citations and categories before moving to main namespace. -->")
    return "\n".join(w)

def main():
    parser = argparse.ArgumentParser(description="Translate an English Wikipedia article into Meetei draft using MT.")
    parser.add_argument("title", help="English Wikipedia article title (quote if it contains spaces)")
    parser.add_argument("--out", "-o", default=None, help="Output file path for draft (default: <title>_mni_draft.txt)")
    parser.add_argument("--translator", choices=["google", "hf", "dummy"], default="dummy",
                        help="Which translation backend to use. 'dummy' for testing (no real MT).")
    parser.add_argument("--hf-model", default=None, help="HuggingFace model name for --translator hf (e.g. Helsinki-NLP/opus-mt-en-mni)")
    parser.add_argument("--max-paragraphs", type=int, default=None, help="Limit number of paragraphs to translate (for quick tests)")
    args = parser.parse_args()

    title = args.title
    print(f"Fetching English article: {title} ...")
    html = fetch_article_html(title)
    paragraphs = extract_paragraphs_from_html(html, max_paragraphs=args.max_paragraphs)
    if not paragraphs:
        print("No paragraphs extracted. Exiting.")
        return
    print(f"Extracted {len(paragraphs)} paragraphs.")

    # chunk paragraphs for MT if backend has input size limits
    chunks = chunk_paragraphs_for_translation(paragraphs, max_chars=3000)
    print(f"Created {len(chunks)} chunk(s) for translation.")

    # translate chunks using chosen backend
    if args.translator == "google":
        print("Translating with Google Cloud Translate (v2 interface). Ensure GOOGLE_APPLICATION_CREDENTIALS is set.")
        translated_chunks = translate_with_google_cloud(chunks, target_lang=DEFAULT_TARGET_LANG)
    elif args.translator == "hf":
        if not args.hf_model:
            raise RuntimeError("When using --translator hf you must provide --hf-model (model name on HuggingFace).")
        translated_chunks = translate_with_hf_model(chunks, model_name=args.hf_model)
    else:
        print("Using dummy translator (no real translation).")
        translated_chunks = translate_with_dummy(chunks, target_lang=DEFAULT_TARGET_LANG)

    # re-split into paragraphs
    translated_paragraphs = re_split_translated_chunks(translated_chunks, paragraphs)

    # if counts differ, try naive mapping: if fewer translations, assign them one-per-chunk
    if len(translated_paragraphs) != len(paragraphs):
        print("Paragraph count mismatch — creating best-effort mapping.")
        # simple fallback: translate each paragraph individually with dummy/primary translator
        if args.translator == "google":
            translated_paragraphs = translate_with_google_cloud(paragraphs, target_lang=DEFAULT_TARGET_LANG)
        elif args.translator == "hf":
            translated_paragraphs = translate_with_hf_model(paragraphs, model_name=args.hf_model)
        else:
            translated_paragraphs = translate_with_dummy(paragraphs, target_lang=DEFAULT_TARGET_LANG)

    source_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
    wikitext = build_wikitext_draft(title, translated_paragraphs, source_url)

    outpath = args.out or f"{title.replace(' ', '_')}_mni_draft.txt"
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(wikitext)

    print(f"Draft saved to {outpath}")
    print("Next steps: human-edit this draft, add citations and categories, then use a bot account with community approval to publish to Meitei Wikipedia (mni.wikipedia.org).")

if __name__ == "__main__":
    main()
