"""
MathVision UI Component Library
Polished presentation components for university project redesign.
"""
import os
import html as html_lib
import streamlit as st

# Project Constants (Empty fields remain hidden per Project Rules)
PROJECT_TITLE = "MathVision"
PROJECT_TAGLINE = "Photograph handwritten mathematics. Check it. Solve it."
INSTITUTION = ""
TEAM_MEMBERS = ""
GUIDE = ""
YEAR = ""
LOGO_FILE = ""

PALETTE = {
    "primary": "#1F3A5F",
    "accent": "#E0A030",
    "background": "#F6F7FB",
    "text": "#1B2430",
    "white": "#FFFFFF",
    "border": "#E2E6EF"
}


def load_css():
    """Load custom stylesheet once into the Streamlit app."""
    css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui_styles.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            css = f.read()
        html(f"<style>{css}</style>")


def html(s: str):
    """
    Render raw HTML safely by stripping leading whitespace from every line,
    filtering out empty lines (preventing CommonMark 4-space indented code blocks),
    and rendering via st.html (or st.markdown fallback).
    """
    cleaned_lines = [line.lstrip() for line in s.strip().splitlines() if line.strip()]
    cleaned = "\n".join(cleaned_lines)
    if hasattr(st, "html"):
        st.html(cleaned)
    else:
        st.markdown(cleaned, unsafe_allow_html=True)


def render_header():
    """Render top brand header with title, tagline, and optional metadata."""
    meta_html = ""
    if INSTITUTION or YEAR:
        parts = [html_lib.escape(p) for p in (INSTITUTION, YEAR) if p]
        meta_html = f"<div style='font-size: 0.9rem; color: #536271; margin-bottom: 4px;'>{' • '.join(parts)}</div>"

    safe_title = html_lib.escape(PROJECT_TITLE)
    safe_tagline = html_lib.escape(PROJECT_TAGLINE)

    html(f"""
<div style="margin-bottom: 1.5rem;">
{meta_html}
<h1 class="mv-title">{safe_title}</h1>
<div class="mv-tagline">{safe_tagline}</div>
</div>
""")


def render_stepper(current_step: int = 1):
    """
    Render 3-step progress indicator:
    1: Upload Photo, 2: Check Reading, 3: Solve
    """
    steps = [
        (1, "Upload"),
        (2, "Check reading"),
        (3, "Solve")
    ]
    parts = ['<div class="mv-stepper">']
    for num, label in steps:
        if num < current_step:
            cls = "mv-step completed"
        elif num == current_step:
            cls = "mv-step active"
        else:
            cls = "mv-step"
        safe_label = html_lib.escape(label)
        parts.append(f"""
<div class="{cls}">
<div class="mv-step-num">{num}</div>
<div class="mv-step-label">{safe_label}</div>
</div>
""")
    parts.append('</div>')
    html("".join(parts))


def render_empty_photo_tips():
    """Friendly empty-state guidance card with four photo tips."""
    html("""
<div class="mv-tips-card">
<div style="font-weight: 700; color: #1F3A5F; font-size: 1.05rem; margin-bottom: 12px;">
Tips for best recognition
</div>
<div class="mv-tip-item">
<span class="mv-tip-bullet">✦</span>
<span><strong>Plain paper:</strong> Use unlined or lightly lined white paper with high contrast.</span>
</div>
<div class="mv-tip-item">
<span class="mv-tip-bullet">✦</span>
<span><strong>Space around:</strong> Leave a clear margin around the handwritten equation.</span>
</div>
<div class="mv-tip-item">
<span class="mv-tip-bullet">✦</span>
<span><strong>Good lighting:</strong> Keep the page evenly illuminated and avoid hand shadows.</span>
</div>
<div class="mv-tip-item">
<span class="mv-tip-bullet">✦</span>
<span><strong>Single line:</strong> Focus on one mathematical expression or equation per photo.</span>
</div>
</div>
""")


def render_skeleton():
    """Render neutral text-free placeholder block to preserve layout during inference."""
    html('<div class="mv-skeleton" aria-hidden="true"></div>')


def render_callout(message: str, kind: str = "info"):
    """
    Render accessible styled callout container with exact text.
    kind: 'warning', 'error', 'info'
    """
    icon_map = {
        "warning": "⚠️",
        "error": "⛔",
        "info": "ℹ️"
    }
    icon = icon_map.get(kind, "ℹ️")
    cls = f"mv-callout mv-callout-{kind}"
    safe_msg = html_lib.escape(message)
    html(f"""
<div class="{cls}" role="alert">
<span class="mv-callout-icon" aria-hidden="true">{icon}</span>
<div style="flex: 1; line-height: 1.45;">{safe_msg}</div>
</div>
""")


def render_footer():
    """Render persistent bottom reminder strip and project imprint."""
    imprint = html_lib.escape(PROJECT_TITLE)
    if YEAR:
        imprint += f" • {html_lib.escape(YEAR)}"

    html(f"""
<div style="margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid #E2E6EF; text-align: center; color: #64748B; font-size: 0.88rem;">
<p style="margin-bottom: 6px;">Always compare the recognized equation with your photo before trusting an answer.</p>
<p style="font-size: 0.8rem; color: #94A3B8; margin: 0;">{imprint}</p>
</div>
""")


def render_svg_pipeline():
    """Render inline SVG flowchart showing the MathVision 6-stage architecture."""
    html("""
<div style="width: 100%; overflow-x: auto; margin: 20px 0; text-align: center;">
<svg viewBox="0 0 920 90" width="100%" height="90" style="max-width: 920px; font-family: -apple-system, sans-serif;">
<defs>
<linearGradient id="gradNavy" x1="0%" y1="0%" x2="100%" y2="100%">
<stop offset="0%" stop-color="#1F3A5F" />
<stop offset="100%" stop-color="#2D5284" />
</linearGradient>
<linearGradient id="gradGold" x1="0%" y1="0%" x2="100%" y2="100%">
<stop offset="0%" stop-color="#E0A030" />
<stop offset="100%" stop-color="#F2BA52" />
</linearGradient>
<marker id="arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M 0 1 L 10 5 L 0 9 z" fill="#8F9CAE" />
</marker>
</defs>
<rect x="5" y="15" width="125" height="58" rx="8" fill="url(#gradNavy)" />
<text x="67" y="42" fill="#FFFFFF" font-size="12" font-weight="700" text-anchor="middle">1. Photo Input</text>
<text x="67" y="58" fill="#CBD5E1" font-size="9.5" text-anchor="middle">Camera / Upload</text>
<line x1="130" y1="44" x2="155" y2="44" stroke="#8F9CAE" stroke-width="2" marker-end="url(#arrow)" />

<rect x="158" y="15" width="125" height="58" rx="8" fill="url(#gradNavy)" />
<text x="220" y="42" fill="#FFFFFF" font-size="12" font-weight="700" text-anchor="middle">2. Crop &amp; Clean</text>
<text x="220" y="58" fill="#CBD5E1" font-size="9.5" text-anchor="middle">Ink Isolation</text>
<line x1="283" y1="44" x2="308" y2="44" stroke="#8F9CAE" stroke-width="2" marker-end="url(#arrow)" />

<rect x="311" y="15" width="128" height="58" rx="8" fill="url(#gradNavy)" />
<text x="375" y="42" fill="#FFFFFF" font-size="12" font-weight="700" text-anchor="middle">3. Recognition</text>
<text x="375" y="58" fill="#CBD5E1" font-size="9.5" text-anchor="middle">Uni-MuMER (Qwen3)</text>
<line x1="439" y1="44" x2="464" y2="44" stroke="#8F9CAE" stroke-width="2" marker-end="url(#arrow)" />

<rect x="467" y="15" width="125" height="58" rx="8" fill="url(#gradNavy)" />
<text x="529" y="42" fill="#FFFFFF" font-size="12" font-weight="700" text-anchor="middle">4. Editable LaTeX</text>
<text x="529" y="58" fill="#CBD5E1" font-size="9.5" text-anchor="middle">Review Guardrails</text>
<line x1="592" y1="44" x2="617" y2="44" stroke="#8F9CAE" stroke-width="2" marker-end="url(#arrow)" />

<rect x="620" y="15" width="125" height="58" rx="8" fill="url(#gradNavy)" />
<text x="682" y="42" fill="#FFFFFF" font-size="12" font-weight="700" text-anchor="middle">5. SymPy Solver</text>
<text x="682" y="58" fill="#CBD5E1" font-size="9.5" text-anchor="middle">AST Verification</text>
<line x1="745" y1="44" x2="770" y2="44" stroke="#8F9CAE" stroke-width="2" marker-end="url(#arrow)" />

<rect x="773" y="15" width="138" height="58" rx="8" fill="url(#gradGold)" />
<text x="842" y="42" fill="#1B2430" font-size="12" font-weight="700" text-anchor="middle">6. Answer &amp; Steps</text>
<text x="842" y="58" fill="#3E2B08" font-size="9.5" text-anchor="middle">Rendered Working</text>
</svg>
</div>
""")
