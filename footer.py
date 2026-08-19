"""
Shared Streamlit footer component.
"""

import streamlit as st


FOOTER_HTML = """
<div style="text-align:center; padding:20px 0; border-top:1px solid #e0e0e0; margin-top:40px;">
    <p style="margin:0 0 10px; font-size:14px; color:#666; font-family:sans-serif;">
        Research assistance by
        <a href="https://www.perplexity.ai" target="_blank" rel="noopener noreferrer"
           style="color:#20808d; text-decoration:none; font-weight:bold;">
           Perplexity
        </a>
    </p>
    <p style="margin:0; font-size:12px; color:#888; font-family:sans-serif; line-height:1.5;">
        MLB analytics and prediction research<br>
        All content is for informational purposes only and does not constitute betting or financial advice.
        Wager responsibly.
    </p>
</div>
"""


def add_perplexity_footer() -> None:
    """Render the shared Perplexity footer."""
    st.markdown(FOOTER_HTML, unsafe_allow_html=True)


def add_betting_oracle_footer() -> None:
    """Backward-compatible alias for older page imports."""
    add_perplexity_footer()