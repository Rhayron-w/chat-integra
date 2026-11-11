# -*- coding: utf-8 -*-
"""Interface web.  Uso: python -m streamlit run app_streamlit.py"""
import uuid

import streamlit as st

from chatbot import Chatbot


@st.cache_resource
def load_bot():
    return Chatbot()


bot = load_bot()
st.set_page_config(page_title=f"{bot.bot_name} | {bot.cfg['empresa']}", page_icon="💬")
st.title(f"💬 {bot.bot_name}")
st.caption(f"Atendimento virtual · {bot.cfg['empresa']}")

QUICK = bot.cfg["atalhos"]


def reset():
    st.session_state.session_id = f"web-{uuid.uuid4().hex[:8]}"
    st.session_state.messages = [{"role": "assistant", "content": bot.greeting()}]


if "session_id" not in st.session_state:
    reset()

clicked = None
with st.sidebar:
    st.subheader("Atalhos")
    for q in QUICK:
        if st.button(q, use_container_width=True):
            clicked = q
    st.divider()
    st.caption(f"Fallback com IA: {'ligado' if bot.use_fallback else 'desligado'}")
    if st.button("Nova conversa"):
        reset()
        st.rerun()

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Digite sua mensagem...") or clicked
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    answer = bot.respond(prompt, session_id=st.session_state.session_id)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
