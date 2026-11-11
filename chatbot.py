"""Núcleo do chatbot de atendimento de ERP: intenções, memória, fluxos
(lead comercial, abertura/consulta de chamado), histórico e fallback com IA."""
import json
import os
import random
import re
import sqlite3
import threading
from datetime import datetime

import torch

from nlp_utils import NeuralNet, bag_of_words, tokenize

DEFAULT_CFG = {
    "empresa": "Sua Empresa", "bot_name": "Eva", "telefone": "", "email": "", "portal": "",
    "produtos": [], "horario_texto": "segunda a sexta, das 8h às 18h",
    "horario": {"dias": [0, 1, 2, 3, 4], "inicio": "08:00", "fim": "18:00"},
    "fuso": "America/Sao_Paulo", "prazo_comercial": "1 dia útil",
    "prazo_suporte": "o prazo do seu contrato de suporte",
    "segmentos": [], "saudacao_inicial": "", "atalhos": [],
}

NAME_PATTERN = re.compile(
    r"(?:meu nome é|meu nome e|me chamo|pode me chamar de|eu sou o|eu sou a)\s+([A-Za-zÀ-ÿ]+)", re.I)
ASK_NAME_PATTERN = re.compile(r"qual\s+(?:é|e)\s+o\s+meu\s+nome", re.I)
EMAIL_RE = re.compile(r"^[\w.+-]+@[\w-]+(\.[\w-]+)+$")
PROTO_RE = re.compile(r"CH-\d{8}-\d{4,}", re.I)
URGENT_RE = re.compile(r"parad|urgente|fora do ar|travad|nao consigo emitir|não consigo emitir", re.I)
YES_RE = re.compile(r"^\s*(sim|s|concordo|aceito|ok|claro|pode|com certeza|de acordo)\b", re.I)
NO_RE = re.compile(r"^\s*(não|nao|n|discordo|recuso)\b", re.I)
CANCEL_RE = re.compile(r"^\s*(cancelar|cancela|desistir|voltar)\s*$", re.I)

v_texto = lambda t: len(t.strip()) >= 2
v_email = lambda t: bool(EMAIL_RE.match(t.strip()))
v_fone = lambda t: len(re.sub(r"\D", "", t)) >= 10
v_desc = lambda t: len(t.strip()) >= 15
v_consent = lambda t: bool(YES_RE.match(t))
v_proto = lambda t: bool(PROTO_RE.search(t))

FLOWS = {
    "lead": [
        ("nome", "Para o cadastro, qual é o seu nome completo?", v_texto, "Pode me dizer o seu nome?"),
        ("empresa", "Qual o nome da sua empresa?", v_texto, "Pode me dizer o nome da empresa?"),
        ("departamento", "Em qual departamento você trabalha? (ex.: TI, Financeiro, Compras)", v_texto, "Pode informar o departamento?"),
        ("email", "Qual o seu e-mail corporativo?", v_email, "Esse e-mail não parece válido. Pode digitar de novo?"),
        ("telefone", "Qual o melhor telefone para contato (com DDD)?", v_fone, "Preciso do telefone com DDD, por exemplo: 11 91234-5678."),
        ("interesse", "Qual solução ou módulo você quer conhecer? (ex.: financeiro, fiscal, estoque, RH)", v_texto, "Pode me dizer qual assunto te interessa?"),
        ("consentimento", "A {empresa} usará os dados informados de acordo com a Declaração de Privacidade ({portal}). Você concorda? (sim/não)", v_consent, "Para registrar seus dados, preciso da sua concordância. Responda 'sim' para concordar ou 'não' para cancelar."),
    ],
    "ticket": [
        ("nome", "Para o cadastro, qual é o seu nome completo?", v_texto, "Pode me dizer o seu nome?"),
        ("email", "Qual o seu e-mail de contato?", v_email, "Esse e-mail não parece válido. Pode digitar de novo?"),
        ("modulo", "Em qual sistema ou módulo o problema acontece? (ex.: Financeiro, Fiscal, Estoque)", v_texto, "Pode informar o módulo ou sistema?"),
        ("descricao", "Descreva o problema: o que você estava fazendo e qual mensagem de erro apareceu?", v_desc, "Preciso de um pouco mais de detalhes (uma frase completa já ajuda)."),
    ],
    "status": [
        ("protocolo", "Qual é o número do protocolo? (ex.: CH-20260101-0001)", v_proto, "Não reconheci esse formato. O protocolo é parecido com CH-20260101-0001."),
    ],
}
FLOW_INTRO = {
    "lead": "Ótimo! Vou anotar alguns dados para o time comercial entrar em contato. (Digite 'cancelar' para sair.)\n\n",
    "ticket": "Certo, vou abrir um chamado para você. (Digite 'cancelar' para sair.)\n\n",
    "status": "",
}
ACTION_FLOW = {"lead": "lead", "ticket": "ticket", "ticket_status": "status"}


class Chatbot:
    def __init__(self, model_file="data.pth", intents_file="intents.json",
                 company_file="company.json", db_file="history.db",
                 threshold=0.75, use_fallback=True):
        self.cfg = dict(DEFAULT_CFG)
        if os.path.exists(company_file):
            with open(company_file, "r", encoding="utf-8") as f:
                self.cfg.update(json.load(f))
        self.bot_name = self.cfg["bot_name"]
        self.threshold = threshold
        self.use_fallback = use_fallback and bool(os.getenv("ANTHROPIC_API_KEY"))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lock = threading.Lock()

        with open(intents_file, "r", encoding="utf-8") as f:
            self.intents = {i["tag"]: i for i in json.load(f)["intents"]}

        data = torch.load(model_file, map_location=self.device)
        self.all_words, self.tags = data["all_words"], data["tags"]
        self.model = NeuralNet(data["input_size"], data["hidden_size"], data["output_size"]).to(self.device)
        self.model.load_state_dict(data["model_state"])
        self.model.eval()

        self.sessions = {}
        self._init_db(db_file)

    # ---------- banco de dados ----------
    def _init_db(self, db_file):
        self.db = sqlite3.connect(db_file, check_same_thread=False)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT, timestamp TEXT, role TEXT, text TEXT, tag TEXT, confidence REAL);
            CREATE TABLE IF NOT EXISTS leads (id INTEGER PRIMARY KEY AUTOINCREMENT, created TEXT,
                session_id TEXT, nome TEXT, empresa TEXT, departamento TEXT, email TEXT, telefone TEXT, interesse TEXT);
            CREATE TABLE IF NOT EXISTS tickets (id INTEGER PRIMARY KEY AUTOINCREMENT, protocolo TEXT,
                created TEXT, session_id TEXT, nome TEXT, email TEXT, modulo TEXT, descricao TEXT,
                prioridade TEXT, status TEXT DEFAULT 'Aberto');
        """)
        try:  # bancos criados em versões anteriores
            self.db.execute("ALTER TABLE leads ADD COLUMN departamento TEXT")
        except sqlite3.OperationalError:
            pass
        self.db.commit()

    def _log(self, session_id, role, text, tag=None, conf=None):
        self.db.execute(
            "INSERT INTO messages (session_id, timestamp, role, text, tag, confidence) VALUES (?,?,?,?,?,?)",
            (session_id, datetime.now().isoformat(timespec="seconds"), role, text, tag, conf))
        self.db.commit()

    def get_history(self, session_id, limit=50):
        rows = self.db.execute("SELECT role, text FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
                               (session_id, limit)).fetchall()
        return list(reversed(rows))

    # ---------- horário e textos ----------
    def _now(self):
        try:
            from zoneinfo import ZoneInfo
            return datetime.now(ZoneInfo(self.cfg["fuso"]))
        except Exception:
            return datetime.now()

    def is_open(self):
        h, now = self.cfg["horario"], self._now()
        return now.weekday() in h["dias"] and h["inicio"] <= now.strftime("%H:%M") < h["fim"]

    def _fill(self, text, session):
        c, nome = self.cfg, session["nome"]
        status = ("Estamos em horário de atendimento agora." if self.is_open()
                  else "No momento estamos fora do horário de atendimento.")
        subs = {"{,nome}": f", {nome}" if nome else "", "{nome}": nome or "", "{empresa}": c["empresa"],
                "{bot}": c["bot_name"], "{telefone}": c["telefone"], "{email}": c["email"],
                "{portal}": c["portal"], "{horario}": c["horario_texto"],
                "{produtos}": ", ".join(c["produtos"]), "{segmentos}": ", ".join(c["segmentos"]), "{status_horario}": status}
        for k, v in subs.items():
            text = text.replace(k, v)
        return text

    def greeting(self):
        texto = self.cfg["saudacao_inicial"] or ("Posso tirar dúvidas sobre nossas soluções, abrir chamados de "
                                                 "suporte ou chamar um atendente. Digite 'menu' para ver as opções.")
        return self._fill("Olá, me chamo {bot}, sua IA! " + texto, {"nome": None})

    # ---------- sessão / classificação ----------
    def _session(self, sid):
        return self.sessions.setdefault(sid, {"nome": None, "last_tag": None, "history": [], "flow": None, "fails": 0})

    def _classify(self, text):
        X = torch.from_numpy(bag_of_words(tokenize(text), self.all_words).reshape(1, -1)).to(self.device)
        with torch.no_grad():
            prob, idx = torch.max(torch.softmax(self.model(X), dim=1), dim=1)
        return self.tags[idx.item()], prob.item()

    # ---------- fluxos (coleta de dados passo a passo) ----------
    def _start_flow(self, session, sid, name):
        session["flow"] = {"name": name, "step": 0, "data": {}, "sid": sid}
        return FLOW_INTRO[name] + self._advance(session)

    def _advance(self, session):
        flow = session["flow"]
        steps = FLOWS[flow["name"]]
        while flow["step"] < len(steps):
            key, question = steps[flow["step"]][:2]
            return question
        return self._finish_flow(session)

    def _handle_flow(self, session, text):
        flow = session["flow"]
        if CANCEL_RE.match(text):
            session["flow"] = None
            return "Tudo bem, cancelei. Posso ajudar em mais alguma coisa?"
        key, _, validator, error = FLOWS[flow["name"]][flow["step"]]
        if key == "consentimento" and NO_RE.match(text):
            session["flow"] = None
            return "Sem problemas, não vou registrar seus dados. Posso ajudar em mais alguma coisa?"
        if not validator(text):
            return error
        value = PROTO_RE.search(text).group(0).upper() if key == "protocolo" else text.strip()
        if key == "email":
            value = value.lower()
        if key == "nome":
            session["nome"] = value.split()[0].capitalize()
        flow["data"][key] = value
        flow["step"] += 1
        return self._advance(session)

    def _finish_flow(self, session):
        flow, c = session["flow"], self.cfg
        d, sid, now = flow["data"], flow["sid"], datetime.now().isoformat(timespec="seconds")
        session["flow"], session["last_tag"] = None, None

        if flow["name"] == "lead":
            self.db.execute("INSERT INTO leads (created, session_id, nome, empresa, departamento, email, telefone, interesse) "
                            "VALUES (?,?,?,?,?,?,?,?)",
                            (now, sid, d["nome"], d["empresa"], d["departamento"], d["email"], d["telefone"], d["interesse"]))
            self.db.commit()
            return (f"Obrigada, {session['nome']}! Registrei seu interesse em \"{d['interesse']}\". "
                    f"Um consultor da {c['empresa']} entrará em contato em até {c['prazo_comercial']}.")

        if flow["name"] == "ticket":
            prio = "Alta" if URGENT_RE.search(d["descricao"]) else "Normal"
            cur = self.db.execute("INSERT INTO tickets (created, session_id, nome, email, modulo, descricao, prioridade) "
                                  "VALUES (?,?,?,?,?,?,?)",
                                  (now, sid, d["nome"], d["email"], d["modulo"], d["descricao"], prio))
            proto = f"CH-{datetime.now():%Y%m%d}-{cur.lastrowid:04d}"
            self.db.execute("UPDATE tickets SET protocolo=? WHERE id=?", (proto, cur.lastrowid))
            self.db.commit()
            msg = (f"Chamado aberto! Seu protocolo é {proto}. Nossa equipe de suporte retornará por e-mail "
                   f"conforme {c['prazo_suporte']}.")
            if prio == "Alta":
                msg += f" Pelo que você descreveu, marquei como prioridade alta. Se o sistema estiver parado, ligue para {c['telefone']}."
            return msg

        row = self.db.execute("SELECT status, created, modulo, prioridade FROM tickets WHERE protocolo=?",
                              (d["protocolo"],)).fetchone()
        if not row:
            return f"Não encontrei o protocolo {d['protocolo']}. Confira o número ou fale com um atendente: {c['telefone']}."
        return (f"Protocolo {d['protocolo']}: status \"{row[0]}\", módulo {row[2]}, prioridade {row[3]}, "
                f"aberto em {row[1][:10]}.")

    # ---------- fallback com IA ----------
    def _llm_fallback(self, session):
        try:
            import anthropic
            c = self.cfg
            system = (f"Você é {c['bot_name']}, assistente virtual de atendimento da {c['empresa']}, empresa de "
                      f"sistemas de gestão empresarial (ERP). Módulos: {', '.join(c['produtos'])}. "
                      "Responda em português do Brasil, de forma curta e profissional. Nunca invente preços, "
                      "prazos, condições de contrato nem o status de chamados. Se não souber, ou se o assunto for "
                      "contrato, cobrança ou problema técnico específico, sugira abrir um chamado (digitar 'abrir chamado') "
                      "ou falar com um atendente.")
            if session["nome"]:
                system += f" O cliente se chama {session['nome']}."
            msgs = session["history"][-9:]
            while msgs and msgs[0]["role"] != "user":
                msgs = msgs[1:]
            resp = anthropic.Anthropic().messages.create(
                model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
                max_tokens=500, system=system, messages=msgs)
            return resp.content[0].text
        except Exception as e:
            print(f"[fallback falhou] {e}")
            return None

    # ---------- entendimento da mensagem ----------
    def _understand(self, session, sid, text):
        m = NAME_PATTERN.search(text)
        if m:
            session["nome"] = m.group(1).capitalize()
            return f"Prazer em te conhecer, {session['nome']}! Como posso ajudar?", None, None
        if ASK_NAME_PATTERN.search(text):
            return (f"Seu nome é {session['nome']}." if session["nome"]
                    else "Ainda não sei o seu nome. Quer me contar?"), None, None

        tag, conf = self._classify(text)
        intent = self.intents.get(tag, {})
        needs = intent.get("context_filter")
        needs = [needs] if isinstance(needs, str) else needs
        if conf >= self.threshold and (not needs or session["last_tag"] in needs):
            session["fails"] = 0
            action, ctx = intent.get("action"), session["last_tag"]
            if action in ACTION_FLOW:
                session["last_tag"] = None
                return self._start_flow(session, sid, ACTION_FLOW[action]), tag, conf
            if action == "confirm":
                session["last_tag"] = None
                if ctx == "oferta_demo":
                    return self._start_flow(session, sid, "lead"), tag, conf
                if ctx == "oferta_atendente":
                    return random.choice(self.intents["falar_atendente"]["responses"]), tag, conf
                session["last_tag"] = tag
            elif action == "deny":
                session["last_tag"] = None
            else:
                session["last_tag"] = intent.get("sets_context", tag)
            return random.choice(intent["responses"]), tag, conf

        # não entendeu
        if self.use_fallback:
            reply = self._llm_fallback(session)
            if reply:
                return reply, None, conf
        session["fails"] += 1
        if session["fails"] >= 2:
            session["fails"], session["last_tag"] = 0, "oferta_atendente"
            return "Parece que não estou conseguindo ajudar com isso. Quer que eu te passe o contato de um atendente?", None, conf
        return "Não entendi bem. Pode reformular? Se preferir, digite 'menu' para ver o que eu sei responder.", None, conf

    # ---------- API principal ----------
    def respond(self, text, session_id="default"):
        with self.lock:
            session = self._session(session_id)
            self._log(session_id, "user", text)
            session["history"].append({"role": "user", "content": text})
            tag = conf = None
            if session["flow"]:
                reply = self._handle_flow(session, text)
            else:
                reply, tag, conf = self._understand(session, session_id, text)
            reply = self._fill(reply, session)
            session["history"].append({"role": "assistant", "content": reply})
            self._log(session_id, "assistant", reply, tag, conf)
            return reply
