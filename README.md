# Eva – chatbot de atendimento da Integra (ERP)

Assistente virtual para site de empresa de sistemas de gestão (ERP): tira dúvidas sobre módulos,
coleta leads comerciais, abre e consulta chamados de suporte e encaminha para atendente humano.

## Arquivos
- `company.json` – **dados da sua empresa** (nome, telefone, e-mail, portal, módulos, horário). Edite antes de usar.
- `intents.json` – intenções, frases de exemplo e respostas (33 intenções)
- `train.py` – treina o modelo (`data.pth`). Rode de novo sempre que mudar o `intents.json`
- `chatbot.py` – núcleo: classificação, memória, fluxos, banco SQLite e fallback com IA
- `api_server.py` + `widget.html` – API Flask e widget de chat para o site

## Como usar
```
python -m pip install -r requirements.txt
python train.py
python -m streamlit run app_streamlit.py
python api_server.py                   # abre http://127.0.0.1:5000 com o widget
```

## O que o bot faz
- **Dúvidas**: o que é ERP, financeiro, fiscal, estoque/compras, vendas/CRM, RH/folha, produção, BI, nuvem, integrações, preços, implantação, treinamento, segurança/LGPD.
- **Lead comercial** ("quero uma demonstração"): pergunta nome, empresa, e-mail, telefone e interesse → tabela `leads`.
- **Chamado de suporte** ("abrir chamado"): pergunta nome, e-mail, módulo e descrição → gera protocolo `CH-AAAAMMDD-0001` na tabela `tickets` (prioridade alta se detectar "sistema parado", "urgente"...).
- **Consulta de chamado**: o cliente informa o protocolo e vê o status. O status é alterado por você na tabela `tickets` (ex.: com DB Browser for SQLite).
- **Horário de atendimento** e **transferência para humano** usando o `company.json`.
- Depois de 2 mensagens sem entender, oferece o contato de um atendente.
- Digite `cancelar` durante qualquer cadastro para sair.

## Marcadores nas respostas do intents.json
`{empresa}` `{bot}` `{telefone}` `{email}` `{portal}` `{horario}` `{status_horario}` `{produtos}` `{,nome}`

Campos opcionais por intenção: `action` (`lead`, `ticket`, `ticket_status`, `confirm`, `deny`),
`sets_context` (guarda um contexto para a próxima fala) e `context_filter` (só vale depois daquele contexto).

## Fallback com IA (opcional)
```
$env:ANTHROPIC_API_KEY="sua_chave"     # PowerShell
```
Com a chave, o Claude responde quando o modelo não entende, sem inventar preços, prazos ou status de chamados.

## Antes de publicar no site
- Troque as respostas genéricas por informações reais e revisadas pela empresa.
- Em `api_server.py`, restrinja o CORS ao seu domínio e use um servidor de produção (ex.: gunicorn/waitress) com HTTPS.
- Leads e chamados têm dados pessoais: avise o usuário no site e trate conforme a LGPD.

## Visual (baseado no Figma da Integra)
- `widget.html`: cabeçalho azul-marinho com o logo, faixa "Olá, me chamo Eva, sua IA" (como os títulos de seção do site), balões da Eva em cartões escuros, botão de envio azul, atalhos em pílulas e aviso da Declaração de Privacidade.
- As cores e a fonte (Manrope) foram **estimadas dos prints**. Ajuste as variáveis no início do CSS (`--navy`, `--teal`, `--blue`, `--cyan`) com os valores exatos do Figma.
- Para usar o logo real, descomente o `<img>` no cabeçalho do widget e aponte para o arquivo.
- `PRIVACY_URL` (no script do widget) deve apontar para a Declaração de Privacidade do site.
- Os atalhos do chat ficam em `company.json` (`atalhos`).
- O cadastro de leads agora pede também o departamento e a concordância com a Declaração de Privacidade, como no formulário "Um novo membro da Integra".
