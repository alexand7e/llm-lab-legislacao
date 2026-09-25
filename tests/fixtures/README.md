# Fixtures de HTML do parser

Recortes pequenos de HTML real, um arquivo por norma coletada, usados pelos
testes do parser sem acesso à rede. A marcação original é preservada (texto
riscado, anotações de alteração, preâmbulo e assinatura), e o cabeçalho de
cada arquivo registra a URL, o hash e a data de coleta da origem.

Os trechos recortados de cada norma ficam em `selections.yaml` (âncoras de
início e fim por regex). Os arquivos `.html` são gerados, não editados à mão:

    uv run python scripts/make_fixtures.py

Para cobrir uma norma nova, colete-a em `data/raw/`, acrescente os trechos
em `selections.yaml` e regenere.
