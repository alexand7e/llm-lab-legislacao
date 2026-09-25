# -----------------------------------------------------------------------------
# File:     tests/test_ingest_parser.py
# Purpose:  Unit tests for the norm parser (synthetic HTML + real fixtures).
# Author:   Alexandre
# Created:  2026-09-25
# License:  Apache-2.0
# -----------------------------------------------------------------------------

from pathlib import Path

import pytest

from lab.ingest.parser import Article, clean, extract_lines, parse_html

FIXTURES = Path(__file__).parent / "fixtures"


def doc(*paragraphs: str) -> str:
    return "<html><body>" + "".join(f"<p>{p}</p>" for p in paragraphs) + "</body></html>"


def parse(*paragraphs: str) -> list[Article]:
    return parse_html(doc(*paragraphs), "x")


def by_id(articles: list[Article]) -> dict[str, Article]:
    return {a.article: a for a in articles}


# -- linhas -------------------------------------------------------------------


def test_struck_text_is_discarded():
    lines = extract_lines(
        doc(
            "<strike>I - redação antiga;</strike>",
            '<span style="text-decoration: line-through">I - outra antiga;</span>',
            "<s>velho</s><del>mais velho</del>I - redação atual;",
        )
    )
    assert [line.text for line in lines] == ["I - redação atual;"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("I - texto; (Redação dada pela Lei nº 1, de 2000)", "I - texto;"),
        ("§ 1º (Revogado). (Redação dada pela Lei nº 1, de 2000) Vigência", "§ 1º (Revogado)."),
        ("Art. 2º Texto. (Incluído pela Lei nº 1) (Vide ADI nº 2 (e outra))", "Art. 2º Texto."),
        ("(Incluído pela Medida Provisória nº 1, de 2021) (Rejeitada)", ""),
    ],
)
def test_clean_strips_amendment_annotations(raw, expected):
    assert clean(raw) == expected


def test_annotation_only_paragraphs_are_dropped():
    assert extract_lines(doc("(Redação dada pela Lei nº 1, de 2000)", "&nbsp;")) == []


def test_line_keeps_raw_text_for_later_steps():
    (line,) = extract_lines(doc("Art. 1º (Revogado pela Lei nº 9, de 2001)"))
    assert "Revogado pela" in line.raw and line.text == "Art. 1º"


# -- hierarquia ---------------------------------------------------------------


def test_preamble_and_signature_are_ignored():
    articles = parse(
        "LEI Nº 1, DE 1º DE JANEIRO DE 2000",
        "O PRESIDENTE DA REPÚBLICA Faço saber que...",
        "Art. 1º Esta Lei vale.",
        "Brasília, 1º de janeiro de 2000.",
        "FULANO DE TAL",
    )
    assert [(a.article, a.caput) for a in articles] == [("1", "Esta Lei vale.")]


def test_headings_build_path_and_reset_lower_levels():
    articles = by_id(
        parse(
            "TÍTULO I Das Normas",
            "CAPÍTULO I DISPOSIÇÕES GERAIS",
            "Seção I Do Escopo",
            "Art. 1º Um.",
            "CAPÍTULO II",
            "DOS DIREITOS",
            "Art. 2º Dois.",
            "TÍTULO II Das Penas",
            "Art. 3º Três.",
        )
    )
    assert articles["1"].path == ["Título I", "Capítulo I", "Seção I"]
    assert articles["2"].path == ["Título I", "Capítulo II"]
    assert articles["2"].headings[-1].name == "DOS DIREITOS"
    assert articles["3"].path == ["Título II"]


def test_heading_keyword_case_and_suffix():
    articles = by_id(parse("SEÇÃO II Da Oferta", "Art. 1º Um.", "CAPÍTULO VI-A", "Nome", "Art. 2º"))
    assert articles["1"].path == ["Seção II"]
    assert articles["2"].path == ["Capítulo VI-A"]


def test_sentence_starting_with_heading_word_is_not_a_heading():
    (article,) = parse("Art. 1º Caput:", "I - inciso;", "Parte civil do texto continua o inciso.")
    assert article.path == []
    assert article.units[0].text.endswith("Parte civil do texto continua o inciso.")


def test_units_get_parent_keys():
    (article,) = parse(
        "Art. 1º Caput:",
        "I - um:",
        "a) alínea do um;",
        "II - dois;",
        "§ 1º Parágrafo:",
        "I - inciso do parágrafo:",
        "a) alínea do parágrafo.",
        "Parágrafo único. Único.",
    )
    assert [(u.type, u.key, u.parent) for u in article.units] == [
        ("inciso", "I", None),
        ("alinea", "I.a", "I"),
        ("inciso", "II", None),
        ("paragrafo", "§1", None),
        ("inciso", "§1.I", "§1"),
        ("alinea", "§1.I.a", "§1.I"),
        ("paragrafo", "§unico", None),
    ]
    assert article.units[1].text == "alínea do um;"


@pytest.mark.parametrize(
    ("line", "number"),
    [
        ("Art. 7º Texto.", "7"),
        ("Art. 7° Texto.", "7"),
        ("Art. 10. Texto.", "10"),
        ("Art. 55-A. Texto.", "55-A"),
        ("Art. 5 7. (VETADO).", "57"),
    ],
)
def test_article_numbers(line, number):
    (article,) = parse(line)
    assert article.article == number


@pytest.mark.parametrize(
    ("line", "ident"),
    [("§ 1º Texto.", "1"), ("§ 1 º Texto.", "1"), ("§ 2o Texto.", "2"), ("§ 3º-A. Texto.", "3-A")],
)
def test_paragraph_numbers(line, ident):
    (article,) = parse("Art. 1º Caput.", line)
    assert (article.units[0].id, article.units[0].text) == (ident, "Texto.")


def test_inciso_suffix():
    (article,) = parse("Art. 1º Caput:", "V - cinco;", "V-A - cinco-a;", "V - B - cinco-b;")
    assert [u.id for u in article.units] == ["V", "V-A", "V-B"]


def test_unrecognized_and_quoted_lines_continue_previous_unit():
    (article,) = parse(
        "Art. 1º Crime:",
        "Pena - Detenção de um a seis meses.",
        "§ 1º Alterar a outra lei:",
        '"IV - texto citado de outra lei".',
    )
    assert article.caput == "Crime:\nPena - Detenção de um a seis meses."
    assert [u.key for u in article.units] == ["§1"]
    assert article.units[0].text.endswith('"IV - texto citado de outra lei".')


def test_article_text_renders_labels():
    (article,) = parse("Art. 7º Caput:", "I - um;", "Parágrafo único. Fim.")
    assert article.id == "x:art:7"
    assert article.text == "Art. 7º Caput:\nI - um;\nParágrafo único. Fim."
    (tenth,) = parse("Art. 10. Dez.", "§ 1º Um.")
    assert tenth.text == "Art. 10. Dez.\n§ 1º Um."


# -- status -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "status"),
    [
        ("Art. 9º (VETADO).", "vetado"),
        ("Art. 9º (Vetado) .", "vetado"),
        ("Art. 9º (Revogado).", "revogado"),
        ("Art. 9º (Revogada pela Lei nº 2, de 2001)", "revogado"),
        ("Art. 9º Revogado.", "revogado"),
        ("Art. 9º A revogação do consentimento é gratuita.", "vigente"),
        ("Art. 9º Texto. (Redação dada pela Lei nº 2, de 2001)", "vigente"),
    ],
)
def test_article_status(line, status):
    (article,) = parse(line)
    assert article.status == status


def test_status_is_per_unit_inside_a_vigente_article():
    (article,) = parse(
        "Art. 1º Caput:",
        "I - (revogado);",
        "II - vigente;",
        "§ 1º (VETADO).",
        "§ 2º (Revogado pela Lei nº 3, de 2002)",
        "§ 3º Vigente.",
    )
    assert article.status == "vigente"
    assert [(u.key, u.status) for u in article.units] == [
        ("I", "revogado"),
        ("II", "vigente"),
        ("§1", "vetado"),
        ("§2", "revogado"),
        ("§3", "vigente"),
    ]


def test_article_text_leaves_out_revoked_and_vetoed_units():
    (article,) = parse("Art. 1º Caput:", "I - (revogado);", "II - vigente.", "§ 1º (VETADO).")
    assert article.text == "Art. 1º Caput:\nII - vigente."


# -- fixtures reais -----------------------------------------------------------


@pytest.mark.parametrize("path", sorted(FIXTURES.glob("*.html")), ids=lambda p: p.stem)
def test_fixture_parses_without_duplicates(path):
    articles = parse_html(path.read_text(encoding="utf-8"), path.stem)
    assert articles
    numbers = [a.article for a in articles]
    assert len(numbers) == len(set(numbers))
    for article in articles:
        keys = [u.key for u in article.units]
        assert len(keys) == len(set(keys)), article.id
        assert "Brasília" not in article.text
        assert "Redação dada" not in article.text


def test_fixture_struck_versions_are_not_duplicated():
    html = (FIXTURES / "marco_civil.html").read_text(encoding="utf-8")
    art5 = by_id(parse_html(html, "marco_civil"))["5"]
    ids = [u.id for u in art5.units]
    assert ids == ["I", "II", "III", "IV", "V", "VI", "VII", "VIII"]
    assert all(": o conjunto" in u.text for u in art5.units if u.id in {"VII", "VIII"})
