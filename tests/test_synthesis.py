"""Gates for the synthesis pipeline (plan.md Part IV, Part 2).

The privacy-bearing pieces get the strictest tests: pseudonymization must survive Arabic
prefix attachment and never let a planted real name through, and the QC filters must kill
the known failure modes (degeneration, leakage, regurgitation) on constructed examples.
"""


import pytest

from src.preprocessing.oddadmix import clean as oddadmix_clean
from src.preprocessing.sudaneseonline import extract_thread, fix_mojibake
from src.synthesis.pseudonyms import Pseudonymizer
from src.synthesis.qc import degenerate, parse_chat

TEST_MAPPING = {
    "Test Person": {
        "fake_ar": "همام", "fake_en": "Humam", "gender": "male",
        "real_variants": ["Hassan Ali", "حسن علي", "Hassan", "حسن"],
    },
    "Owner": {
        "fake_ar": "منتصر", "fake_en": "Muntasir", "gender": "male",
        "real_variants": ["خالد", "Khalid"],
    },
}


@pytest.fixture
def pseudo():
    return Pseudonymizer(mapping=TEST_MAPPING)


# --- pseudonymization -------------------------------------------------------------------------

def test_full_names_win_over_first_names(pseudo):
    assert pseudo.apply("قابلت Hassan Ali امبارح") == "قابلت Humam امبارح"


def test_arabic_prefix_attachment_is_caught(pseudo):
    # conjunctions attach to the word in Arabic — the most common mid-sentence form
    assert pseudo.apply("وحسن قال لخالد") == "وهمام قال لمنتصر"


def test_phone_masked_but_dates_survive(pseudo):
    out = pseudo.apply("في 2022-03-08 اتصل بي +249912345678")
    assert "2022-03-08" in out
    assert "<phone>" in out and "249" not in out


def test_scan_finds_planted_names(pseudo):
    assert pseudo.scan("رسالة فيها حسن مخفي") == ["حسن"]
    assert pseudo.scan("رسالة نظيفة تماما") == []


def test_no_fake_name_contains_a_real_variant(pseudo):
    for entry in TEST_MAPPING.values():
        assert not pseudo.scan(entry["fake_ar"] + " " + entry["fake_en"])


# --- qc filters -------------------------------------------------------------------------------

def _chat_payload(lines, partner="Humam"):
    return {"text": "\n".join(lines), "meta": {"partner": partner}}


def test_parse_chat_accepts_wellformed():
    lines = [f"{'K' if i % 2 else 'Humam'}: كلام رقم {i}" for i in range(10)]
    turns = parse_chat(_chat_payload(lines))
    assert turns and len(turns) == 10


def test_parse_chat_rejects_unknown_speaker_and_short():
    lines = ["K: هلا", "Ghost: منو دا"] * 5
    assert parse_chat(_chat_payload(lines)) is None
    assert parse_chat(_chat_payload(["K: هلا", "Humam: اهلين"])) is None


def test_degeneration_filters_fire():
    assert degenerate("اي " * 60) is not None
    healthy = ("الزول دا مشى السوق واشترى حاجات كتيرة وجاب معاهو قصص عجيبة. "
               "بعدين قابل صاحبو القديم قدام الجامع واتونسوا ساعة كاملة عن أيام الجامعة. "
               "قال ليهو الدنيا اتغيرت شديد والناس بقت مشغولة بالقروش بس. "
               "في الآخر اتفقوا يتقابلوا يوم الجمعة الجاية عشان يكملوا الونسة براحة.")
    assert degenerate(healthy) is None
    english = ("hello there this is purely english text with no arabic content whatsoever "
               "and it keeps going on about different things like weather food travel plans "
               "music books films sports and every other topic imaginable in one paragraph")
    assert degenerate(english) == "not_arabic_enough"


# --- forum extraction -------------------------------------------------------------------------

def test_mojibake_roundtrip_repair():
    real = "الخرطوم عاصمة السودان والناس فيها طيبين"
    mojibake = real.encode("utf-8").decode("cp1256")
    assert fix_mojibake(mojibake) == real
    # clean text with natural ط/ظ must pass through untouched
    clean = "ظروف الطقس في الخرطوم"
    assert fix_mojibake(clean) == clean


def test_extract_thread_dedupes_quote_pyramids():
    post1 = "كلام طويل في الموضوع الأول يتكرر في الاقتباس لاحقا بالضبط"
    page = f"""<title>موضوع للنقاش</title>
    <ul><font>{post1}</font></ul>
    <ul><font>{post1}<br>رد جديد مختلف تماما على الكلام الفات دا</font></ul>"""
    title, posts = extract_thread(page)
    assert title == "موضوع للنقاش"
    joined = "\n".join(posts)
    assert joined.count(post1) == 1, "the quoted copy must be deduped"
    assert "رد جديد" in joined


# --- transcript cleaning ----------------------------------------------------------------------

def test_oddadmix_clean_strips_artifacts():
    out = oddadmix_clean("هُنَا اختَلَفَ الوَضْعُ [موسيقى] <laugh> اي اي اي اي والله ـــ")
    assert out == "هنا اختلف الوضع اي اي والله"
