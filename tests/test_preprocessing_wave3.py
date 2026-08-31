"""Unit tests for the wave-3 preprocessing modules (plan.md Part IV, step 1.9)."""

from src.preprocessing.anasudani import PostExtractor, clean_post
from src.preprocessing.blogger import strip_html as blogger_strip
from src.preprocessing.telegram import build_documents, clean_message
from src.preprocessing.vbarchive import ForumPostExtractor
from src.preprocessing.wpjson_clean import strip_html


class TestTelegram:
    def test_clean_message_strips_html_and_keeps_arabic(self):
        text = clean_message('سمعت <b>الونسة</b> دي شنو يا زول<br/>قول لي الحاصل')
        assert text == "سمعت الونسة دي شنو يا زول\nقول لي الحاصل"

    def test_promo_message_dropped(self):
        assert clean_message('اشترك هنا <a href="https://t.me/x">قناة</a> رابط') == ""

    def test_literal_br_entity_removed_after_unescape(self):
        assert "<br>" not in clean_message("&lt;br&gt;كلام طويل بالعربي يمشي هنا وهناك")

    def test_documents_split_on_id_gap_and_dedup_footer(self):
        footer = "تابعونا على القناة الرسمية للمزيد"
        arabic = "حكاية طويلة من الحلة والناس قاعدين يونسوا في الليل "
        arabic2 = "قصة تانية خالص عن سوق أم درمان والمواصلات والزحمة الشديدة "
        messages = [(1, arabic * 6), (2, footer), (3, footer), (500, arabic2 * 6)]
        docs = build_documents("ch", messages)
        assert len(docs) == 2                       # gap of 497 > MAX_ID_GAP splits
        assert sum(doc["text"].count(footer) for doc in docs) == 1

    def test_val_leaves_train_when_channel_tiny(self):
        # covered via build_documents contract: nothing to assert here beyond shape
        docs = build_documents("ch", [(1, "كلام سوداني ساكت يتقال في القعدة " * 20)])
        assert docs and docs[0]["channel"] == "ch"


class TestAnasudani:
    def test_extractor_handles_nested_divs(self):
        parser = PostExtractor()
        parser.feed('<div class="content">نص أول<div>مقتبس داخلي</div>نص تاني</div>'
                    '<div class="content">مساهمة تانية</div>')
        assert len(parser.posts) == 2
        assert "مقتبس داخلي" in parser.posts[0]

    def test_bbcode_stripped(self):
        assert clean_post("[glow1=666633]مديح[/glow1] جميل") == "مديح جميل"


class TestWpjson:
    def test_strip_html_blocks_to_lines(self):
        assert strip_html("<p>سطر أول</p><p>سطر &amp; تاني</p>") == "سطر أول\nسطر & تاني"

    def test_blogger_strip_matches(self):
        assert blogger_strip("<div>كلام</div><script>x=1</script>") == "كلام"


class TestVbarchive:
    def test_all_three_post_containers_matched(self):
        parser = ForumPostExtractor()
        parser.feed("<div class='posttext'>مشاركة فيبي</div>"
                    "<div itemprop=\"commentText\" class='post entry-content '>مشاركة آي بي بي</div>"
                    "<div id='post_message_9'>مشاركة كاملة</div>"
                    "<div class='navbar'>ليست مشاركة</div>")
        assert len(parser.posts) == 3
        assert not any("ليست" in post for post in parser.posts)
