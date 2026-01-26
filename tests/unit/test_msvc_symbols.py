"""Tests for MSVC-mangled symbol parsing in PPC assembly."""

import unittest

from m2c.asm_instruction import (
    AsmGlobalSymbol,
    AsmState,
    Macro,
    NaiveParsingArch,
    parse_arg_elems,
    parse_word,
    valid_word,
)


class TestParseWordMsvcSymbols(unittest.TestCase):
    """Test parse_word() with MSVC-mangled symbols."""

    def test_simple_msvc_symbol(self) -> None:
        """MSVC symbol without relocation."""
        elems = list("?foo@@bar")
        result = parse_word(elems)
        self.assertEqual(result, "?foo@@bar")
        self.assertEqual(elems, [])

    def test_msvc_symbol_with_ha_reloc(self) -> None:
        """MSVC symbol ending with @ha relocation."""
        elems = list("?TheDebug@@3VDebug@@A@ha")
        result = parse_word(elems)
        self.assertEqual(result, "?TheDebug@@3VDebug@@A")
        self.assertEqual(elems, list("@ha"))

    def test_msvc_symbol_with_l_reloc(self) -> None:
        """MSVC symbol ending with @l relocation."""
        elems = list("?TheDebug@@3VDebug@@A@l")
        result = parse_word(elems)
        self.assertEqual(result, "?TheDebug@@3VDebug@@A")
        self.assertEqual(elems, list("@l"))

    def test_msvc_symbol_with_h_reloc(self) -> None:
        """MSVC symbol ending with @h relocation."""
        elems = list("?foo@h")
        result = parse_word(elems)
        self.assertEqual(result, "?foo")
        self.assertEqual(elems, list("@h"))

    def test_msvc_symbol_with_sda21_reloc(self) -> None:
        """MSVC symbol ending with @sda21 relocation."""
        elems = list("?foo@@bar@sda21")
        result = parse_word(elems)
        self.assertEqual(result, "?foo@@bar")
        self.assertEqual(elems, list("@sda21"))

    def test_msvc_symbol_with_sda2_reloc(self) -> None:
        """MSVC symbol ending with @sda2 relocation."""
        elems = list("?foo@@bar@sda2")
        result = parse_word(elems)
        self.assertEqual(result, "?foo@@bar")
        self.assertEqual(elems, list("@sda2"))

    def test_msvc_double_at_before_ha_reloc(self) -> None:
        """MSVC symbol with @@ right before @ha relocation."""
        elems = list("?foo@@ha")
        result = parse_word(elems)
        self.assertEqual(result, "?foo@")
        self.assertEqual(elems, list("@ha"))

    def test_msvc_at_not_reloc_followed_by_letters(self) -> None:
        """@ha followed by more letters is NOT a relocation."""
        elems = list("?foo@habla")
        result = parse_word(elems)
        self.assertEqual(result, "?foo@habla")
        self.assertEqual(elems, [])

    def test_msvc_sda21_not_reloc_followed_by_letters(self) -> None:
        """@sda21 followed by more letters is NOT a relocation."""
        elems = list("?foo@sda21extra")
        result = parse_word(elems)
        self.assertEqual(result, "?foo@sda21extra")
        self.assertEqual(elems, [])

    def test_msvc_at_in_middle_then_reloc(self) -> None:
        """@h in middle of symbol, @l as relocation."""
        elems = list("?foo@h@l")
        result = parse_word(elems)
        self.assertEqual(result, "?foo@h")
        self.assertEqual(elems, list("@l"))

    def test_msvc_complex_real_symbol(self) -> None:
        """Real-world MSVC symbol from dc3-decomp."""
        elems = list("?ReadEndian@BinStream@@QAAXPAXH@Z@ha")
        result = parse_word(elems)
        self.assertEqual(result, "?ReadEndian@BinStream@@QAAXPAXH@Z")
        self.assertEqual(elems, list("@ha"))

    def test_msvc_symbol_with_trailing_comma(self) -> None:
        """MSVC symbol followed by comma (operand separator)."""
        elems = list("?foo@@bar@ha,")
        result = parse_word(elems)
        self.assertEqual(result, "?foo@@bar")
        self.assertEqual(elems, list("@ha,"))

    def test_msvc_symbol_with_trailing_paren(self) -> None:
        """MSVC symbol in parentheses."""
        elems = list("?foo@@bar@l)")
        result = parse_word(elems)
        self.assertEqual(result, "?foo@@bar")
        self.assertEqual(elems, list("@l)"))

    def test_non_msvc_symbol_unchanged(self) -> None:
        """Regular symbol without ? should work as before."""
        elems = list("normalSymbol@ha")
        result = parse_word(elems)
        self.assertEqual(result, "normalSymbol")
        self.assertEqual(elems, list("@ha"))

    def test_non_msvc_symbol_with_question_in_middle(self) -> None:
        """Symbol with ? in middle (not MSVC-style) should include ?."""
        elems = list("foo?bar@ha")
        result = parse_word(elems)
        self.assertEqual(result, "foo?bar")
        self.assertEqual(elems, list("@ha"))

    def test_question_mark_in_valid_word(self) -> None:
        """Verify ? is now in valid_word character set."""
        self.assertIn("?", valid_word)


class TestParseArgElemsMsvcSymbols(unittest.TestCase):
    """Test full argument parsing with MSVC symbols."""

    def setUp(self) -> None:
        self.arch = NaiveParsingArch()
        self.asm_state = AsmState()

    def parse(self, arg: str) -> object:
        """Helper to parse an argument string."""
        elems = list(arg)
        return parse_arg_elems(
            elems,
            self.arch,
            self.asm_state,
            top_level=True,
        )

    def test_msvc_symbol_with_ha_macro(self) -> None:
        """MSVC symbol with @ha should produce Macro."""
        result = self.parse("?TheDebug@@3VDebug@@A@ha")
        self.assertIsInstance(result, Macro)
        self.assertEqual(result.macro_name, "ha")
        self.assertIsInstance(result.argument, AsmGlobalSymbol)
        self.assertEqual(result.argument.symbol_name, "?TheDebug@@3VDebug@@A")

    def test_msvc_symbol_with_l_macro(self) -> None:
        """MSVC symbol with @l should produce Macro."""
        result = self.parse("?TheDebug@@3VDebug@@A@l")
        self.assertIsInstance(result, Macro)
        self.assertEqual(result.macro_name, "l")
        self.assertIsInstance(result.argument, AsmGlobalSymbol)
        self.assertEqual(result.argument.symbol_name, "?TheDebug@@3VDebug@@A")

    def test_msvc_symbol_with_sda21_macro(self) -> None:
        """MSVC symbol with @sda21 should produce Macro."""
        result = self.parse("?foo@@bar@sda21")
        self.assertIsInstance(result, Macro)
        self.assertEqual(result.macro_name, "sda21")
        self.assertIsInstance(result.argument, AsmGlobalSymbol)
        self.assertEqual(result.argument.symbol_name, "?foo@@bar")

    def test_normal_symbol_still_works(self) -> None:
        """Regular symbol with relocation should still work."""
        result = self.parse("lbl_82017228@ha")
        self.assertIsInstance(result, Macro)
        self.assertEqual(result.macro_name, "ha")
        self.assertIsInstance(result.argument, AsmGlobalSymbol)
        self.assertEqual(result.argument.symbol_name, "lbl_82017228")

    def test_msvc_symbol_no_reloc(self) -> None:
        """MSVC symbol without relocation."""
        result = self.parse("?foo@@bar")
        self.assertIsInstance(result, AsmGlobalSymbol)
        self.assertEqual(result.symbol_name, "?foo@@bar")


if __name__ == "__main__":
    unittest.main()
