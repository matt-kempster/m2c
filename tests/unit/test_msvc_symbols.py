"""Tests for MSVC-mangled symbol parsing in PPC assembly.

The approach uses preprocessing to auto-quote MSVC symbols before they reach
the generic parser, rather than modifying the parser itself.
"""

import unittest

from m2c.asm_instruction import (
    AsmGlobalSymbol,
    AsmState,
    Macro,
    NaiveParsingArch,
    parse_arg_elems,
)
from m2c.instruction import _normalize_msvc_symbols


class TestNormalizeMsvcSymbols(unittest.TestCase):
    """Test the _normalize_msvc_symbols() preprocessing function."""

    def test_simple_msvc_symbol_with_ha(self) -> None:
        """MSVC symbol with @ha relocation gets quoted."""
        result = _normalize_msvc_symbols("?TheDebug@@3VDebug@@A@ha")
        self.assertEqual(result, '"?TheDebug@@3VDebug@@A"@ha')

    def test_simple_msvc_symbol_with_l(self) -> None:
        """MSVC symbol with @l relocation gets quoted."""
        result = _normalize_msvc_symbols("?TheDebug@@3VDebug@@A@l")
        self.assertEqual(result, '"?TheDebug@@3VDebug@@A"@l')

    def test_simple_msvc_symbol_with_h(self) -> None:
        """MSVC symbol with @h relocation gets quoted."""
        result = _normalize_msvc_symbols("?foo@h")
        self.assertEqual(result, '"?foo"@h')

    def test_simple_msvc_symbol_with_sda21(self) -> None:
        """MSVC symbol with @sda21 relocation gets quoted."""
        result = _normalize_msvc_symbols("?foo@@bar@sda21")
        self.assertEqual(result, '"?foo@@bar"@sda21')

    def test_simple_msvc_symbol_with_sda2(self) -> None:
        """MSVC symbol with @sda2 relocation gets quoted."""
        result = _normalize_msvc_symbols("?foo@@bar@sda2")
        self.assertEqual(result, '"?foo@@bar"@sda2')

    def test_msvc_double_at_before_ha_reloc(self) -> None:
        """MSVC symbol with @@ right before @ha relocation."""
        result = _normalize_msvc_symbols("?foo@@ha")
        self.assertEqual(result, '"?foo@"@ha')

    def test_msvc_at_not_reloc_followed_by_letters(self) -> None:
        """@ha followed by more letters is NOT a relocation - no quoting."""
        result = _normalize_msvc_symbols("?foo@habla")
        self.assertEqual(result, "?foo@habla")

    def test_msvc_sda21_not_reloc_followed_by_letters(self) -> None:
        """@sda21 followed by more letters is NOT a relocation - no quoting."""
        result = _normalize_msvc_symbols("?foo@sda21extra")
        self.assertEqual(result, "?foo@sda21extra")

    def test_msvc_at_in_middle_then_reloc(self) -> None:
        """@h in middle of symbol, @l as relocation."""
        result = _normalize_msvc_symbols("?foo@h@l")
        self.assertEqual(result, '"?foo@h"@l')

    def test_msvc_complex_real_symbol(self) -> None:
        """Real-world MSVC symbol from dc3-decomp."""
        result = _normalize_msvc_symbols("?ReadEndian@BinStream@@QAAXPAXH@Z@ha")
        self.assertEqual(result, '"?ReadEndian@BinStream@@QAAXPAXH@Z"@ha')

    def test_full_instruction_lis(self) -> None:
        """Full lis instruction with MSVC symbol."""
        result = _normalize_msvc_symbols("lis r11, ?TheDebug@@3VDebug@@A@ha")
        self.assertEqual(result, 'lis r11, "?TheDebug@@3VDebug@@A"@ha')

    def test_full_instruction_addi(self) -> None:
        """Full addi instruction with MSVC symbol."""
        result = _normalize_msvc_symbols("addi r29, r11, ?TheDebug@@3VDebug@@A@l")
        self.assertEqual(result, 'addi r29, r11, "?TheDebug@@3VDebug@@A"@l')

    def test_non_msvc_symbol_unchanged(self) -> None:
        """Regular symbol without ? should be unchanged."""
        result = _normalize_msvc_symbols("lis r10, lbl_82017228@ha")
        self.assertEqual(result, "lis r10, lbl_82017228@ha")

    def test_already_quoted_symbol_unchanged(self) -> None:
        """Already quoted MSVC symbol should be unchanged."""
        result = _normalize_msvc_symbols('bl "?ReadEndian@BinStream@@QAAXPAXH@Z"')
        self.assertEqual(result, 'bl "?ReadEndian@BinStream@@QAAXPAXH@Z"')

    def test_msvc_symbol_without_reloc_unchanged(self) -> None:
        """MSVC symbol without relocation suffix should be unchanged."""
        result = _normalize_msvc_symbols("?foo@@bar")
        self.assertEqual(result, "?foo@@bar")

    def test_msvc_symbol_with_paren_boundary(self) -> None:
        """MSVC symbol followed by parenthesis."""
        result = _normalize_msvc_symbols("?foo@@bar@l(r11)")
        self.assertEqual(result, '"?foo@@bar"@l(r11)')

    def test_msvc_symbol_with_comma_boundary(self) -> None:
        """MSVC symbol followed by comma."""
        result = _normalize_msvc_symbols("?foo@@bar@ha, r11")
        self.assertEqual(result, '"?foo@@bar"@ha, r11')


class TestParseArgElemsMsvcSymbols(unittest.TestCase):
    """Test full argument parsing with preprocessed (quoted) MSVC symbols.

    These tests verify that once MSVC symbols are quoted by preprocessing,
    the parser correctly handles them.
    """

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

    def test_quoted_msvc_symbol_with_ha_macro(self) -> None:
        """Quoted MSVC symbol with @ha should produce Macro."""
        # After preprocessing: "?TheDebug@@3VDebug@@A"@ha
        result = self.parse('"?TheDebug@@3VDebug@@A"@ha')
        self.assertIsInstance(result, Macro)
        self.assertEqual(result.macro_name, "ha")
        self.assertIsInstance(result.argument, AsmGlobalSymbol)
        self.assertEqual(result.argument.symbol_name, "?TheDebug@@3VDebug@@A")

    def test_quoted_msvc_symbol_with_l_macro(self) -> None:
        """Quoted MSVC symbol with @l should produce Macro."""
        result = self.parse('"?TheDebug@@3VDebug@@A"@l')
        self.assertIsInstance(result, Macro)
        self.assertEqual(result.macro_name, "l")
        self.assertIsInstance(result.argument, AsmGlobalSymbol)
        self.assertEqual(result.argument.symbol_name, "?TheDebug@@3VDebug@@A")

    def test_quoted_msvc_symbol_with_sda21_macro(self) -> None:
        """Quoted MSVC symbol with @sda21 should produce Macro."""
        result = self.parse('"?foo@@bar"@sda21')
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

    def test_quoted_msvc_symbol_no_reloc(self) -> None:
        """Quoted MSVC symbol without relocation."""
        result = self.parse('"?foo@@bar"')
        self.assertIsInstance(result, AsmGlobalSymbol)
        self.assertEqual(result.symbol_name, "?foo@@bar")


if __name__ == "__main__":
    unittest.main()
