import io
import unittest

from m2c_pycparser import c_ast as ca, c_generator
from m2c_pycparser.c_parser import CParser


def dump(node: ca.Node) -> str:
    output = io.StringIO()
    node.show(buf=output)
    return output.getvalue()


class TestComputedGotoParser(unittest.TestCase):
    def test_target_children(self) -> None:
        tree = CParser().parse(
            "void f(void **table, int index) { goto *table[index]; }"
        )
        fn = tree.ext[0]
        assert isinstance(fn, ca.FuncDef)
        jump = (fn.body.block_items or [])[0]
        assert isinstance(jump, ca.IndirectGoto)
        assert isinstance(jump.expr, ca.ArrayRef)
        self.assertEqual(list(jump.children()), [("expr", jump.expr)])
        self.assertEqual(dump(jump.expr), "ArrayRef: \n  ID: table\n  ID: index\n")
        assert jump.coord is not None
        self.assertEqual(jump.coord.column, 35)

    def test_label_is_not_variable(self) -> None:
        tree = CParser().parse("void f(int done) { void *p = &&done; done: return; }")
        fn = tree.ext[0]
        assert isinstance(fn, ca.FuncDef)
        decl = (fn.body.block_items or [])[0]
        assert isinstance(decl, ca.Decl)
        assert isinstance(decl.init, ca.LabelAddress)
        self.assertEqual(decl.init.name, "done")
        self.assertEqual(list(decl.init.children()), [])

    def test_target_roundtrip(self) -> None:
        targets = (
            "table[index]",
            "index ? &&one : &&two",
            "target = &&one",
            "(index++, table[index])",
            "(void *)((unsigned int *)table)[index]",
            "choose(index)",
            "&&one + offsets[index]",
        )
        for target in targets:
            with self.subTest(target=target):
                source = (
                    "void f(void) { goto *" + target + "; one: return; two: return; }"
                )
                tree = CParser().parse(source)
                output = c_generator.CGenerator().visit(tree)
                self.assertEqual(dump(tree), dump(CParser().parse(output)))

    def test_relative_label_table(self) -> None:
        source = """
void f(int index) {
    static const int offsets[] = {&&one - &&one, &&two - &&one};
    goto *(&&one + offsets[index]);
one: return;
two: return;
}
"""
        tree = CParser().parse(source)
        output = c_generator.CGenerator().visit(tree)
        self.assertEqual(dump(tree), dump(CParser().parse(output)))

    def test_typedef_label_and_expression_statement(self) -> None:
        source = (
            "typedef int done; void f(void) { &&done; goto *&&done; done: return; }"
        )
        tree = CParser().parse(source)
        output = c_generator.CGenerator().visit(tree)
        self.assertIn("&&done;", output)
        self.assertEqual(dump(tree), dump(CParser().parse(output)))

    def test_ordinary_goto_and_logical_and(self) -> None:
        tree = CParser().parse(
            "void f(int a, int b) { if (a && b) goto done; done: return; }"
        )
        fn = tree.ext[0]
        assert isinstance(fn, ca.FuncDef)
        stmt = (fn.body.block_items or [])[0]
        assert isinstance(stmt, ca.If)
        assert isinstance(stmt.cond, ca.BinaryOp)
        self.assertEqual(stmt.cond.op, "&&")
        assert isinstance(stmt.iftrue, ca.Goto)
        self.assertEqual(stmt.iftrue.name, "done")
