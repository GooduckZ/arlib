import unittest
from arlib.utils.sexpr import *

class TestSExprParser(unittest.TestCase):
    """Test cases for S-expression parser."""

    def test_tokenize(self):
        """Test tokenization of S-expressions."""
        self.assertEqual(tokenize("(+ 1 2)"), ['(', '+', '1', '2', ')'])
        self.assertEqual(tokenize("()"), ['(', ')'])
        self.assertEqual(tokenize("(+ 1.5 foo)"), ['(', '+', '1.5', 'foo', ')'])

    def testparse_atoms(self):
        """Test parsing of atomic values."""
        self.assertEqual(parse_atom("42"), 42)
        self.assertEqual(parse_atom("3.14"), 3.14)
        self.assertEqual(parse_atom("xyz"), "xyz")

    def test_parse_expressions(self):
        """Test parsing of complete S-expressions."""
        self.assertEqual(parse("(+ 1 2)"), ['+', 1, 2])
        self.assertEqual(parse("(+ 1 (* 2 3))"), ['+', 1, ['*', 2, 3]])
        self.assertEqual(parse("(define x 42)"), ['define', 'x', 42])
        self.assertIsNone(parse(""))

    def test_parse_errors(self):
        """Test error handling for invalid expressions."""
        with self.assertRaises(ParseError):
            parse("(")  # Unclosed parenthesis
        with self.assertRaises(ParseError):
            parse(")")  # Unexpected closing parenthesis
        with self.assertRaises(ParseError):
            parse("(+ 1 2) 3")  # Trailing tokens

    def test_nested_expressions(self):
        """Test parsing of deeply nested expressions."""
        expr = "(list 1 (list 2 (list 3 4)))"
        expected = ['list', 1, ['list', 2, ['list', 3, 4]]]
        self.assertEqual(parse(expr), expected)

    def test_mixed_types(self):
        """Test parsing of expressions with mixed atomic types."""
        expr = "(function 42 3.14 symbol)"
        expected = ['function', 42, 3.14, 'symbol']
        self.assertEqual(parse(expr), expected)


if __name__ == '__main__':
    unittest.main()