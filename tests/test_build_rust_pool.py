import pytest

from scripts.data import build_rust_pool as B

TEST_SRC = '''use twofer::twofer;
use std::collections::HashMap;

fn helper() -> i32 { 1 }

#[test]
fn empty_string() {
    assert_eq!(twofer(""), "One for you, one for me.");
}

#[test]
#[ignore]
fn braces_in_literals() {
    let s = "{ not a block }";
    assert_eq!(twofer("x"), format!("One for {}, one for me.", "x")); let _c = '}';
    assert_eq!(helper(), 1);
}
'''


def test_convert_keeps_helpers_drops_crate_use_and_unwraps_tests():
    head, blocks, info = B.convert_tests(TEST_SRC, "twofer")
    assert "use std::collections::HashMap;" in head and "fn helper()" in head
    assert "use twofer" not in head and "#[test]" not in head and "#[ignore]" not in head
    assert len(blocks) == 2 and info["n_tests"] == 2
    assert blocks[1].startswith("{") and "let _c = '}';" in blocks[1] and blocks[1].endswith("}")
    text = B.assemble(head, blocks[:1])
    assert "assert_eq!" in text and "braces_in_literals" not in text


@pytest.mark.parametrize("src,why", [
    ("#[test]\n#[should_panic]\nfn t() { panic!(); }", "should_panic"),
    ("#[test]\nfn t() -> Result<(), String> { Ok(()) }", "test_returns_value"),
    ("use rand::Rng;\n#[test]\nfn t() { assert!(true); }", "external_crate_in_tests"),
    ("fn nothing() {}", "no_test_functions"),
])
def test_unsupported_shapes_are_rejected_not_guessed(src, why):
    with pytest.raises(ValueError, match=why):
        B.convert_tests(src, "x")


def test_brace_end_ignores_braces_in_strings_chars_and_comments():
    s = '{ let a = "}"; let b = \'}\'; // }\n /* } */ }'
    assert B.brace_end(s, 0) == len(s)


def test_crate_qualified_calls_and_inner_uses_are_stripped():
    src = "#[test]\nfn t() {\n    use hello_world::Thing;\n    assert_eq!(\"x\", hello_world::hello());\n}\n"
    head, blocks, _ = B.convert_tests(src, "hello_world")
    assert "hello_world" not in blocks[0] and "hello()" in blocks[0]
