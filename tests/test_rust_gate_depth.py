from scripts import rust_gate_depth as G

TESTS = """}

fn main() {
    let candidate = f;
    assert_eq!(candidate(vec![1, 2], 3),
               vec![1, 2, 3]);
    assert!(candidate(vec![], 0).is_empty());
    assert_eq!(candidate(vec![5], 1), vec![5; 1]); // trailing ; { } in a comment
    assert_ne!(candidate(vec![9], 2), vec![0]);
}
"""


def test_split_main_counts_multiline_asserts_and_keeps_preamble():
    head, pre, asserts, tail = G.split_main(TESTS)
    assert len(asserts) == 4 and len(pre) == 1 and "let candidate = f;" in pre[0]
    assert "fn main() {" in head and tail.strip().endswith("}")


def test_build_gate_keeps_only_first_k_assertions():
    g2 = G.build_gate(TESTS, 2)
    assert g2.count("assert") == 2 and "assert_ne" not in g2 and "is_empty" in g2
    assert G.norm(G.build_gate(TESTS, 4)) == G.norm(TESTS)  # k = all gives back the full suite
