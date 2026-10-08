"""End-to-end validation of the Rust checker on the real night-run MultiPL-E tasks.

Hand-written reference solutions (expected VERIFIED) and deliberately wrong mutants (expected FAILED),
all scored through gwaya.domains.checkers.check_rust exactly as run_study does.
"""
import json
import sys

from gwaya.domains.checkers import check_rust
from gwaya.domains.task import Task

TASKS = sys.argv[1]
OUT = sys.argv[2]

REF = {
"rs/HumanEval_39_prime_fib": """fn prime_fib(n: isize) -> isize {
    fn is_prime(x: isize) -> bool { if x < 2 { return false; } let mut d = 2; while d * d <= x { if x % d == 0 { return false; } d += 1; } true }
    let (mut a, mut b, mut found) = (0isize, 1isize, 0isize);
    loop { let c = a + b; a = b; b = c; if is_prime(a) { found += 1; if found == n { return a; } } }
}""",
"rs/mbpp_728_sum_list": """fn sum_list(lst1: Vec<isize>, lst2: Vec<isize>) -> Vec<isize> { lst1.iter().zip(lst2.iter()).map(|(a, b)| a + b).collect() }""",
"rs/HumanEval_35_max_element": """fn max_element(l: Vec<isize>) -> isize { *l.iter().max().unwrap() }""",
"rs/HumanEval_66_digitSum": """fn digitSum(s: String) -> isize { s.chars().filter(|c| c.is_ascii_uppercase()).map(|c| c as isize).sum() }""",
"rs/HumanEval_10_make_palindrome": """fn make_palindrome(string: String) -> String {
    let s: Vec<char> = string.chars().collect();
    let is_pal = |t: &[char]| t.iter().eq(t.iter().rev());
    let mut i = 0;
    while i < s.len() && !is_pal(&s[i..]) { i += 1; }
    let mut out = string.clone();
    out.extend(s[..i].iter().rev());
    out
}""",
"rs/mbpp_401_add_nested_tuples": """fn add_nested_tuples(test_tup1: Vec<Vec<isize>>, test_tup2: Vec<Vec<isize>>) -> Vec<Vec<isize>> {
    test_tup1.iter().zip(test_tup2.iter()).map(|(a, b)| a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()).collect()
}""",
"rs/HumanEval_64_vowels_count": """fn vowels_count(s: String) -> isize {
    let l = s.to_lowercase();
    let mut n = l.chars().filter(|c| "aeiou".contains(*c)).count() as isize;
    if l.ends_with('y') { n += 1; }
    n
}""",
"rs/HumanEval_79_decimal_to_binary": """fn decimal_to_binary(decimal: isize) -> String { format!("db{:b}db", decimal) }""",
"rs/HumanEval_8_sum_product": """fn sum_product(numbers: Vec<isize>) -> (isize, isize) { (numbers.iter().sum(), numbers.iter().product()) }""",
"rs/mbpp_573_unique_product": """fn unique_product(list_data: Vec<isize>) -> isize {
    let mut seen = std::collections::HashSet::new();
    list_data.into_iter().filter(|x| seen.insert(*x)).product()
}""",
"rs/mbpp_748_capital_words_spaces": """fn capital_words_spaces(str1: String) -> String {
    let mut out = String::new();
    let mut prev: Option<char> = None;
    for c in str1.chars() {
        if c.is_ascii_uppercase() { if let Some(p) = prev { if p.is_alphanumeric() || p == '_' { out.push(' '); } } }
        out.push(c);
        prev = Some(c);
    }
    out
}""",
"rs/mbpp_750_add_tuple": """fn add_tuple(test_list: Vec<isize>, test_tup: (isize, isize)) -> Vec<isize> { let mut v = test_list; v.push(test_tup.0); v.push(test_tup.1); v }""",
"rs/HumanEval_106_f": """fn f(n: isize) -> Vec<isize> {
    (1..=n).map(|i| if i % 2 == 0 { (1..=i).product() } else { (1..=i).sum() }).collect()
}""",
"rs/mbpp_435_last_Digit": """fn last_Digit(n: isize) -> isize { (n % 10).abs() }""",
"rs/mbpp_442_positive_count": """fn positive_count(nums: Vec<isize>) -> f64 {
    let p = nums.iter().filter(|x| **x > 0).count() as f64;
    ((p / nums.len() as f64) * 100.0).round() / 100.0
}""",
"rs/mbpp_266_lateralsurface_cube": """fn lateralsurface_cube(l: isize) -> isize { 4 * l * l }""",
"rs/mbpp_239_get_total_number_of_sequences": """fn get_total_number_of_sequences(m: isize, n: isize) -> isize {
    // count sequences of length n with a[i] >= 2*a[i-1] and 1 <= a[i] <= m
    let m = m as usize; let n = n as usize;
    let mut dp = vec![vec![0isize; n + 1]; m + 1];
    for i in 0..=m { for j in 0..=n {
        dp[i][j] = if i == 0 || j == 0 { 0 } else if i < j { 0 } else if j == 1 { i as isize } else { dp[i - 1][j] + dp[i / 2][j - 1] };
    } }
    dp[m][n]
}""",
"rs/mbpp_801_test_three_equal": """fn test_three_equal(x: isize, y: isize, z: isize) -> isize {
    if x == y && y == z { 3 } else if x == y || y == z || x == z { 2 } else { 0 }
}""",
"rs/mbpp_281_all_unique": """fn all_unique(test_list: Vec<isize>) -> bool { let s: std::collections::HashSet<_> = test_list.iter().collect(); s.len() == test_list.len() }""",
"rs/mbpp_244_next_Perfect_Square": """fn next_Perfect_Square(N: isize) -> isize { let mut r = (N.max(0) as f64).sqrt().floor() as isize; while r * r <= N { r += 1; } r * r }""",
"rs/mbpp_394_check_distinct": """fn check_distinct(test_tup: Vec<isize>) -> bool { let s: std::collections::HashSet<_> = test_tup.iter().collect(); s.len() == test_tup.len() }""",
"rs/mbpp_747_lcs_of_three": """fn lcs_of_three(X: String, Y: String, Z: String) -> isize {
    let (a, b, c): (Vec<char>, Vec<char>, Vec<char>) = (X.chars().collect(), Y.chars().collect(), Z.chars().collect());
    let mut dp = vec![vec![vec![0isize; c.len() + 1]; b.len() + 1]; a.len() + 1];
    for i in 1..=a.len() { for j in 1..=b.len() { for k in 1..=c.len() {
        dp[i][j][k] = if a[i-1] == b[j-1] && b[j-1] == c[k-1] { dp[i-1][j-1][k-1] + 1 }
                      else { dp[i-1][j][k].max(dp[i][j-1][k]).max(dp[i][j][k-1]) };
    } } }
    dp[a.len()][b.len()][c.len()]
}""",
"rs/mbpp_135_hexagonal_num": """fn hexagonal_num(n: isize) -> isize { n * (2 * n - 1) }""",
"rs/mbpp_172_count_occurance": """fn count_occurance(s: String) -> isize { s.matches("std").count() as isize }""",
"rs/mbpp_395_first_non_repeating_character": """fn first_non_repeating_character(str1: String) -> Option<String> {
    let cs: Vec<char> = str1.chars().collect();
    cs.iter().find(|c| cs.iter().filter(|d| d == c).count() == 1).map(|c| c.to_string())
}""",
}
WRONG = {  # plausible model mistakes: off-by-one, wrong formula, wrong edge case
"rs/HumanEval_35_max_element": """fn max_element(l: Vec<isize>) -> isize { *l.iter().min().unwrap() }""",
"rs/mbpp_266_lateralsurface_cube": """fn lateralsurface_cube(l: isize) -> isize { 6 * l * l }""",
"rs/HumanEval_8_sum_product": """fn sum_product(numbers: Vec<isize>) -> (isize, isize) { (numbers.iter().sum(), if numbers.is_empty() { 0 } else { numbers.iter().product() }) }""",
"rs/mbpp_135_hexagonal_num": """fn hexagonal_num(n: isize) -> isize { n * (2 * n + 1) }""",
"rs/HumanEval_79_decimal_to_binary": """fn decimal_to_binary(decimal: isize) -> String { format!("{:b}", decimal) }""",
"rs/mbpp_442_positive_count": """fn positive_count(nums: Vec<isize>) -> f64 { nums.iter().filter(|x| **x >= 0).count() as f64 / nums.len() as f64 }""",
"rs/mbpp_435_last_Digit": """fn last_Digit(n: isize) -> isize { n / 10 }""",
"rs/mbpp_172_count_occurance": """fn count_occurance(s: String) -> isize { s.matches("st").count() as isize }""",
"rs/HumanEval_39_prime_fib": """fn prime_fib(n: isize) -> isize { let v = [2, 3, 5, 13, 89, 233]; v[((n - 1) as usize).min(5)] }""",
}

tasks = {}
for line in open(TASKS):
    d = json.loads(line)
    if d.get("domain") == "rust":
        tasks[d["task_id"]] = Task(domain="rust", task_id=d["task_id"], prompt=d["prompt"], checker_payload=d["checker_payload"])

rows = []
for kind, table, expect in (("reference", REF, "VERIFIED"), ("wrong", WRONG, "FAILED")):
    for tid, code in table.items():
        r = check_rust(tasks[tid], "```rust\n" + code + "\n```")
        ok = r.status == expect
        rows.append({"kind": kind, "task_id": tid, "expected": expect, "status": r.status, "as_expected": ok,
                     "reason": (r.evidence.get("details") or {}).get("reason", r.evidence.get("reason")),
                     "error": (r.evidence.get("error") or "")[:300]})
        print(f"{'OK ' if ok else 'BAD'} {kind:9s} {tid:42s} {r.status:10s} {rows[-1]['reason']} {'' if ok else rows[-1]['error'][:150]}")
json.dump(rows, open(OUT, "w"), indent=1)
print("summary:", {k: f"{sum(r['as_expected'] for r in rows if r['kind'] == k)}/{sum(1 for r in rows if r['kind'] == k)}" for k in ("reference", "wrong")})
