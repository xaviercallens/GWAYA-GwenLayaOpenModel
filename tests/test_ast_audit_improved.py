"""
Tests for improved AST audit functionality.
Tests the fixes for pass statements in except handlers, function stub detection,
MagicMock/Mock detection, and comment stripping in non-Python languages.
"""
import pytest
from gwaya.ast_audit import ZeroStubAudit

class TestPassStatementAllowance:
    """Tests for allowing pass in except handlers and class bodies."""
    
    def test_pass_in_except_handler_is_allowed(self):
        """Pass is allowed in except handlers."""
        code = """
try:
    x = 1
except ValueError:
    pass
"""
        result = ZeroStubAudit.audit_python_code(code)
        assert result.is_clean, "Pass in except handler should be allowed"
    
    def test_pass_in_class_body_is_allowed(self):
        """Pass is allowed in class bodies."""
        code = """
class MyException(Exception):
    pass
"""
        result = ZeroStubAudit.audit_python_code(code)
        assert result.is_clean, "Pass in class body should be allowed"
    
    def test_pass_in_function_body_is_forbidden(self):
        """Pass in function bodies is still forbidden."""
        code = """
def foo():
    pass
"""
        result = ZeroStubAudit.audit_python_code(code)
        assert not result.is_clean, "Pass in function body should be forbidden"
        assert any("pass" in v.lower() for v in result.violations)


class TestFunctionStubDetection:
    """Tests for detecting stub function bodies."""
    
    def test_function_with_only_docstring(self):
        """Function body that is only a docstring is flagged."""
        code = """
def foo():
    \"\"\"This is a docstring.\"\"\"
"""
        result = ZeroStubAudit.audit_python_code(code)
        assert not result.is_clean, "Function with only docstring should be flagged"
        assert any("docstring" in v.lower() for v in result.violations)
    
    def test_function_with_docstring_and_pass(self):
        """Function with docstring and pass is flagged."""
        code = """
def foo():
    \"\"\"This is a docstring.\"\"\"
    pass
"""
        result = ZeroStubAudit.audit_python_code(code)
        assert not result.is_clean, "Function with docstring and pass should be flagged"
    
    def test_function_with_only_ellipsis(self):
        """Function body that is only Ellipsis is flagged."""
        code = """
def foo():
    ...
"""
        result = ZeroStubAudit.audit_python_code(code)
        assert not result.is_clean, "Function with only ellipsis should be flagged"
        assert any("ellipsis" in v.lower() for v in result.violations)
    
    def test_function_with_only_raise_not_implemented(self):
        """Function body with only raise NotImplementedError is flagged."""
        code = """
def foo():
    raise NotImplementedError
"""
        result = ZeroStubAudit.audit_python_code(code)
        assert not result.is_clean, "Function with only raise NotImplementedError should be flagged"
        assert any("notimplementederror" in v.lower() for v in result.violations)
    
    def test_function_with_return_none_and_no_annotation(self):
        """Function that returns None without annotation is allowed."""
        code = """
def foo():
    return None
"""
        result = ZeroStubAudit.audit_python_code(code)
        # Should be allowed because there's no return annotation
        assert result.is_clean, "Function with return None and no annotation should be allowed"
    
    def test_function_with_return_none_and_annotation(self):
        """Function that returns None but promises a value is flagged."""
        code = """
def foo() -> int:
    return None
"""
        result = ZeroStubAudit.audit_python_code(code)
        assert not result.is_clean, "Function that promises value but returns None should be flagged"
        assert any("return" in v.lower() and "none" in v.lower() for v in result.violations)


class TestMagicMockDetection:
    """Tests for detecting MagicMock/Mock/patch usage as return values."""
    
    def test_function_returning_mock(self):
        """Function that returns Mock() is flagged."""
        code = """
def foo():
    return Mock()
"""
        result = ZeroStubAudit.audit_python_code(code)
        assert not result.is_clean, "Function returning Mock should be flagged"
        assert any("mock" in v.lower() for v in result.violations)
    
    def test_function_returning_magic_mock(self):
        """Function that returns MagicMock() is flagged."""
        code = """
def foo():
    return MagicMock()
"""
        result = ZeroStubAudit.audit_python_code(code)
        assert not result.is_clean, "Function returning MagicMock should be flagged"
        assert any("magicmock" in v.lower() for v in result.violations)
    
    def test_function_returning_patch(self):
        """Function that returns patch() is flagged."""
        code = """
def foo():
    return patch("something")
"""
        result = ZeroStubAudit.audit_python_code(code)
        assert not result.is_clean, "Function returning patch should be flagged"
        assert any("patch" in v.lower() for v in result.violations)


class TestFakeVariableNaming:
    """Tests for fake_*/dummy_* variable detection."""
    
    def test_fake_variable_in_fake_function(self):
        """fake_var in fake_function is flagged."""
        code = """
def fake_function():
    fake_var = 1
    return fake_var
"""
        result = ZeroStubAudit.audit_python_code(code)
        assert not result.is_clean, "fake_var in fake_function should be flagged"
    
    def test_fake_variable_in_real_function(self):
        """fake_var in regular function is allowed."""
        code = """
def real_function():
    fake_var = 1
    return fake_var
"""
        result = ZeroStubAudit.audit_python_code(code)
        assert result.is_clean, "fake_var in real_function should be allowed"
    
    def test_dummy_variable_in_dummy_function(self):
        """dummy_var in dummy_function is flagged."""
        code = """
def dummy_function():
    dummy_var = 1
    return dummy_var
"""
        result = ZeroStubAudit.audit_python_code(code)
        assert not result.is_clean, "dummy_var in dummy_function should be flagged"


class TestCommentStrippingInLanguages:
    """Tests for comment and string stripping in non-Python languages."""
    
    def test_lean_sorry_in_comment(self):
        """'sorry' in a Lean comment should be allowed."""
        code = """
def test : True := by
  -- sorry, this is unfinished
  trivial
"""
        result = ZeroStubAudit.audit_lean_code(code)
        assert result.is_clean, "sorry in Lean comment should be allowed"
    
    def test_lean_sorry_in_code(self):
        """'sorry' in actual Lean code should be flagged."""
        code = """
def test : True := by
  sorry
"""
        result = ZeroStubAudit.audit_lean_code(code)
        assert not result.is_clean, "sorry in Lean code should be flagged"
    
    def test_rust_todo_in_comment(self):
        """'todo!' in a Rust comment should be allowed."""
        code = r'''
fn main() {
    // TODO: implement this
    println!("ok");
}
'''
        result = ZeroStubAudit.audit_rust_code(code)
        assert result.is_clean, "todo! in Rust comment should be allowed"
    
    def test_rust_todo_in_code(self):
        """'todo!()' in actual Rust code should be flagged."""
        code = """
fn main() {
    todo!()
}
"""
        result = ZeroStubAudit.audit_rust_code(code)
        assert not result.is_clean, "todo!() in Rust code should be flagged"
    
    def test_cpp_todo_in_comment(self):
        """'TODO' in a C++ comment should be allowed."""
        code = """
int main() {
    // TODO: implement this
    return 0;
}
"""
        result = ZeroStubAudit.audit_cpp_code(code)
        assert result.is_clean, "TODO in C++ comment should be allowed"
    
    def test_cpp_todo_in_code(self):
        """'TODO' in actual C++ code should be flagged."""
        code = """
int main() {
    TODO
    return 0;
}
"""
        result = ZeroStubAudit.audit_cpp_code(code)
        assert not result.is_clean, "TODO in C++ code should be flagged"
    
    def test_go_panic_todo_in_comment(self):
        """'panic(TODO)' in a Go comment should be allowed."""
        code = """
func main() {
    // panic("TODO: implement this")
    println("ok")
}
"""
        result = ZeroStubAudit.audit_go_code(code)
        assert result.is_clean, 'panic("TODO") in Go comment should be allowed'
    
    def test_go_panic_todo_in_code(self):
        """'panic(TODO)' in actual Go code should be flagged."""
        code = """
func main() {
    panic("TODO")
}
"""
        result = ZeroStubAudit.audit_go_code(code)
        assert not result.is_clean, 'panic("TODO") in Go code should be flagged'


class TestComplexScenarios:
    """Tests for more complex scenarios."""
    
    def test_real_function_with_proper_implementation(self):
        """Real function with proper implementation should pass."""
        code = """
def add(a: int, b: int) -> int:
    \"\"\"Add two numbers.\"\"\"
    return a + b
"""
        result = ZeroStubAudit.audit_python_code(code)
        assert result.is_clean, "Real implementation should pass audit"
    
    def test_multiple_violations(self):
        """Multiple violations in same code."""
        code = """
def mock_function():
    \"\"\"Only docstring.\"\"\"

def another_stub():
    raise NotImplementedError
"""
        result = ZeroStubAudit.audit_python_code(code)
        assert not result.is_clean, "Multiple violations should be caught"
        assert len(result.violations) >= 2, "Should report multiple violations"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
