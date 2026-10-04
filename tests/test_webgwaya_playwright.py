"""
tests/test_webgwaya_playwright.py
=================================
Automated End-to-End browser tests for WebGWAYA using Playwright.
Tests:
- Application layout and telemetry headers
- Tab transitions across all 5 navigation views
- Interactive AST ZeroStubAudit sandbox (live validation & rejection)
- AI Studio scenario loading and Laya Evaluator verification
- Benchmark dashboard charts and cards rendering
- ChromaDB RAG search interaction
- LoRA configuration generator
"""
from __future__ import annotations

import os
import pytest
import subprocess
import time
import socket
from playwright.sync_api import sync_playwright

BASE_URL = os.environ.get("WEBGWAYA_URL", "http://127.0.0.1:8000")

def is_port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

@pytest.fixture(scope="module", autouse=True)
def start_server():
    if not is_port_open(8000):
        # We need to run uvicorn
        env = os.environ.copy()
        env["GWAYA_ALLOW_UNISOLATED"] = "1"
        proc = subprocess.Popen(["uvicorn", "webgwaya.app:app", "--port", "8000"], env=env)
        for _ in range(50):
            if is_port_open(8000):
                break
            time.sleep(0.1)
        yield
        proc.terminate()
        proc.wait()
    else:
        yield

@pytest.fixture(scope="module")
def browser_context():
    # Ensure Playwright finds chromium in D:\.playwright_browsers
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = "D:\\.playwright_browsers"
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1400, "height": 900})
        yield context
        browser.close()



def test_playwright_webgwaya_header_and_tabs(browser_context):
    page = browser_context.new_page()
    page.goto(BASE_URL, wait_until="networkidle")

    # 1. Verify branding
    assert "WebGWAYA" in page.title()
    assert page.locator(".logo-badge").inner_text().startswith("⚡ WebGWAYA")

    # 2. Verify Tab Transitions
    tabs = [
        ("tab-architecture", "GWAYA Concept & Physics"),
        ("tab-studio", "Live AI Coding Studio"),
        ("tab-benchmark", "Multi-Model Gains & Benchmark"),
        ("tab-rag", "Private RAG Vault"),
        ("tab-lora", "LoRA Admin & Fine-Tuning"),
    ]

    for tab_id, tab_label in tabs:
        btn = page.locator(f"button[data-tab='{tab_id}']")
        btn.click()
        assert btn.get_attribute("class") and "active" in btn.get_attribute("class")
        content = page.locator(f"#{tab_id}")
        assert content.is_visible()
    
    page.close()


def test_playwright_ast_audit_interaction(browser_context):
    page = browser_context.new_page()
    page.goto(BASE_URL, wait_until="networkidle")

    # Navigate to Architecture tab
    page.locator("button[data-tab='tab-architecture']").click()

    # Initial code has `pass` stub -> verify it gets rejected
    btn_audit = page.locator("#btn-run-ast-audit")
    btn_audit.click()

    page.wait_for_selector("#ast-audit-result .badge-red", timeout=5000)
    result_text = page.locator("#ast-audit-result").inner_text()
    assert "AST SHIELD: REJECTED" in result_text

    # Now replace with clean, valid logic
    input_box = page.locator("#ast-test-input")
    input_box.fill("def solve(items: list[int]) -> int:\n    return sum(items)\n")
    btn_audit.click()

    page.wait_for_selector("#ast-audit-result .badge-green", timeout=5000)
    result_clean = page.locator("#ast-audit-result").inner_text()
    assert "AST SHIELD: PASSED" in result_clean

    page.close()


def test_playwright_studio_scenario_and_evaluator(browser_context):
    page = browser_context.new_page()
    page.goto(BASE_URL, wait_until="networkidle")

    # Navigate to Studio tab
    page.locator("button[data-tab='tab-studio']").click()

    # Select scenario
    selector = page.locator("#scenario-selector")
    selector.select_option("scenario_01_merge_intervals")
    page.locator("#btn-load-scenario").click()

    # Verify input populated
    goal_val = page.locator("#studio-goal").input_value()
    assert "merge_intervals" in goal_val

    # Trigger Laya Evaluator Only
    page.locator("#btn-eval-only").click()
    page.wait_for_selector("#laya-verdict-badge.badge-green", timeout=8000)

    verdict_text = page.locator("#laya-verdict-badge").inner_text()
    assert "VERIFIED_TESTS" in verdict_text

    tests_val = page.locator("#eval-tests").inner_text()
    assert tests_val.startswith("3/3")

    page.close()


def test_playwright_benchmark_dashboard(browser_context):
    page = browser_context.new_page()
    page.goto(BASE_URL, wait_until="networkidle")

    # Navigate to Benchmark tab
    page.locator("button[data-tab='tab-benchmark']").click()
    page.wait_for_selector("#chart-pass-rates", timeout=5000)

    # Verify canvas rendered
    chart_canvas = page.locator("#chart-pass-rates")
    assert chart_canvas.is_visible()

    # Verify summary cards exist
    cards = page.locator("#benchmark-summary-cards .glass-card")
    page.wait_for_timeout(1000)
    count = cards.count()
    assert count >= 1

    page.close()


def test_playwright_rag_search_interaction(browser_context):
    page = browser_context.new_page()
    page.goto(BASE_URL, wait_until="networkidle")

    # Navigate to RAG tab
    page.locator("button[data-tab='tab-rag']").click()

    query_input = page.locator("#rag-query-input")
    query_input.fill("ZeroStubAudit")
    page.locator("#btn-query-rag").click()

    # Wait for results
    page.wait_for_selector("#rag-results-container pre.code-block", timeout=6000)
    res_text = page.locator("#rag-results-container").inner_text()
    assert "ast_audit.py" in res_text or "Similarity" in res_text

    page.close()


def test_playwright_lora_config_generator(browser_context):
    page = browser_context.new_page()
    page.goto(BASE_URL, wait_until="networkidle")

    # Navigate to LoRA tab
    page.locator("button[data-tab='tab-lora']").click()

    page.locator("#btn-generate-lora-config").click()
    page.wait_for_timeout(1500)

    output_text = page.locator("#lora-config-output").inner_text()
    assert "target_modules" in output_text or "status" in output_text

    page.close()


def test_playwright_gpu_start_stop_lifecycle(browser_context):
    page = browser_context.new_page()
    page.goto(BASE_URL, wait_until="networkidle")

    btn_toggle = page.locator("#btn-gpu-toggle")
    assert btn_toggle.is_visible()

    # Step A: Ensure GPU is started first
    current_label = page.locator("#gpu-btn-label").inner_text()
    if "START" in current_label:
        btn_toggle.click()
        page.wait_for_function("document.getElementById('gpu-btn-label').innerText.includes('ACTIVE')", timeout=25000)

    # Verify ACTIVE state
    assert "ACTIVE" in page.locator("#gpu-btn-label").inner_text()
    status_text = page.locator("#gpu-status-text").inner_text()
    assert "RTX" in status_text or "Active" in status_text or "MB" in status_text
    assert "Ollama CUDA" in page.locator("#ollama-status-text").inner_text()

    # Step B: Test STOP GPU via top header button
    btn_toggle.click()
    page.wait_for_function("document.getElementById('gpu-btn-label').innerText.includes('START')", timeout=15000)

    # Verify STOPPED state in header telemetry
    assert "START" in page.locator("#gpu-btn-label").inner_text()
    assert "STOPPED" in page.locator("#gpu-status-text").inner_text()
    assert "Offloaded" in page.locator("#ollama-status-text").inner_text()

    # Verify STOPPED banner appears in Studio tab
    page.locator("button[data-tab='tab-studio']").click()
    alert_banner = page.locator("#studio-gpu-alert")
    assert alert_banner.is_visible()
    assert "STOPPED" in alert_banner.inner_text()

    # Step C: Test START GPU via Studio alert banner
    page.locator("#btn-banner-start-gpu").click()
    page.wait_for_function("document.getElementById('gpu-btn-label').innerText.includes('ACTIVE')", timeout=25000)

    # Verify ACTIVE state restored
    assert "ACTIVE" in page.locator("#gpu-btn-label").inner_text()
    page.wait_for_timeout(500)
    assert not alert_banner.is_visible()

    # Step D: Test STOP AFTER (leave GPU stopped for the user's other project)
    btn_toggle.click()
    page.wait_for_function("document.getElementById('gpu-btn-label').innerText.includes('START')", timeout=15000)
    assert "START" in page.locator("#gpu-btn-label").inner_text()
    assert "STOPPED" in page.locator("#gpu-status-text").inner_text()

    page.close()
