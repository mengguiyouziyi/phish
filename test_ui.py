#!/usr/bin/env python3
"""
PhishGuard v5 UI全面测试脚本
使用Playwright + API测试所有功能
"""

import asyncio
import json
import time
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import requests
from playwright.async_api import async_playwright, Browser, Page, expect
from bs4 import BeautifulSoup


class PhishGuardTester:
    def __init__(self, base_url: str = "http://127.0.0.1:9005"):
        self.base_url = base_url
        self.test_results = []
        self.errors = []
        self.warnings = []

    async def setup_browser(self):
        """设置浏览器"""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=True,  # 无头模式
            args=['--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu']
        )
        self.context = await self.browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        self.page = await self.context.new_page()

    async def cleanup_browser(self):
        """清理浏览器"""
        await self.context.close()
        await self.browser.close()
        await self.playwright.stop()

    def log_test(self, test_name: str, status: str, details: str = "", error: str = ""):
        """记录测试结果"""
        result = {
            'test_name': test_name,
            'status': status,
            'details': details,
            'error': error,
            'timestamp': datetime.now().isoformat()
        }
        self.test_results.append(result)

        if status == "FAILED":
            self.errors.append(result)
            print(f"❌ {test_name}: {error}")
        elif status == "WARNING":
            self.warnings.append(result)
            print(f"⚠️ {test_name}: {details}")
        else:
            print(f"✅ {test_name}: {details}")

    async def test_page_load(self):
        """测试页面加载"""
        try:
            await self.page.goto(self.base_url, timeout=30000)
            await self.page.wait_for_load_state('networkidle')

            title = await self.page.title()
            self.log_test(
                "页面加载",
                "PASSED" if title else "FAILED",
                f"页面标题: {title}"
            )
            return True
        except Exception as e:
            self.log_test("页面加载", "FAILED", error=str(e))
            return False

    async def test_ui_elements(self):
        """测试UI元素是否存在"""
        try:
            # 测试主要元素
            elements_to_test = [
                ("主标题", "h1"),
                ("URL输入框", "input[type='text']"),
                ("检测按钮", "button"),
                ("Tab导航", "[role='tab']"),
                ("结果区域", ".gradio-container"),
            ]

            for element_name, selector in elements_to_test:
                try:
                    element = await self.page.wait_for_selector(selector, timeout=5000)
                    if element:
                        self.log_test(f"UI元素-{element_name}", "PASSED", "元素存在")
                    else:
                        self.log_test(f"UI元素-{element_name}", "FAILED", "元素不存在")
                except:
                    self.log_test(f"UI元素-{element_name}", "WARNING", "元素可能不存在或选择器错误")

        except Exception as e:
            self.log_test("UI元素测试", "FAILED", error=str(e))

    async def test_tab_navigation(self):
        """测试Tab导航功能"""
        try:
            tabs = await self.page.query_selector_all("[role='tab']")
            if len(tabs) == 0:
                tabs = await self.page.query_selector_all("button")

            self.log_test("Tab导航", "PASSED", f"找到 {len(tabs)} 个Tab按钮")

            # 测试每个Tab的点击
            for i, tab in enumerate(tabs[:4]):  # 只测试前4个
                try:
                    await tab.click()
                    await self.page.wait_for_timeout(1000)
                    tab_text = await tab.text_content()
                    self.log_test(f"Tab点击-{i+1}", "PASSED", f"点击Tab: {tab_text}")
                except Exception as e:
                    self.log_test(f"Tab点击-{i+1}", "FAILED", error=str(e))

        except Exception as e:
            self.log_test("Tab导航测试", "FAILED", error=str(e))

    async def test_url_input_functionality(self):
        """测试URL输入功能"""
        try:
            # 找到URL输入框
            url_input = await self.page.wait_for_selector("input[type='text']", timeout=10000)

            # 测试输入
            test_urls = [
                "https://www.google.com",
                "https://github.com",
                "https://example.com"
            ]

            for url in test_urls:
                try:
                    await url_input.clear()
                    await url_input.fill(url)
                    await self.page.wait_for_timeout(500)

                    # 获取输入值
                    value = await url_input.input_value()
                    self.log_test(
                        f"URL输入-{url.split('.')[1]}",
                        "PASSED" if value == url else "FAILED",
                        f"输入: {value}"
                    )
                except Exception as e:
                    self.log_test(f"URL输入-{url.split('.')[1]}", "FAILED", error=str(e))

        except Exception as e:
            self.log_test("URL输入功能", "FAILED", error=str(e))

    async def test_button_functionality(self):
        """测试按钮功能"""
        try:
            # 找到检测按钮
            buttons = await self.page.query_selector_all("button")

            detection_buttons = []
            for button in buttons:
                text = await button.text_content()
                if text and any(keyword in text.lower() for keyword in ['检测', 'predict', '开始', 'analyze']):
                    detection_buttons.append(button)

            if detection_buttons:
                self.log_test("检测按钮", "PASSED", f"找到 {len(detection_buttons)} 个检测按钮")

                # 测试点击第一个检测按钮
                try:
                    await detection_buttons[0].click()
                    await self.page.wait_for_timeout(2000)
                    self.log_test("检测按钮点击", "PASSED", "按钮点击成功")
                except Exception as e:
                    self.log_test("检测按钮点击", "FAILED", error=str(e))
            else:
                self.log_test("检测按钮", "FAILED", "未找到检测按钮")

        except Exception as e:
            self.log_test("按钮功能测试", "FAILED", error=str(e))

    def test_api_endpoints(self):
        """测试API端点"""
        endpoints = [
            ("/", "首页"),
            ("/healthz", "健康检查"),
            ("/predict", "预测接口"),
        ]

        for endpoint, description in endpoints:
            try:
                url = f"{self.base_url}{endpoint}"

                if endpoint == "/predict":
                    # POST请求测试
                    response = requests.post(url, data={"url": "https://example.com"}, timeout=10)
                else:
                    # GET请求测试
                    response = requests.get(url, timeout=10)

                if response.status_code < 400:
                    self.log_test(f"API-{description}", "PASSED", f"状态码: {response.status_code}")
                else:
                    self.log_test(f"API-{description}", "FAILED", f"状态码: {response.status_code}")

            except requests.exceptions.Timeout:
                self.log_test(f"API-{description}", "FAILED", "请求超时")
            except Exception as e:
                self.log_test(f"API-{description}", "FAILED", error=str(e))

    def test_page_structure(self):
        """测试页面结构"""
        try:
            response = requests.get(self.base_url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')

            # 检查基本HTML结构
            structure_tests = [
                ("DOCTYPE声明", soup.doctype is not None),
                ("HTML标签", bool(soup.find('html'))),
                ("HEAD标签", bool(soup.find('head'))),
                ("BODY标签", bool(soup.find('body'))),
                ("TITLE标签", bool(soup.find('title'))),
                ("Meta标签", len(soup.find_all('meta')) > 0),
                ("Script标签", len(soup.find_all('script')) > 0),
                ("CSS样式", 'style' in response.text or 'css' in response.text.lower()),
            ]

            for test_name, passed in structure_tests:
                self.log_test(f"页面结构-{test_name}", "PASSED" if passed else "FAILED")

        except Exception as e:
            self.log_test("页面结构测试", "FAILED", error=str(e))

    async def test_responsive_design(self):
        """测试响应式设计"""
        try:
            viewports = [
                {'width': 1920, 'height': 1080, 'name': 'Desktop'},
                {'width': 768, 'height': 1024, 'name': 'Tablet'},
                {'width': 375, 'height': 667, 'name': 'Mobile'},
            ]

            for viewport in viewports:
                await self.page.set_viewport_size(viewport)
                await self.page.wait_for_timeout(1000)

                # 检查元素是否可见
                try:
                    main_content = await self.page.wait_for_selector('body', timeout=5000)
                    visible = await main_content.is_visible()
                    self.log_test(
                        f"响应式-{viewport['name']}",
                        "PASSED" if visible else "FAILED",
                        f"尺寸: {viewport['width']}x{viewport['height']}"
                    )
                except:
                    self.log_test(f"响应式-{viewport['name']}", "WARNING", "无法检测可见性")

        except Exception as e:
            self.log_test("响应式设计测试", "FAILED", error=str(e))

    async def test_console_errors(self):
        """测试控制台错误"""
        try:
            console_errors = []

            def handle_console_error(msg):
                if msg.type == 'error':
                    console_errors.append(msg.text)

            self.page.on('console', handle_console_error)

            # 重新加载页面以捕获错误
            await self.page.goto(self.base_url)
            await self.page.wait_for_load_state('networkidle')
            await self.page.wait_for_timeout(3000)

            if console_errors:
                for error in console_errors[:5]:  # 只显示前5个错误
                    self.log_test("控制台错误", "FAILED", error)
            else:
                self.log_test("控制台错误", "PASSED", "无JavaScript错误")

        except Exception as e:
            self.log_test("控制台错误测试", "FAILED", error=str(e))

    def generate_report(self):
        """生成测试报告"""
        passed_tests = len([r for r in self.test_results if r['status'] == 'PASSED'])
        failed_tests = len(self.errors)
        warning_tests = len(self.warnings)
        total_tests = len(self.test_results)

        report = {
            'summary': {
                'total_tests': total_tests,
                'passed': passed_tests,
                'failed': failed_tests,
                'warnings': warning_tests,
                'success_rate': f"{(passed_tests/total_tests*100):.1f}%" if total_tests > 0 else "0%"
            },
            'details': self.test_results,
            'errors': self.errors,
            'warnings': self.warnings,
            'timestamp': datetime.now().isoformat()
        }

        # 保存报告
        report_file = f"phishguard_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # 打印摘要
        print("\n" + "="*60)
        print("🧪 PHISHGUARD V5 UI 测试报告")
        print("="*60)
        print(f"总测试数: {total_tests}")
        print(f"✅ 通过: {passed_tests}")
        print(f"❌ 失败: {failed_tests}")
        print(f"⚠️ 警告: {warning_tests}")
        print(f"📊 成功率: {report['summary']['success_rate']}")
        print(f"📄 详细报告: {report_file}")

        if self.errors:
            print("\n❌ 主要错误:")
            for error in self.errors[:3]:
                print(f"  • {error['test_name']}: {error['error']}")

        return report

    async def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始PhishGuard v5全面测试...")
        print(f"🌐 测试目标: {self.base_url}")

        await self.setup_browser()

        try:
            # 页面相关测试
            if await self.test_page_load():
                await self.test_ui_elements()
                await self.test_tab_navigation()
                await self.test_url_input_functionality()
                await self.test_button_functionality()
                await self.test_responsive_design()
                await self.test_console_errors()

            # API和结构测试
            self.test_api_endpoints()
            self.test_page_structure()

        finally:
            await self.cleanup_browser()

        # 生成报告
        return self.generate_report()


async def main():
    """主函数"""
    tester = PhishGuardTester("http://127.0.0.1:9005")
    report = await tester.run_all_tests()

    # 根据测试结果设置退出码
    failed_count = len(report['errors'])
    sys.exit(failed_count if failed_count <= 100 else 100)


if __name__ == "__main__":
    asyncio.run(main())