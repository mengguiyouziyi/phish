#!/usr/bin/env python3
"""
从GitHub收集高质量钓鱼网站数据集
"""

import requests
import json
import time
import re
from typing import List, Dict, Any
from pathlib import Path
import pandas as pd

class GitHubPhishCollector:
    """GitHub钓鱼网站数据收集器"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.collected_data = []

    def search_github_repositories(self, keywords: List[str]) -> List[Dict]:
        """搜索GitHub仓库"""
        repositories = []

        for keyword in keywords:
            print(f"🔍 搜索关键词: {keyword}")

            # GitHub API搜索
            search_url = f"https://api.github.com/search/repositories?q={keyword}+phishing&sort=stars&order=desc"
            try:
                response = self.session.get(search_url)
                if response.status_code == 200:
                    data = response.json()
                    for repo in data.get('items', [])[:5]:  # 取前5个
                        repositories.append({
                            'name': repo['full_name'],
                            'url': repo['html_url'],
                            'stars': repo['stargazers_count'],
                            'description': repo['description'],
                            'language': repo.get('language', 'Unknown')
                        })
                        print(f"  📦 发现仓库: {repo['full_name']} (⭐{repo['stargazers_count']})")
            except Exception as e:
                print(f"  ❌ 搜索失败: {e}")

            time.sleep(1)  # 避免API限制

        return repositories

    def extract_urls_from_github_files(self, repo_url: str) -> List[str]:
        """从GitHub仓库文件中提取URL"""
        urls = []

        try:
            # 转换为raw内容URL
            raw_url = repo_url.replace('github.com', 'raw.githubusercontent.com').replace('/tree/', '/')

            # 常见的钓鱼网站数据文件路径
            file_paths = [
                'phishing_urls.txt',
                'urls.txt',
                'malicious_urls.txt',
                'suspicious_domains.txt',
                'phishing_domains.txt',
                'data/phishing.txt',
                'datasets/urls.txt'
            ]

            for file_path in file_paths:
                file_url = f"{raw_url}/{file_path}"
                try:
                    response = self.session.get(file_url, timeout=10)
                    if response.status_code == 200:
                        content = response.text
                        # 提取URL
                        extracted = self.extract_urls_from_text(content)
                        urls.extend(extracted)
                        print(f"  📄 从 {file_path} 提取了 {len(extracted)} 个URL")
                        break
                except:
                    continue

        except Exception as e:
            print(f"  ❌ 处理仓库失败: {e}")

        return urls

    def extract_urls_from_text(self, text: str) -> List[str]:
        """从文本中提取URL"""
        url_patterns = [
            r'https?://[^\s<>"{}|\\^`\[\]]+',
            r'http?://[^\s<>"{}|\\^`\[\]]+',
            r'[a-zA-Z0-9][a-zA-Z0-9-]{1,61}[a-zA-Z0-9]\.[a-zA-Z]{2,}',
            r'www\.[a-zA-Z0-9-]+\.[a-zA-Z]{2,}'
        ]

        urls = []
        for pattern in url_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                # 清理和标准化URL
                url = match.strip().rstrip('.,;:!?\'"()[]{}')
                if not url.startswith(('http://', 'https://')):
                    url = 'http://' + url
                urls.append(url)

        return list(set(urls))  # 去重

    def collect_from_known_sources(self) -> List[Dict[str, Any]]:
        """从已知的钓鱼网站数据源收集"""
        known_sources = [
            {
                'name': 'PhishTank',
                'url': 'https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-links-ACTIVE.txt',
                'type': 'active_phishing'
            },
            {
                'name': 'OpenPhish',
                'url': 'https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-links-OPENPHISH.txt',
                'type': 'openphish'
            },
            {
                'name': 'URLHaus',
                'url': 'https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-links-URLHAUS.txt',
                'type': 'urlhaus'
            },
            {
                'name': 'Certego',
                'url': 'https://raw.githubusercontent.com/certego/chromium-phishing-domains/main/chromium-phishing-domains.txt',
                'type': 'certego'
            }
        ]

        collected_data = []

        for source in known_sources:
            print(f"🎯 收集来源: {source['name']}")

            try:
                response = self.session.get(source['url'], timeout=30)
                if response.status_code == 200:
                    content = response.text
                    urls = self.extract_urls_from_text(content)

                    for url in urls[:100]:  # 每个来源最多100个
                        collected_data.append({
                            'url': url,
                            'source': source['name'],
                            'type': source['type'],
                            'label': 1,  # 钓鱼网站
                            'confidence': 'high',
                            'collected_at': time.time()
                        })

                    print(f"  ✅ 收集了 {min(len(urls), 100)} 个URL")
                else:
                    print(f"  ❌ 请求失败: {response.status_code}")

            except Exception as e:
                print(f"  ❌ 收集失败: {e}")

            time.sleep(2)  # 避免请求过快

        return collected_data

    def collect_high_quality_benign(self) -> List[Dict[str, Any]]:
        """收集高质量良性网站"""
        benign_sources = {
            'top_companies': [
                'https://www.google.com',
                'https://www.microsoft.com',
                'https://www.apple.com',
                'https://www.amazon.com',
                'https://www.facebook.com',
                'https://www.twitter.com',
                'https://www.linkedin.com',
                'https://www.github.com',
                'https://www.youtube.com',
                'https://www.netflix.com',
                'https://www.spotify.com',
                'https://www.adobe.com',
                'https://www.oracle.com',
                'https://www.ibm.com',
                'https://www.intel.com'
            ],
            'financial': [
                'https://www.bankofamerica.com',
                'https://www.chase.com',
                'https://www.wellsfargo.com',
                'https://www.citibank.com',
                'https://www.hsbc.com',
                'https://www.barclays.com',
                'https://www.goldmansachs.com',
                'https://www.morganstanley.com'
            ],
            'government': [
                'https://www.gov.cn',
                'https://www.whitehouse.gov',
                'https://www.europa.eu',
                'https://www.gov.uk',
                'https://www.canada.ca',
                'https://www.australia.gov.au'
            ],
            'education': [
                'https://www.harvard.edu',
                'https://www.mit.edu',
                'https://www.stanford.edu',
                'https://www.berkeley.edu',
                'https://www.cmu.edu',
                'https://www.tsinghua.edu.cn',
                'https://www.pku.edu.cn'
            ]
        }

        benign_data = []

        for category, urls in benign_sources.items():
            for url in urls:
                benign_data.append({
                    'url': url,
                    'source': category,
                    'type': 'benign',
                    'label': 0,  # 良性网站
                    'confidence': 'high',
                    'collected_at': time.time()
                })

        return benign_data

    def validate_urls(self, urls: List[str]) -> List[str]:
        """验证URL是否可访问"""
        valid_urls = []

        for url in urls[:50]:  # 限制验证数量
            try:
                response = self.session.head(url, timeout=10, allow_redirects=True)
                if response.status_code in [200, 301, 302, 303, 307, 308]:
                    valid_urls.append(url)
                    print(f"  ✅ {url}")
                else:
                    print(f"  ⚠️  {url} - {response.status_code}")
            except:
                print(f"  ❌ {url} - 不可访问")

            time.sleep(0.5)  # 避免请求过快

        return valid_urls

    def run_collection(self):
        """运行完整的数据收集流程"""
        print("🚀 开始GitHub高质量钓鱼网站数据收集")
        print("=" * 60)

        all_data = []

        # 1. 从已知来源收集钓鱼网站
        print("\n🎯 步骤1: 收集钓鱼网站数据")
        phishing_data = self.collect_from_known_sources()
        all_data.extend(phishing_data)

        # 2. 收集良性网站
        print(f"\n✅ 步骤2: 收集良性网站数据")
        benign_data = self.collect_high_quality_benign()
        all_data.extend(benign_data)

        # 3. 搜索GitHub仓库
        print(f"\n🔍 步骤3: 搜索GitHub仓库")
        keywords = [
            'phishing dataset',
            'malicious URLs',
            'phishing domains',
            'security dataset',
            'threat intelligence'
        ]

        repositories = self.search_github_repositories(keywords)

        # 4. 从GitHub仓库提取URL
        print(f"\n📦 步骤4: 从GitHub仓库提取URL")
        github_urls = []
        for repo in repositories[:3]:  # 处理前3个仓库
            print(f"\n处理仓库: {repo['name']}")
            urls = self.extract_urls_from_github_files(repo['url'])
            github_urls.extend(urls)

        # 添加GitHub收集的钓鱼网站
        for url in github_urls[:200]:  # 限制数量
            all_data.append({
                'url': url,
                'source': 'github_repos',
                'type': 'github_phishing',
                'label': 1,
                'confidence': 'medium',
                'collected_at': time.time()
            })

        # 5. 保存数据
        print(f"\n💾 步骤5: 保存数据")
        df = pd.DataFrame(all_data)

        # 数据统计
        phishing_count = len(df[df['label'] == 1])
        benign_count = len(df[df['label'] == 0])

        print(f"📊 数据统计:")
        print(f"  总记录数: {len(df)}")
        print(f"  钓鱼网站: {phishing_count}")
        print(f"  良性网站: {benign_count}")

        # 保存数据
        output_dir = Path("github_data")
        output_dir.mkdir(exist_ok=True)

        # 保存原始数据
        df.to_parquet(output_dir / "raw_github_data.parquet", index=False)

        # 保存分类数据
        phishing_df = df[df['label'] == 1]
        benign_df = df[df['label'] == 0]

        phishing_df.to_parquet(output_dir / "phishing_urls.parquet", index=False)
        benign_df.to_parquet(output_dir / "benign_urls.parquet", index=False)

        # 保存URL列表
        with open(output_dir / "phishing_urls.txt", "w", encoding="utf-8") as f:
            for url in phishing_df['url'].unique():
                f.write(url + "\n")

        with open(output_dir / "benign_urls.txt", "w", encoding="utf-8") as f:
            for url in benign_df['url'].unique():
                f.write(url + "\n")

        print(f"✅ 数据已保存到 {output_dir}/")
        print(f"  - raw_github_data.parquet: 完整数据集")
        print(f"  - phishing_urls.parquet: 钓鱼网站数据")
        print(f"  - benign_urls.parquet: 良性网站数据")
        print(f"  - phishing_urls.txt: 钓鱼网站URL列表")
        print(f"  - benign_urls.txt: 良性网站URL列表")

        return df

if __name__ == "__main__":
    collector = GitHubPhishCollector()
    data = collector.run_collection()