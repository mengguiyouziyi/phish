#!/usr/bin/env python3
"""
训练优化的FusionDNN模型，改进复杂URL识别
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import sys
sys.path.append('.')

from phishguard_v1.models.fusion_model import FusionDNN, predict_proba

def extract_url_features(url):
    """提取URL特征"""
    if not url.startswith("http"):
        url = "http://" + url

    # URL特征
    features = {}

    # 基本长度特征
    features["url_len"] = len(url)
    host = url.split('//')[-1].split('/')[0]
    features["host_len"] = len(host)
    path = '/' + '/'.join(url.split('/')[3:]) if len(url.split('/')) > 3 else '/'
    features["path_len"] = len(path)

    # 字符统计
    features["num_digits"] = sum(c.isdigit() for c in url)
    features["num_letters"] = sum(c.isalpha() for c in url)
    features["num_specials"] = sum(not c.isalnum() for c in url)

    # 特殊字符统计
    features["num_dots"] = url.count('.')
    features["num_hyphen"] = url.count('-')
    features["num_slash"] = url.count('/')
    features["num_qm"] = url.count('?')
    features["num_at"] = url.count('@')
    features["num_pct"] = url.count('%')

    # 布尔特征
    features["has_ip"] = any(part.isdigit() for part in host.split('.'))
    features["subdomain_depth"] = host.count('.') if host != 'localhost' else 0
    features["tld_suspicious"] = 1 if any(tld in host.lower() for tld in ['.tk', '.ml', '.ga', '.cf']) else 0
    features["has_punycode"] = 1 if 'xn--' in host.lower() else 0
    features["scheme_https"] = 1 if url.startswith('https') else 0

    # 查询和片段长度
    query = url.split('?')[-1] if '?' in url else ''
    features["query_len"] = len(query)
    fragment = url.split('#')[-1] if '#' in url else ''
    features["fragment_len"] = len(fragment)

    # HTTP响应特征（模拟）
    features["status_code"] = 200
    features["bytes"] = 1024

    return features

def create_enhanced_dataset():
    """创建增强的数据集"""
    print("🔧 创建增强数据集...")

    # 良性URL - 包含更多带路径的URL
    benign_urls = [
        # 基础域名
        "https://www.google.com",
        "https://www.facebook.com",
        "https://www.twitter.com",
        "https://www.instagram.com",
        "https://www.linkedin.com",
        "https://www.microsoft.com",
        "https://www.apple.com",
        "https://www.amazon.com",
        "https://www.youtube.com",
        "https://www.wikipedia.org",

        # 中文网站
        "https://www.baidu.com",
        "https://www.taobao.com",
        "https://www.qq.com",
        "https://www.weibo.com",
        "https://www.zhihu.com",
        "https://www.douban.com",
        "https://www.tmall.com",
        "https://www.jd.com",
        "https://www.163.com",
        "https://www.sina.com.cn",

        # 带路径的良性URL
        "https://www.baidu.com/index.php",
        "https://www.baidu.com/s?wd=test",
        "https://www.baidu.com/img/bd_logo1.png",
        "https://www.google.com/search?q=test",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://www.facebook.com/profile.php",
        "https://www.twitter.com/home",
        "https://www.instagram.com/p/Cx12345/",
        "https://www.amazon.com/dp/B123456789",
        "https://www.taobao.com/item.htm?id=123456",
        "https://www.jd.com/product/123456.html",
        "https://www.qq.com/news/",
        "https://www.weibo.com/u/1234567890",
        "https://www.zhihu.com/question/123456",
        "https://www.douban.com/subject/123456/",
        "https://www.wikipedia.org/wiki/Python",

        # 更多带参数的URL
        "https://www.google.com/maps/place/Beijing",
        "https://www.youtube.com/results?search_query=test",
        "https://www.facebook.com/sharer/sharer.php?u=test",
        "https://www.twitter.com/intent/tweet?url=test",
        "https://www.amazon.com/s?k=laptop",
        "https://www.taobao.com/search?q=phone",
        "https://www.jd.com/search?keyword=book",
        "https://www.baidu.com/s?ie=utf-8&f=8&rsv_bp=1&rsv_idx=1",
        "https://www.google.com/url?sa=t&rct=j&q=&esrc=s",
        "https://www.facebook.com/groups/123456789/",

        # API路径
        "https://api.github.com/user",
        "https://api.twitter.com/2/users/by",
        "https://graph.facebook.com/me",
        "https://maps.googleapis.com/maps/api/geocode/json",
        "https://api.weixin.qq.com/cgi-bin/token",

        # 静态资源
        "https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png",
        "https://www.facebook.com/rsrc.php/v3/yj/r/2sGX-1s7yfD.png",
        "https://www.baidu.com/img/PCtm_d9c8750bed0b3c7d089fa7d55720d6cf.png",
        "https://www.taobao.com/favicon.ico",
        "https://www.jd.com/favicon.ico",
        "https://www.qq.com/qqidc/images/favicon.ico",

        # 登录页面
        "https://login.microsoftonline.com/",
        "https://accounts.google.com/signin",
        "https://www.facebook.com/login.php",
        "https://twitter.com/login",
        "https://www.instagram.com/accounts/login/",
        "https://passport.baidu.com/v2/login",
        "https://login.taobao.com/member/login.jhtml",
        "https://passport.weibo.com/visitor/visitor",
        "https://mail.qq.com/cgi-bin/frame_html",
        "https://exmail.qq.com/login",

        # 个人主页
        "https://www.facebook.com/zuck",
        "https://twitter.com/jack",
        "https://www.weibo.com/u/1195242865",
        "https://www.zhihu.com/people/explore",
        "https://www.douban.com/people/123456/",
        "https://github.com/torvalds",
        "https://www.linkedin.com/in/satyanadella/",

        # 新闻和内容页面
        "https://news.google.com/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGRqTVhZU0FtVnVHZ0pWVXlnQVAB",
        "https://www.bbc.com/news/world-asia-china-123456",
        "https://www.cnn.com/2023/12/01/tech/china-tech/index.html",
        "https://www.reuters.com/world/china/china-economy-2023-12-01/",
        "https://www.theguardian.com/world/china",
        "https://www.nytimes.com/section/world/asia/china",
        "https://www.wsj.com/news/types/china-news",
        "https://www.ft.com/china",
        "https://www.bloomberg.com/asia",
        "https://www.scmp.com/economy/china-economy",

        # 电商产品页面
        "https://www.amazon.com/dp/B07VGRJDFY",
        "https://www.taobao.com/item.htm?id=123456789",
        "https://www.jd.com/product/123456789.html",
        "https://www.tmall.com/item.htm?id=123456789",
        "https://www.suning.com/product/123456789.html",
        "https://www.gome.com.cn/product/123456789.html",
        "https://www.yhd.com/product/123456789.html",
        "https://www.dangdang.com/product/123456789.html",
        "https://www.amazon.in/dp/B08N5KWB9H",
        "https://www.amazon.co.uk/dp/B08N5KWB9H",

        # 搜索结果页面
        "https://www.google.com/search?q=how+to+code",
        "https://www.baidu.com/s?wd=python+tutorial",
        "https://www.bing.com/search?q=machine+learning",
        "https://duckduckgo.com/?q=data+science",
        "https://search.yahoo.com/search?p=web+development",
        "https://yandex.com/search/?text=artificial+intelligence",
        "https://www.ask.com/web?q=programming+languages",
        "https://www.aol.com/search/query?q=software+engineering",
        "https://www.ecosia.org/search?q=climate+change",
        "https://www.qwant.com/?q=renewable+energy",

        # 文件下载
        "https://www.mozilla.org/en-US/firefox/download/",
        "https://www.google.com/chrome/",
        "https://www.microsoft.com/en-us/edge/download",
        "https://www.apple.com/safari/",
        "https://www.videolan.org/vlc/",
        "https://www.7-zip.org/download.html",
        "https://www.win-rar.com/download.html",
        "https://www.adobe.com/products/reader.html",
        "https://www.libreoffice.org/download/download-libreoffice/",
        "https://www.gimp.org/downloads/",

        # 论坛和社区
        "https://www.reddit.com/r/programming/",
        "https://stackoverflow.com/questions/123456/how-to-code",
        "https://github.com/torvalds/linux",
        "https://www.quora.com/What-is-the-best-programming-language",
        "https://www.douban.com/group/123456/",
        "https://www.zhihu.com/question/123456/answer/789012",
        "https://www.v2ex.com/t/123456",
        "https://www.csdn.net/nav/ai",
        "https://www.jianshu.com/p/1234567890",
        "https://www.cnblogs.com/test/p/123456.html"
    ]

    # 钓鱼URL
    phishing_urls = [
        # 经典钓鱼URL
        "http://update-security-windows.com",
        "http://apple-account-verify.com",
        "http://paypal-secure-account.com",
        "http://amazon-security-check.com",
        "http://microsoft-account-security.com",
        "http://google-account-security.com",
        "http://facebook-login-verify.com",
        "http://twitter-security-account.com",
        "http://linkedin-account-verify.com",
        "http://instagram-security-check.com",

        # 带路径的钓鱼URL
        "http://verify-paypal-account.com/login.php",
        "http://appleid-apple.com.verify/index.php",
        "http://amazon-security.com/verify-account/",
        "http://microsoft-security.com/account/login",
        "http://google-security.com/account/verify",
        "http://facebook-security.com/login/verify",
        "http://twitter-security.com/account/login",
        "http://linkedin-security.com/account/verify",
        "http://instagram-security.com/account/login",

        # 高度可疑的钓鱼URL
        "http://paypal-security-center.com/login.php",
        "http://apple-account-security.com/verify.php",
        "http://amazon-account-security.com/login.php",
        "http://microsoft-account-security.com/verify.php",
        "http://google-account-security.com/login.php",
        "http://facebook-account-security.com/verify.php",
        "http://twitter-account-security.com/login.php",
        "http://linkedin-account-security.com/verify.php",
        "http://instagram-account-security.com/login.php",

        # 带参数的钓鱼URL
        "http://paypal-security.com/login.php?redirect=phishing",
        "http://apple-security.com/verify.php?user= victim",
        "http://amazon-security.com/account.php?session=phishing",
        "http://microsoft-security.com/login.php?token= fake",
        "http://google-security.com/verify.php?auth=phishing",
        "http://facebook-security.com/login.php?next=phishing",
        "http://twitter-security.com/account.php?oauth=phishing",
        "http://linkedin-security.com/verify.php?code=phishing",
        "http://instagram-security.com/login.php?csrf=phishing",

        # IP地址钓鱼URL
        "http://192.168.1.1/paypal/login.php",
        "http://123.456.789.012/apple/verify.php",
        "http://10.0.0.1/amazon/login.php",
        "http://172.16.0.1/microsoft/verify.php",
        "http://203.0.113.0/google/login.php",

        # 短链接钓鱼URL
        "http://bit.ly/verify-paypal",
        "http://tinyurl.com/apple-security",
        "http://goo.gl/amazon-verify",
        "http://ow.ly/microsoft-login",
        "http://is.gd/google-security",

        # 带子域名的钓鱼URL
        "http://login.paypal.com.verify.com",
        "http://apple.id.apple.com.security.com",
        "http://amazon.account.amazon.com.security.com",
        "http://microsoft.account.microsoft.com.security.com",
        "http://google.account.google.com.security.com",

        # 混合钓鱼URL
        "http://paypal-security-center.com/account/login/verify.php",
        "http://apple-id-verify.com/account/security/check.php",
        "http://amazon-security-check.com/account/verify/login.php",
        "http://microsoft-account-security.com/verify/account/login.php",
        "http://google-account-security.com/login/verify/account.php",

        # 更多钓鱼URL变体
        "http://secure-paypal-login.com",
        "http://apple-account-verify.com",
        "http://amazon-account-security.com",
        "http://microsoft-account-verify.com",
        "http://google-account-security.com",
        "http://facebook-login-verify.com",
        "http://twitter-account-security.com",
        "http://linkedin-account-verify.com",
        "http://instagram-account-security.com",

        # 带特殊字符的钓鱼URL
        "http://paypal-security.com/login.php?redirect=phishing&token=fake",
        "http://apple-security.com/verify.php?user=victim&session=phishing",
        "http://amazon-security.com/account.php?session=phishing&id=fake",
        "http://microsoft-security.com/login.php?token=fake&redirect=phishing",
        "http://google-security.com/verify.php?auth=phishing&user=victim",

        # 长URL钓鱼
        "http://paypal-security-center.com/account/login/verify.php?redirect=phishing&token=fake&session=12345",
        "http://apple-id-verify.com/account/security/check.php?user=victim&session=phishing&token=fake",
        "http://amazon-security-check.com/account/verify/login.php?redirect=phishing&id=fake&session=12345",
        "http://microsoft-account-security.com/verify/account/login.php?token=fake&redirect=phishing&user=victim",
        "http://google-account-security.com/login/verify/account.php?auth=phishing&user=victim&session=fake",

        # 国际化钓鱼URL
        "http://paypal-security.com.cn/login.php",
        "http://apple-account-verify.com.cn/verify.php",
        "http://amazon-security.com.cn/account.php",
        "http://microsoft-security.com.cn/login.php",
        "http://google-security.com.cn/verify.php",

        # 带端口的钓鱼URL
        "http://paypal-security.com:8080/login.php",
        "http://apple-security.com:8443/verify.php",
        "http://amazon-security.com:8080/account.php",
        "http://microsoft-security.com:8443/login.php",
        "http://google-security.com:8080/verify.php",

        # 带路径深度的钓鱼URL
        "http://paypal-security.com/account/login/verify/secure/auth.php",
        "http://apple-security.com/account/security/verify/check/auth.php",
        "http://amazon-security.com/account/verify/login/secure/auth.php",
        "http://microsoft-security.com/verify/account/login/secure/auth.php",
        "http://google-security.com/login/verify/account/secure/auth.php",

        # 带多个参数的钓鱼URL
        "http://paypal-security.com/login.php?redirect=phishing&token=fake&session=12345&user=victim&id=fake",
        "http://apple-security.com/verify.php?user=victim&session=phishing&token=fake&id=12345&redirect=phishing",
        "http://amazon-security.com/account.php?session=phishing&id=fake&redirect=phishing&token=fake&user=victim",
        "http://microsoft-security.com/login.php?token=fake&redirect=phishing&user=victim&session=phishing&id=fake",
        "http://google-security.com/verify.php?auth=phishing&user=victim&session=fake&token=phishing&id=fake"
    ]

    # 提取特征
    benign_features = [extract_url_features(url) for url in benign_urls]
    phishing_features = [extract_url_features(url) for url in phishing_urls]

    # 创建数据框
    benign_df = pd.DataFrame(benign_features)
    benign_df['label'] = 1

    phishing_df = pd.DataFrame(phishing_features)
    phishing_df['label'] = 0

    # 合并数据
    df = pd.concat([benign_df, phishing_df], ignore_index=True)

    print(f"📊 数据集统计:")
    print(f"  总样本数: {len(df)}")
    print(f"  良性URL: {len(benign_df)} ({len(benign_df)/len(df)*100:.1f}%)")
    print(f"  钓鱼URL: {len(phishing_df)} ({len(phishing_df)/len(df)*100:.1f}%)")

    return df

def train_optimized_model():
    """训练优化的模型"""
    print("🚀 开始训练优化模型...")

    # 创建数据集
    df = create_enhanced_dataset()

    # 特征列
    feature_cols = ["url_len", "host_len", "path_len", "num_digits", "num_letters", "num_specials",
                   "num_dots", "num_hyphen", "num_slash", "num_qm", "num_at", "num_pct",
                   "has_ip", "subdomain_depth", "tld_suspicious", "has_punycode", "scheme_https",
                   "query_len", "fragment_len", "status_code", "bytes"]

    X = df[feature_cols].values
    y = df['label'].values

    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 转换为张量
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.LongTensor(y_test)

    # 创建模型
    model = FusionDNN(num_features=len(feature_cols))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练
    print("🔥 开始训练...")
    model.train()

    epochs = 100
    batch_size = 32

    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        # 随机打乱数据
        perm = torch.randperm(X_train_tensor.size(0))
        X_train_shuffled = X_train_tensor[perm]
        y_train_shuffled = y_train_tensor[perm]

        for i in range(0, len(X_train_tensor), batch_size):
            batch_X = X_train_shuffled[i:i+batch_size]
            batch_y = y_train_shuffled[i:i+batch_size]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # 评估
    print("📊 评估模型...")
    model.eval()

    with torch.no_grad():
        train_outputs = model(X_train_tensor)
        train_preds = torch.argmax(train_outputs, dim=1).numpy()
        train_acc = accuracy_score(y_train, train_preds)

        test_outputs = model(X_test_tensor)
        test_preds = torch.argmax(test_outputs, dim=1).numpy()
        test_acc = accuracy_score(y_test, test_preds)
        test_precision = precision_score(y_test, test_preds)
        test_recall = recall_score(y_test, test_preds)
        test_f1 = f1_score(y_test, test_preds)

    print(f"  训练准确率: {train_acc:.4f}")
    print(f"  测试准确率: {test_acc:.4f}")
    print(f"  精确率: {test_precision:.4f}")
    print(f"  召回率: {test_recall:.4f}")
    print(f"  F1分数: {test_f1:.4f}")

    # 保存模型
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'input_features': len(feature_cols),
        'feature_names': feature_cols,
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'training_accuracy': train_acc,
        'test_accuracy': test_acc,
        'precision': test_precision,
        'recall': test_recall,
        'f1_score': test_f1,
        'training_data_size': len(df),
        'benign_count': len(df[df['label'] == 1]),
        'phishing_count': len(df[df['label'] == 0])
    }

    torch.save(checkpoint, 'artifacts/fusion_optimized.pt')
    print("✅ 优化模型已保存到 artifacts/fusion_optimized.pt")

    return model, scaler, feature_cols

if __name__ == "__main__":
    train_optimized_model()