#!/usr/bin/env python3
"""
训练高级FusionDNN模型，进一步提升性能
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
import sys
sys.path.append('.')

from phishguard_v1.models.fusion_model import FusionDNN, predict_proba

def extract_advanced_features(url):
    """提取高级特征"""
    if not url.startswith("http"):
        url = "http://" + url

    features = {}

    # 基础长度特征
    features["url_len"] = len(url)
    host = url.split('//')[-1].split('/')[0]
    path = '/' + '/'.join(url.split('/')[3:]) if len(url.split('/')) > 3 else '/'
    features["host_len"] = len(host)
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
    features["num_equal"] = url.count('=')
    features["num_amp"] = url.count('&')
    features["num_plus"] = url.count('+')
    features["num_hash"] = url.count('#')

    # 布尔特征
    features["has_ip"] = any(part.isdigit() for part in host.split('.'))
    features["subdomain_depth"] = host.count('.') if host != 'localhost' else 0
    features["tld_suspicious"] = 1 if any(tld in host.lower() for tld in ['.tk', '.ml', '.ga', '.cf', '.top', '.click']) else 0
    features["has_punycode"] = 1 if 'xn--' in host.lower() else 0
    features["scheme_https"] = 1 if url.startswith('https') else 0

    # 查询和片段长度
    query = url.split('?')[-1] if '?' in url else ''
    features["query_len"] = len(query)
    fragment = url.split('#')[-1] if '#' in url else ''
    features["fragment_len"] = len(fragment)

    # 高级特征
    features["path_depth"] = len([p for p in path.split('/') if p]) if path != '/' else 0
    features["has_params"] = 1 if '?' in url else 0
    features["num_params"] = len([p for p in query.split('&') if p]) if query else 0
    features["has_file_ext"] = 1 if any(ext in path.lower() for ext in ['.php', '.html', '.htm', '.asp', '.aspx', '.jsp', '.cgi', '.pl']) else 0
    features["is_suspicious_file"] = 1 if any(ext in path.lower() for ext in ['.exe', '.bat', '.cmd', '.scr', '.pif']) else 0

    # 域名特征
    features["domain_len"] = len(host.split('.')[0]) if '.' in host else len(host)
    features["has_www"] = 1 if host.startswith('www.') else 0
    features["is_long_domain"] = 1 if len(host) > 30 else 0

    # 字符比例特征
    total_chars = len(url)
    if total_chars > 0:
        features["digit_ratio"] = features["num_digits"] / total_chars
        features["special_ratio"] = features["num_specials"] / total_chars
        features["letter_ratio"] = features["num_letters"] / total_chars
    else:
        features["digit_ratio"] = 0
        features["special_ratio"] = 0
        features["letter_ratio"] = 0

    # HTTP响应特征（模拟）
    features["status_code"] = 200
    features["bytes"] = 1024

    return features

class AdvancedFusionDNN(nn.Module):
    """高级FusionDNN模型"""
    def __init__(self, num_features):
        super(AdvancedFusionDNN, self).__init__()
        self.fc1 = nn.Linear(num_features, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.2)

        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.dropout4 = nn.Dropout(0.2)

        self.fc5 = nn.Linear(32, 2)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)

        x = self.relu(self.bn4(self.fc4(x)))
        x = self.dropout4(x)

        x = self.fc5(x)
        return x

def create_comprehensive_dataset():
    """创建全面的数据集"""
    print("🔧 创建全面数据集...")

    # 良性URL - 更多样化的良性URL
    benign_urls = [
        # 基础知名域名
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
        "https://www.github.com",
        "https://www.stackoverflow.com",
        "https://www.reddit.com",
        "https://www.medium.com",
        "https://www.quora.com",

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
        "https://www.sohu.com",
        "https://www.ifeng.com",
        "https://www.360.cn",
        "https://www.alibaba.com",
        "https://www.tencent.com",
        "https://www.netease.com",
        "https://www.bytedance.com",
        "https://www.meituan.com",
        "https://www.didi.com",

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
        "https://github.com/torvalds/linux",
        "https://stackoverflow.com/questions/123456/how-to-code",
        "https://www.medium.com/@username/article-title",
        "https://www.reddit.com/r/programming/comments/123456",

        # API和服务URL
        "https://api.github.com/user/repos",
        "https://graph.facebook.com/v2.0/me",
        "https://maps.googleapis.com/maps/api/geocode/json",
        "https://api.twitter.com/2/tweets/search/recent",
        "https://api.weixin.qq.com/cgi-bin/token",
        "https://api.linkedin.com/v2/me",
        "https://api.instagram.com/v1/users/self",
        "https://www.googleapis.com/drive/v3/files",
        "https://api.amazon.com/user/profile",
        "https://api.taobao.com/router/rest",

        # 静态资源
        "https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png",
        "https://www.facebook.com/rsrc.php/v3/yj/r/2sGX-1s7yfD.png",
        "https://www.baidu.com/img/PCtm_d9c8750bed0b3c7d089fa7d55720d6cf.png",
        "https://www.taobao.com/favicon.ico",
        "https://www.jd.com/favicon.ico",
        "https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css",
        "https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js",
        "https://unpkg.com/react@18/umd/react.production.min.js",

        # 登录和认证
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

        # 电商产品
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

        # 新闻和媒体
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
        "https://www.people.com.cn/",
        "https://www.xinhuanet.com/",
        "https://www.chinadaily.com.cn/",
        "https://www.globaltimes.cn/",
        "https://www.caixin.com/",

        # 搜索结果
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
        "https://www.sogou.com/web?query=python",
        "https://www.so.com/s?q=人工智能",

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
        "https://www.python.org/downloads/",
        "https://www.openoffice.org/download/",
        "https://www.ubuntu.com/download/desktop",

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
        "https://www.cnblogs.com/test/p/123456.html",
        "https://segmentfault.com/a/123456",
        "https://www.oschina.net/question/123456",
        "https://www.infoq.cn/article/123456",
        "https://www.juejin.cn/post/123456",

        # 政府和教育机构
        "https://www.gov.cn/",
        "https://www.people.com.cn/",
        "https://www.xinhuanet.com/",
        "https://www.mit.edu/",
        "https://www.stanford.edu/",
        "https://www.harvard.edu/",
        "https://www.berkeley.edu/",
        "https://www.tsinghua.edu.cn/",
        "https://www.pku.edu.cn/",
        "https://www.fudan.edu.cn/",
        "https://www.zju.edu.cn/",
        "https://www.ustc.edu.cn/",

        # 金融服务
        "https://www.icbc.com.cn/",
        "https://www.bankofamerica.com/",
        "https://www.chase.com/",
        "https://www.wellsfargo.com/",
        "https://www.citibank.com/",
        "https://www.hsbc.com/",
        "https://www.standardchartered.com/",
        "https://www.barclays.co.uk/",
        "https://www.goldmansachs.com/",
        "https://www.morganstanley.com/",

        # 医疗健康
        "https://www.who.int/",
        "https://www.mayoclinic.org/",
        "https://www.webmd.com/",
        "https://www.healthline.com/",
        "https://www.medicalnewstoday.com/",
        "https://www.hopkinsmedicine.org/",
        "https://www.clevelandclinic.org/",
        "https://www.mountsinai.org/",
        "https://www.brighamandwomens.org/",
        "https://www.uchicago.edu/",

        # 旅游和酒店
        "https://www.booking.com/",
        "https://www.expedia.com/",
        "https://www.airbnb.com/",
        "https://www.marriott.com/",
        "https://www.hilton.com/",
        "https://www.hyatt.com/",
        "https://www.ihg.com/",
        "https://www.wyndhamhotels.com/",
        "https://www.choicehotels.com/",
        "https://www.accor.com/"
    ]

    # 钓鱼URL - 更全面的钓鱼URL
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
        "http://yahoo-account-security.com",
        "http://linkedin-security-check.com",
        "http://dropbox-security-center.com",
        "http://microsoft-security-center.com",
        "http://google-security-center.com",
        "http://facebook-security-center.com",
        "http://twitter-security-center.com",
        "http://instagram-security-center.com",

        # 带路径的钓鱼URL
        "http://verify-paypal-account.com/login.php",
        "http://apple-id-apple.com.verify/index.php",
        "http://amazon-security.com/verify-account/",
        "http://microsoft-security.com/account/login",
        "http://google-security.com/account/verify",
        "http://facebook-security.com/login/verify",
        "http://twitter-security.com/account/login",
        "http://linkedin-security.com/account/verify",
        "http://instagram-security.com/account/login",
        "http://yahoo-security.com/account/verify",
        "http://dropbox-security.com/account/verify",
        "http://microsoft-account-security.com/login.php",
        "http://google-account-security.com/verify.php",
        "http://facebook-account-security.com/login.php",
        "http://twitter-account-security.com/verify.php",
        "http://instagram-account-security.com/login.php",
        "http://yahoo-account-security.com/account.php",

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
        "http://yahoo-account-security.com/verify.php",
        "http://dropbox-account-security.com/login.php",
        "http://microsoft-account-security-center.com/verify.php",
        "http://google-account-security-center.com/login.php",
        "http://facebook-account-security-center.com/verify.php",
        "http://twitter-account-security-center.com/login.php",
        "http://instagram-account-security-center.com/verify.php",
        "http://yahoo-account-security-center.com/account.php",

        # 带参数的钓鱼URL
        "http://paypal-security.com/login.php?redirect=phishing",
        "http://apple-security.com/verify.php?user=victim",
        "http://amazon-security.com/account.php?session=phishing",
        "http://microsoft-security.com/login.php?token=fake",
        "http://google-security.com/verify.php?auth=phishing",
        "http://facebook-security.com/login.php?next=phishing",
        "http://twitter-security.com/account.php?oauth=phishing",
        "http://linkedin-security.com/verify.php?code=phishing",
        "http://instagram-security.com/login.php?csrf=phishing",
        "http://yahoo-security.com/account.php?sid=phishing",
        "http://dropbox-security.com/login.php?return=phishing",
        "http://microsoft-security.com/verify.php?id=fake",
        "http://google-security.com/login.php?user=victim",
        "http://facebook-security.com/verify.php?token=fake",
        "http://twitter-security.com/login.php?auth=phishing",
        "http://instagram-security.com/verify.php?user=victim",

        # IP地址钓鱼URL
        "http://192.168.1.1/paypal/login.php",
        "http://123.456.789.012/apple/verify.php",
        "http://10.0.0.1/amazon/login.php",
        "http://172.16.0.1/microsoft/verify.php",
        "http://203.0.113.0/google/login.php",
        "http://198.51.100.0/facebook/verify.php",
        "http://192.0.2.0/twitter/login.php",
        "http://203.0.113.1/linkedin/verify.php",
        "http://198.51.100.1/instagram/login.php",
        "http://192.0.2.1/yahoo/account.php",

        # 短链接钓鱼URL
        "http://bit.ly/verify-paypal",
        "http://tinyurl.com/apple-security",
        "http://goo.gl/amazon-verify",
        "http://ow.ly/microsoft-login",
        "http://is.gd/google-security",
        "http://t.co/facebook-verify",
        "http://buff.ly/twitter-security",
        "http://rebrand.ly/linkedin-verify",
        "http://clk.im/instagram-security",
        "http://short.link/yahoo-account",

        # 带子域名的钓鱼URL
        "http://login.paypal.com.verify.com",
        "http://apple.id.apple.com.security.com",
        "http://amazon.account.amazon.com.security.com",
        "http://microsoft.account.microsoft.com.security.com",
        "http://google.account.google.com.security.com",
        "http://facebook.login.facebook.com.verify.com",
        "http://twitter.account.twitter.com.security.com",
        "http://linkedin.profile.linkedin.com.verify.com",
        "http://instagram.account.instagram.com.security.com",
        "http://yahoo.mail.yahoo.com.verify.com",

        # 混合钓鱼URL
        "http://paypal-security-center.com/account/login/verify.php",
        "http://apple-id-verify.com/account/security/check.php",
        "http://amazon-security-check.com/account/verify/login.php",
        "http://microsoft-account-security.com/verify/account/login.php",
        "http://google-account-security.com/login/verify/account.php",
        "http://facebook-security-center.com/account/login/verify.php",
        "http://twitter-account-verify.com/account/security/login.php",
        "http://linkedin-security-center.com/account/verify/login.php",
        "http://instagram-security-check.com/account/security/verify.php",
        "http://yahoo-account-security.com/verify/account/login.php",

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
        "http://yahoo-account-verify.com",
        "http://dropbox-security-center.com",
        "http://microsoft-login-verify.com",
        "http://google-login-verify.com",
        "http://facebook-security-center.com",
        "http://twitter-login-verify.com",
        "http://linkedin-security-center.com",
        "http://instagram-login-verify.com",
        "http://yahoo-security-center.com",

        # 带特殊字符的钓鱼URL
        "http://paypal-security.com/login.php?redirect=phishing&token=fake",
        "http://apple-security.com/verify.php?user=victim&session=phishing",
        "http://amazon-security.com/account.php?session=phishing&id=fake",
        "http://microsoft-security.com/login.php?token=fake&redirect=phishing",
        "http://google-security.com/verify.php?auth=phishing&user=victim",
        "http://facebook-security.com/login.php?next=phishing&token=fake",
        "http://twitter-security.com/account.php?oauth=phishing&user=victim",
        "http://linkedin-security.com/verify.php?code=phishing&token=fake",
        "http://instagram-security.com/login.php?csrf=phishing&user=victim",
        "http://yahoo-security.com/account.php?sid=phishing&token=fake",

        # 长URL钓鱼
        "http://paypal-security-center.com/account/login/verify.php?redirect=phishing&token=fake&session=12345",
        "http://apple-id-verify.com/account/security/check.php?user=victim&session=phishing&token=fake",
        "http://amazon-security-check.com/account/verify/login.php?redirect=phishing&id=fake&session=12345",
        "http://microsoft-account-security.com/verify/account/login.php?token=fake&redirect=phishing&user=victim",
        "http://google-account-security.com/login/verify/account.php?auth=phishing&user=victim&session=fake",
        "http://facebook-security-center.com/account/login/verify.php?next=phishing&token=fake&user=victim",
        "http://twitter-account-verify.com/account/security/login.php?oauth=phishing&user=victim&token=fake",
        "http://linkedin-security-center.com/account/verify/login.php?code=phishing&token=fake&user=victim",
        "http://instagram-security-check.com/account/security/verify.php?csrf=phishing&user=victim&token=fake",
        "http://yahoo-account-security.com/verify/account/login.php?sid=phishing&token=fake&user=victim",

        # 国际化钓鱼URL
        "http://paypal-security.com.cn/login.php",
        "http://apple-account-verify.com.cn/verify.php",
        "http://amazon-security.com.cn/account.php",
        "http://microsoft-security.com.cn/login.php",
        "http://google-security.com.cn/verify.php",
        "http://facebook-security.com.cn/login.php",
        "http://twitter-security.com.cn/account.php",
        "http://linkedin-security.com.cn/verify.php",
        "http://instagram-security.com.cn/login.php",
        "http://yahoo-security.com.cn/account.php",

        # 带端口的钓鱼URL
        "http://paypal-security.com:8080/login.php",
        "http://apple-security.com:8443/verify.php",
        "http://amazon-security.com:8080/account.php",
        "http://microsoft-security.com:8443/login.php",
        "http://google-security.com:8080/verify.php",
        "http://facebook-security.com:8080/login.php",
        "http://twitter-security.com:8443/account.php",
        "http://linkedin-security.com:8080/verify.php",
        "http://instagram-security.com:8443/login.php",
        "http://yahoo-security.com:8080/account.php",

        # 带路径深度的钓鱼URL
        "http://paypal-security.com/account/login/verify/secure/auth.php",
        "http://apple-security.com/account/security/verify/check/auth.php",
        "http://amazon-security.com/account/verify/login/secure/auth.php",
        "http://microsoft-security.com/verify/account/login/secure/auth.php",
        "http://google-security.com/login/verify/account/secure/auth.php",
        "http://facebook-security.com/account/login/verify/secure/auth.php",
        "http://twitter-security.com/account/security/login/secure/auth.php",
        "http://linkedin-security.com/account/verify/login/secure/auth.php",
        "http://instagram-security.com/account/security/verify/secure/auth.php",
        "http://yahoo-security.com/verify/account/login/secure/auth.php",

        # 带多个参数的钓鱼URL
        "http://paypal-security.com/login.php?redirect=phishing&token=fake&session=12345&user=victim&id=fake",
        "http://apple-security.com/verify.php?user=victim&session=phishing&token=fake&id=12345&redirect=phishing",
        "http://amazon-security.com/account.php?session=phishing&id=fake&redirect=phishing&token=fake&user=victim",
        "http://microsoft-security.com/login.php?token=fake&redirect=phishing&user=victim&session=phishing&id=fake",
        "http://google-security.com/verify.php?auth=phishing&user=victim&session=fake&token=phishing&id=fake",
        "http://facebook-security.com/login.php?next=phishing&token=fake&user=victim&session=phishing&id=fake",
        "http://twitter-security.com/account.php?oauth=phishing&user=victim&session=phishing&token=fake&id=fake",
        "http://linkedin-security.com/verify.php?code=phishing&token=fake&user=victim&session=phishing&id=fake",
        "http://instagram-security.com/login.php?csrf=phishing&user=victim&session=phishing&token=fake&id=fake",
        "http://yahoo-security.com/account.php?sid=phishing&token=fake&user=victim&session=phishing&id=fake",

        # 最新钓鱼URL模式
        "http://secure-account-verification.com/paypal",
        "http://account-security-check.com/apple",
        "http://user-verification-center.com/amazon",
        "http://login-secure-verification.com/microsoft",
        "http://account-security-verify.com/google",
        "http://secure-login-center.com/facebook",
        "http://user-authentication-center.com/twitter",
        "http://account-verification-service.com/linkedin",
        "http://secure-user-verification.com/instagram",
        "http://login-security-center.com/yahoo"
    ]

    # 提取特征
    benign_features = [extract_advanced_features(url) for url in benign_urls]
    phishing_features = [extract_advanced_features(url) for url in phishing_urls]

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

def train_advanced_model():
    """训练高级模型"""
    print("🚀 开始训练高级模型...")

    # 创建数据集
    df = create_comprehensive_dataset()

    # 获取特征列
    feature_cols = [col for col in df.columns if col != 'label']
    print(f"📊 特征数量: {len(feature_cols)}")

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
    model = AdvancedFusionDNN(num_features=len(feature_cols))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.7)

    # 训练
    print("🔥 开始训练...")
    model.train()

    epochs = 150
    batch_size = 32
    best_loss = float('inf')
    patience = 20
    patience_counter = 0

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
        scheduler.step()

        # 早停机制
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

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

        # 混淆矩阵
        cm = confusion_matrix(y_test, test_preds)
        print(f"  混淆矩阵:")
        print(f"    TN: {cm[0][0]}, FP: {cm[0][1]}")
        print(f"    FN: {cm[1][0]}, TP: {cm[1][1]}")

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
        'phishing_count': len(df[df['label'] == 0]),
        'confusion_matrix': cm.tolist(),
        'model_type': 'AdvancedFusionDNN'
    }

    torch.save(checkpoint, 'artifacts/fusion_advanced.pt')
    print("✅ 高级模型已保存到 artifacts/fusion_advanced.pt")

    return model, scaler, feature_cols

if __name__ == "__main__":
    train_advanced_model()