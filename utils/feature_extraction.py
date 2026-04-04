# utils/feature_extraction.py
# Feature extraction module for phishing URL detection

import re
from urllib.parse import urlparse


# List of suspicious keywords commonly found in phishing URLs
SUSPICIOUS_WORDS = [
    'login', 'verify', 'bank', 'secure', 'account', 'update',
    'signin', 'confirm', 'billing', 'password', 'alert', 'suspended',
    'limited', 'unusual', 'validate', 'reactivate', 'unlock', 'claim',
    'prize', 'winner', 'free', 'gift', 'offer', 'bonus', 'reward',
    'paypal', 'amazon', 'apple', 'microsoft', 'google', 'facebook',
    'netflix', 'ebay', 'chase', 'wellsfargo', 'citibank', 'irs'
]

# Suspicious TLDs commonly used in phishing
SUSPICIOUS_TLDS = ['.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top', '.click', '.loan', '.work']

# Legitimate TLDs less likely to be phishing
LEGITIMATE_TLDS = ['.com', '.org', '.net', '.edu', '.gov', '.io', '.co']


def extract_features(url: str) -> list:
    """
    Extract a feature vector from a given URL.
    
    Features extracted:
    1.  url_length         - Total character length of URL
    2.  num_dots           - Number of '.' characters
    3.  num_hyphens        - Number of '-' characters
    4.  num_underscores    - Number of '_' characters
    5.  num_slashes        - Number of '/' characters
    6.  num_digits         - Count of numeric digits
    7.  has_at_symbol      - Presence of '@' (binary)
    8.  has_https          - Uses HTTPS scheme (binary)
    9.  has_ip_address     - IP address used as hostname (binary)
    10. subdomain_count    - Number of subdomains
    11. path_length        - Length of the URL path
    12. num_params         - Number of query parameters
    13. num_suspicious_words - Count of suspicious keywords
    14. has_suspicious_tld - TLD is in suspicious list (binary)
    15. double_slash_redirect - Contains '//' after initial scheme
    16. num_special_chars  - Count of special characters (@, ?, =, &, %)
    17. domain_length      - Length of the domain name
    18. has_port           - Whether a non-standard port is specified
    
    Returns:
        list: A list of numeric feature values
    """
    features = []
    url_lower = url.lower().strip()

    # --- Parse the URL ---
    try:
        parsed = urlparse(url_lower)
        scheme = parsed.scheme        # 'http' or 'https'
        netloc = parsed.netloc        # domain + port
        path = parsed.path            # path after domain
        query = parsed.query          # query string
    except Exception:
        parsed = None
        scheme, netloc, path, query = '', '', '', ''

    # Remove port from netloc for domain analysis
    hostname = netloc.split(':')[0] if netloc else ''

    # 1. URL length
    features.append(len(url))

    # 2. Number of dots
    features.append(url_lower.count('.'))

    # 3. Number of hyphens
    features.append(url_lower.count('-'))

    # 4. Number of underscores
    features.append(url_lower.count('_'))

    # 5. Number of forward slashes (excluding scheme slashes)
    path_slashes = path.count('/') if path else 0
    features.append(path_slashes)

    # 6. Number of digits in URL
    features.append(sum(c.isdigit() for c in url_lower))

    # 7. Presence of '@' symbol (used to trick browsers)
    features.append(1 if '@' in url_lower else 0)

    # 8. HTTPS usage
    features.append(1 if scheme == 'https' else 0)

    # 9. IP address as hostname (e.g., http://192.168.0.1/login)
    ip_pattern = re.compile(
        r'^(\d{1,3}\.){3}\d{1,3}$'
    )
    features.append(1 if ip_pattern.match(hostname) else 0)

    # 10. Subdomain count (parts of domain minus main domain + TLD)
    domain_parts = hostname.split('.')
    subdomain_count = max(0, len(domain_parts) - 2)
    features.append(subdomain_count)

    # 11. URL path length
    features.append(len(path))

    # 12. Number of query parameters
    param_count = len(query.split('&')) if query else 0
    features.append(param_count)

    # 13. Number of suspicious words in URL
    suspicious_count = sum(1 for word in SUSPICIOUS_WORDS if word in url_lower)
    features.append(suspicious_count)

    # 14. Suspicious TLD
    has_suspicious_tld = 0
    for tld in SUSPICIOUS_TLDS:
        if hostname.endswith(tld):
            has_suspicious_tld = 1
            break
    features.append(has_suspicious_tld)

    # 15. Double slash redirect (e.g., http://legit.com//evil.com)
    has_redirect = 1 if '//' in path else 0
    features.append(has_redirect)

    # 16. Number of special characters
    special_chars = sum(1 for c in url_lower if c in '@?=&%#!')
    features.append(special_chars)

    # 17. Domain/hostname length
    features.append(len(hostname))

    # 18. Non-standard port in URL
    has_port = 1 if (parsed and parsed.port and parsed.port not in (80, 443)) else 0
    features.append(has_port)

    return features


def get_feature_names() -> list:
    """Return the list of feature names (used for model training)."""
    return [
        'url_length',
        'num_dots',
        'num_hyphens',
        'num_underscores',
        'num_slashes',
        'num_digits',
        'has_at_symbol',
        'has_https',
        'has_ip_address',
        'subdomain_count',
        'path_length',
        'num_params',
        'num_suspicious_words',
        'has_suspicious_tld',
        'double_slash_redirect',
        'num_special_chars',
        'domain_length',
        'has_port'
    ]


def get_feature_description(url: str) -> dict:
    """
    Returns a human-readable dictionary of extracted features for a given URL.
    Useful for debugging and display in the UI.
    """
    values = extract_features(url)
    names = get_feature_names()
    return dict(zip(names, values))
