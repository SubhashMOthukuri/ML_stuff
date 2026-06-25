"""
Better Airbnb downloader with headers
Fixes 403 Forbidden error
"""

import os
import requests
from tqdm import tqdm

os.makedirs('data/airbnb', exist_ok=True)

# Add headers to look like browser
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
}

# URLs
urls = {
    'listings': 'https://data.insideairbnb.com/united-states/ny/new-york-city/2025-06-02/data/listings.csv.gz',
    'reviews': 'https://data.insideairbnb.com/united-states/ny/new-york-city/2025-06-02/data/reviews.csv.gz',
}

print("="*60)
print("AIRBNB DATA DOWNLOADER (v2)")
print("="*60)

for name, url in urls.items():
    filepath = f'data/airbnb/{name}.csv.gz'
    
    print(f"\n📥 Downloading {name}...")
    print(f"   URL: {url}")
    
    try:
        response = requests.get(url, headers=headers, timeout=30, stream=True)
        response.raise_for_status()
        
        # Get file size
        total_size = int(response.headers.get('content-length', 0))
        
        # Download with progress bar
        with open(filepath, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size:
                        pct = (downloaded / total_size) * 100
                        print(f"   Progress: {pct:.1f}%", end='\r')
        
        size_mb = os.path.getsize(filepath) / 1024**2
        print(f"   ✓ Downloaded: {size_mb:.0f}MB")
        
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Error: {e}")
        print(f"   💡 Tip: Download manually from https://insideairbnb.com/")

print("\n" + "="*60)
print("✅ DONE!")
print("="*60)