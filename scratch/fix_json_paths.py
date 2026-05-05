import os
import re

target_dir = r'static/comparison_outputs'
pattern = re.compile(r'[a-zA-Z]:\\+Antigravity\\+vs13-model\\+', re.IGNORECASE)

print(f"Cleaning up JSON paths in {target_dir}...")

count = 0
for root, dirs, files in os.walk(target_dir):
    for f in files:
        if f.endswith('.json'):
            path = os.path.join(root, f)
            with open(path, 'r', encoding='utf-8') as jf:
                content = jf.read()
            
            if pattern.search(content):
                new_content = pattern.sub('../', content)
                # Also handle forward slashes
                new_content = new_content.replace('\\\\', '/')
                new_content = new_content.replace('\\', '/')
                # Ensure no double slashes like ..//
                new_content = new_content.replace('//', '/')
                
                with open(path, 'w', encoding='utf-8') as jf2:
                    jf2.write(new_content)
                count += 1

print(f"Done. Fixed {count} files.")
