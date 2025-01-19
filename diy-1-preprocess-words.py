import os
import re

# Put all the CLIP 'opinions' .txt (from gradient ascent) in a folder (can have subfolders):
words_folder = "texts"
output_folder = "words_processing"
raw_words_file = os.path.join(output_folder, "raw_words.txt")

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# Helper function to clean words
def clean_word(word):
    # Remove non-printable characters and ANSI escape codes
    ansi_escape = re.compile(r'(?:\x1B[@-_][0-?]*[ -/]*[@-~])|[^\x20-\x7E]+')
    return ansi_escape.sub('', word).strip()

# Process all files
all_words = set()  # Use a set for deduplication
total_words_count = 0

for root, _, files in os.walk(words_folder):
    for file in files:
        if file.endswith(".txt"):
            file_path = os.path.join(root, file)
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    # Split words by space and clean them
                    words = [clean_word(word) for word in line.split()]
                    # Filter out empty words after cleaning
                    words = [word for word in words if word]
                    # Add to set and update total count
                    all_words.update(words)
                    total_words_count += len(words)

# Save unique words to raw_words.txt
with open(raw_words_file, "w", encoding="utf-8") as f:
    f.write("\n".join(sorted(all_words)))

# Print stats
unique_words_count = len(all_words)
print(f"Saved words to {raw_words_file}")
print(f"Total words processed: {total_words_count}")
print(f"Unique words found: {unique_words_count}")