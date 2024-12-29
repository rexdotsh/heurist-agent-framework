import re

def strip_tweet_text(text: str) -> str:
    # Remove URLs
    text_cleaned = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    # Remove @ mentions
    text_cleaned = re.sub(r'@\w+', '', text_cleaned).strip()
    return text_cleaned