from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords
import nltk
import re


# Download stopwords if not already available
nltk.download('stopwords')

def clean_text(text):
    """Cleans and tokenizes text."""
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    text = text.lower()  # Convert to lowercase
    tokens = text.split()  # Tokenize text
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stop words
    return ' '.join(tokens)

def topic_modeling(df, n_topics=5, n_top_words=10):
    """Performs topic modeling on chat messages."""
    # Clean messages
    df['cleaned_message'] = df['message'].apply(clean_text)

    # Convert text to a document-term matrix
    vectorizer = CountVectorizer(max_features=1000)  # You can adjust max_features
    dtm = vectorizer.fit_transform(df['cleaned_message'])

    # Apply LDA
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(dtm)

    # Extract topics
    words = vectorizer.get_feature_names_out()
    topics = {}
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [words[i] for i in topic.argsort()[-n_top_words:][::-1]]
        topics[f"Topic {topic_idx + 1}"] = top_words

    return topics

# Example Usage
#topics = topic_modeling(df, n_topics=5, n_top_words=10)
#for topic, words in topics.items():
 #   print(f"{topic}: {', '.join(words)}")
