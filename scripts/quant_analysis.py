import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

class QuantitativeAnalysis:
    def __init__ (self, dataframe: pd.DataFrame):
        """
        Initialize with a pandas DataFrame.
        """
        self.df= dataframe.copy()

    def calculate_headline_length(self):
        """
        Creates a new column 'headline_length' with the number of characters in each headline.
        """
        self.df["headline"] = self.df["headline"].astype(str)
        self.df["headline_length"]= self.df["headline"].apply(len)
        print("[INFO] Headline length column created.")
        return self.df
    


    def describe_headline_length(self):
        """
        Prints descriptive statistics for the headline lengths.
        """
        if 'headline_length' not in self.df.columns:
            print("[WARN] 'headline_length' not found. Calculating it first...")
            self.calculate_headline_length()

        stats = self.df['headline_length'].describe()
        print("[INFO] Headline Length Description:\n", stats)
        return stats


    def plot_headline_length_distribution(self, bins=30):
        """
        Plots a histogram of the headline lengths.
        """
        if 'headline_length' not in self.df.columns:
            print("[WARN] 'headline_length' not found. Calculating it first...")
            self.calculate_headline_length()

        sns.histplot(self.df['headline_length'], bins=bins, kde=True)
        plt.title("Distribution of Headline Lengths")
        plt.xlabel("Number of Characters")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def count_articles_per_publisher(self, top_n=10):
        """
        Counts the number of articles per publisher and optionally plots the top N.

        Args:
            top_n (int): Number of top publishers to display.

        Returns:
            pd.Series: Publisher counts sorted descending.
        """
        # Clean publisher names
        self.df['publisher'] = self.df['publisher'].astype(str).str.strip().str.lower()

        # Count occurrences
        publisher_counts = self.df['publisher'].value_counts()

        print(f"[INFO] Top {top_n} publishers by article count:\n", publisher_counts.head(top_n))
        
        sns.barplot(x=publisher_counts.head(top_n).values,
                    y=publisher_counts.head(top_n).index,
                    palette="mako")
        plt.title(f"Top {top_n} Publishers by Article Count")
        plt.xlabel("Number of Articles")
        plt.ylabel("Publisher")
        plt.grid(True, axis='x', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

        return publisher_counts
    
    def analyze_publication_trends(self, frequency="D", plot=True):
        """
        Analyze article publication trends over time.

        Parameters:
        - frequency (str): Time aggregation level ('D' for day, 'W' for week, 'M' for month).
        - plot (bool): Whether to plot the publication frequency.

        Returns:
        - pandas.Series with counts indexed by date.
        """
        if 'date' not in self.df.columns:
            raise ValueError("'date' column not found in the dataset.")

        # Convert to datetime if not already
        # self.df['date'] = pd.to_datetime(self.df['date'], utc=True)
        self.df['date'] = pd.to_datetime(self.df['date'], format='mixed', utc=True, errors='coerce')

        # Group by date
        trend = self.df.set_index('date').resample(frequency).size()

        if plot:
            trend.plot(figsize=(12, 5))
            plt.title(f"Publication Frequency Over Time ({frequency})")
            plt.xlabel("Date")
            plt.ylabel("Number of Articles")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        return trend
    
    def extract_keywords_tfidf(self, top_n=20):
        """
        Extract top keywords using TF-IDF from the headline column.

        Parameters:
        - top_n (int): number of top keywords to return.

        Returns:
        - List of top keywords.
        """
        stop_words = stopwords.words('english')

        tfidf = TfidfVectorizer(
            stop_words=stop_words,
            max_features=top_n,
            lowercase=True,
            token_pattern=r'\b[a-zA-Z]{3,}\b'
        )

        self.df['headline'] = self.df['headline'].astype(str)
        X = tfidf.fit_transform(self.df['headline'])

        top_keywords = tfidf.get_feature_names_out()
        print(f"[INFO] Top {top_n} TF-IDF Keywords:\n", top_keywords)
        return top_keywords.tolist()


    def extract_topics_lda(self, n_topics=5, n_words=5):
        """
        Perform LDA topic modeling on the headlines.

        Parameters:
        - n_topics (int): number of topics to extract.
        - n_words (int): number of top words per topic.

        Returns:
        - List of topics (each topic is a list of words).
        """
        stop_words = stopwords.words('english')

        vectorizer = CountVectorizer(
            stop_words=stop_words,
            lowercase=True,
            token_pattern=r'\b[a-zA-Z]{3,}\b'
        )

        self.df['headline'] = self.df['headline'].astype(str)
        X = vectorizer.fit_transform(self.df['headline'])

        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(X)

        words = vectorizer.get_feature_names_out()
        topics = []

        for idx, topic in enumerate(lda.components_):
            top_words = [words[i] for i in topic.argsort()[-n_words:]]
            print(f"Topic {idx + 1}: {top_words}")
            topics.append(top_words)

        return topics
