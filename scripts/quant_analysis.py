import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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