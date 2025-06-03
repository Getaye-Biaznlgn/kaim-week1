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