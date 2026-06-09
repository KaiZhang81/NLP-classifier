from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


file_path = Path(__file__).resolve().parent / "FinancialPhraseBank-v1.0" / "Sentences_50Agree.txt"

# '@' are used to separate text and sentiment, 'latin-1' encoding
df = pd.read_csv(file_path, sep='@', names=['Text', 'Sentiment'], encoding='latin-1')


# bar chart for positive, neutral, and negative instances 
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Sentiment')
plt.title('Distribution of Sentiments')
plt.show()


# counts the words per sentence and plots a histogram to show text lengths
df['Word_Count'] = df['Text'].apply(lambda x: len(str(x).split()))
plt.figure(figsize=(8, 4))
sns.histplot(data=df, x='Word_Count', hue='Sentiment', kde=True, bins=30)
plt.title('Word Count Distribution per Sentiment')
plt.show()


# cloud of most used words seperated in positive and negative
pos_text = " ".join(df[df['Sentiment'] == 'positive']['Text'].astype(str))
neg_text = " ".join(df[df['Sentiment'] == 'negative']['Text'].astype(str))

wc_pos = WordCloud(width=400, height=400, background_color='white').generate(pos_text)
wc_neg = WordCloud(width=400, height=400, background_color='white').generate(neg_text)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(wc_pos, interpolation='bilinear')
axes[0].set_title('Positive Words')
axes[0].axis("off")

axes[1].imshow(wc_neg, interpolation='bilinear')
axes[1].set_title('Negative Words')
axes[1].axis("off")

plt.show()