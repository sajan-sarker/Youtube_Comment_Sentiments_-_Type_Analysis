import numpy as np
import pandas as pd
from youtube_comment_downloader import YoutubeCommentDownloader

from scripts.utils import *

def clean_comment(comment: str) -> str:
    comment = comment.lower().strip()
    comment = remove_emojis(comment)
    comment = remove_html_tags(comment)
    comment = remove_urls(comment)
    comment = remove_punctuation(comment)
    comment = remove_special_characters(comment)
    comment = replace_slang(comment)
    return str(comment)

def download_comments(url: str) -> pd.DataFrame:
    downloader = YoutubeCommentDownloader()
    comments = downloader.get_comments_from_url(url)

    # object to array
    data = []
    for comment in comments:
        data.append(comment['text'])

    comments = pd.DataFrame(data, columns=['comment'])              # create dataframe
    comments['comment'] = comments['comment'].apply(clean_comment)      # clean the comments

    # replace empty rows with nan value
    for index, row in comments.iterrows():
        if str(row['comment']).strip() == "":
            comments.at[index, 'comment'] = np.nan

    comments.dropna(subset=['comment'], inplace=True)                 # remove the nan values

    return comments