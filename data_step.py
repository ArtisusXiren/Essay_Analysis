import pandas as pd
from zenml.steps import step
@step
def data_step():
    df=pd.read_csv(r"C:\Users\ArtisusXiren\Downloads\summaries_train.csv")
    df2=pd.read_csv(r"C:\Users\ArtisusXiren\Downloads\prompts_train.csv")
    df3=pd.read_csv(r"C:\Users\ArtisusXiren\Downloads\summaries_test.csv")
    df4=pd.read_csv(r"C:\Users\ArtisusXiren\Downloads\prompts_test.csv")
    merge_df=pd.merge(df,df2, on='prompt_id',how='inner')
    return merge_df