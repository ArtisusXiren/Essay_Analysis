import re 
from string import digits
import string 
import pandas as pd
from zenml.steps import step
@step
def preprocess_step(merge_df: pd.DataFrame) -> pd.DataFrame:
    merge_df=merge_df.loc[:,['student_id','prompt_id','prompt_question','prompt_text','text','content','wording']]
    def preprocess(text: str)-> str:
        text=text.lower()
        text=text.strip()
        text=re.sub("'","",text)
        text=re.sub(" +","",text)
        exclude=set(string.punctuation)
        text=''.join(ch for ch in text if ch not in exclude)
        remove=str.maketrans('','',digits)
        text=text.translate(remove)
        return text
    merge_df['prompt_question']=merge_df['prompt_question'].apply(preprocess)
    merge_df['text']=merge_df['text'].apply(preprocess)
    return merge_df