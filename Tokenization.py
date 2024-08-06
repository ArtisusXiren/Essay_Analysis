from nltk import WhitespaceTokenizer
import numpy as np
import pandas as pd
from zenml.steps import step
from typing import Tuple, List
@step
def tokenizer(merge_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[np.ndarray], List[np.ndarray]]:
    tokenizer=WhitespaceTokenizer()
    prompt=[tokenizer.tokenize(ch) for ch in merge_df['prompt_question'] ]
    question=[x for x in prompt]
    prompt_=[tokenizer.tokenize(ch) for ch in merge_df['text'] ]
    answer=[x for x in prompt_]
    tokens=list(set([token for sublist in answer for token in sublist]))
    answer_index={token:i for i,token in enumerate(tokens)}
    Tokens=list(set([token for sublist in question for token in sublist]))
    question_index={token:i for i,token in enumerate(Tokens)}
    answer_encoded_hot=np.zeros((len(answer),len(tokens)),dtype=np.float64)
    question_encoded_hot=np.zeros((len(question),len(Tokens)),dtype=np.float64)
    for i,ch in enumerate(answer):
        for token in ch:
            answer_encoded_hot[i,answer_index[token]]=1
    for i,ch in enumerate(question):
          for token in ch:
            question_encoded_hot[i,question_index[token]]=1
    array_of_answers = [row for row in answer_encoded_hot ]
    array_of_question = [row for row in question_encoded_hot ]
    max_shape = max(arr.shape for arr in array_of_answers)
    arrays_answer = [np.pad(arr, [(0, max_shape[0] - arr.shape[0])], mode='constant') for arr in array_of_answers]
    arrays_answer = [arr.astype(np.float64) for arr in arrays_answer]
    max_shape = max(arr.shape for arr in array_of_question)
    arrays_question = [np.pad(arr, [(0, max_shape[0] - arr.shape[0])], mode='constant') for arr in array_of_question]
    arrays_question = [arr.astype(np.float64) for arr in arrays_question]
    merge_df['answer']=arrays_answer
    merge_df['question']=arrays_question
    merge_df=merge_df.loc[:,['student_id','prompt_id','question','answer','content','wording']]
    return merge_df,arrays_answer,arrays_question