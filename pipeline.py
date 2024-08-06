from zenml.pipelines import pipeline
from data_step import data_step
from preprocess_step import preprocess_step
from Tokenization import tokenizer
from model_training import model_training
from result import logging_step
@pipeline
def text_pipeline(data_loader,preprocessor,tokenizer,model_training,res):
    df=data_loader()
    preprocess_df=preprocessor(df)
    tokenizer_df,arrays_answer,arrays_question=tokenizer(preprocess_df)
    mae_wordings, mae_content=model_training(tokenizer_df,arrays_answer,arrays_question)
    res(mae_wordings,mae_content)
    
pipeline_instance=text_pipeline(
data_loader=data_step(),
preprocessor=preprocess_step(),
tokenizer=tokenizer(),
model_training=model_training(),   
res=logging_step()
)
pipeline_instance.run()