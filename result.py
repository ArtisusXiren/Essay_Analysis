from zenml.steps import step

@step
def logging_step(mae_wordings, mae_content):
    print(f"Mean Absolute Error for wordings: {mae_wordings}")
    print(f"Mean Absolute Error for content: {mae_content}")