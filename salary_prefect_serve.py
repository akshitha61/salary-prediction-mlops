from salary_prefect_flow import salary_training_pipeline

if __name__ == "__main__":
    salary_training_pipeline.serve(
        name="salary-training-deployment",
        interval=60
    )