from transformers import pipeline

def generate_business_plan(prompt, max_length=500):
    model_path = "../model/fine_tuned_model"
    business_plan_generator = pipeline("text-generation", model=model_path)
    return business_plan_generator(prompt, max_length=max_length)

if __name__ == "__main__":
    prompt = "Generate a business plan for a $500K E-2 Visa startup"
    response = generate_business_plan(prompt)
    print(response)