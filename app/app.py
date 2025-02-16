import streamlit as st
import json
import sys
import os
import redis
from pathlib import Path
from importlib import import_module

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Initialize Redis connection
redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST', 'redis'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    db=0
)

# Cache configuration
CACHE_TTL = 3600  # Cache business plans for 1 hour

# Lazy load generate_plan to avoid PyTorch initialization conflicts
def get_generator():
    return import_module('src.generate_plan').generate_business_plan

# Load model only once and cache it
@st.cache_resource
def load_generator():
    return get_generator()

# Cache business plan generation results
def get_cached_plan(prompt):
    cache_key = f"business_plan:{hash(prompt)}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return json.loads(cached_result)
    return None

def cache_plan(prompt, plan):
    cache_key = f"business_plan:{hash(prompt)}"
    redis_client.setex(
        cache_key,
        CACHE_TTL,
        json.dumps(plan)
    )

def load_questionnaire():
    questionnaire_path = os.path.join(project_root, "templates", "questionnaire_template.json")
    with open(questionnaire_path, "r") as f:
        return json.load(f)

def render_question(question):
    """Render different types of form fields based on question type"""
    if question["type"] == "text":
        return st.text_input(question["question"], key=question["id"])
    elif question["type"] == "text_area":
        return st.text_area(question["question"], key=question["id"])
    elif question["type"] == "number":
        # Ensure all numeric values are float
        step = float(question.get("step", 1))
        min_value = float(question.get("min", 0))
        default = float(question.get("default", 0))
        
        try:
            return st.number_input(
                question["question"],
                min_value=min_value,
                step=step,
                value=default,
                key=question["id"],
                format="%.2f"
            )
        except Exception as e:
            st.error(f"Error rendering number input: {str(e)}")
            return default
    elif question["type"] == "radio":
        return st.radio(
            question["question"],
            options=question.get("options", ["Yes", "No"]),
            key=question["id"]
        )
    return None

def format_prompt(responses):
    """Format questionnaire responses into a structured prompt"""
    # Extract key information
    investment = responses.get('total_investment_amount_in_usd', '500000')
    business_type = responses.get('type_of_business_industry_services_products_offered', 'business')
    
    # Format the prompt with all relevant details
    prompt = f"""Generate a business plan for a ${investment} {business_type} E-2 Visa startup with the following details:

Business Information:
- Location: {responses.get('business_address_if_available', 'TBD')}
- Opening Date: {responses.get('estimated_business_opening_date', 'TBD')}
- Business Structure: {responses.get('business_structure_llc_corporation_etc', 'LLC')}

Investment Details:
- Total Investment: ${investment}
- Funds Usage: {responses.get('how_will_the_funds_be_used', 'TBD')}
- Current Investment Status: {responses.get('have_any_funds_already_been_invested', 'Not yet invested')}

Operations:
- Business Role: {responses.get('what_will_be_your_role_in_the_business_s_day_to_day_operations', 'Owner/Manager')}
- Current Status: {responses.get('have_you_already_started_business_operations', 'Not started')}
- Location Status: {responses.get('have_you_already_secured_a_business_lease_or_location', 'In progress')}

Market Analysis:
- Target Customers: {responses.get('who_are_your_target_customers', 'TBD')}
- Competitors: {responses.get('who_are_your_main_competitors', 'TBD')}
- Market Research: {responses.get('have_you_conducted_market_or_industry_research', 'In progress')}

Employment:
- Initial Team Size: {responses.get('how_many_employees_does_your_business_currently_have', '0')}
- Growth Plan: {responses.get('how_many_new_employees_do_you_plan_to_hire', 'TBD')}
- Position Types: {responses.get('what_types_of_jobs_will_you_create', 'TBD')}

Marketing Strategy:
- Channels: {responses.get('what_marketing_strategies_will_you_use', 'TBD')}
- Budget: {responses.get('do_you_have_a_marketing_budget_set_aside', 'TBD')}

Business Vision:
- Mission: {responses.get('what_is_the_mission_of_your_business', 'TBD')}
- Goals: {responses.get('what_are_your_short_term_1_year_and_long_term_5_years_goals', 'TBD')}
- Competitive Advantage: {responses.get('what_key_advantages_do_you_have_over_competitors', 'TBD')}"""

    return prompt

def main():
    st.title("E-2 Visa Business Plan Generator")
    st.markdown("Fill out this questionnaire to generate your customized E-2 visa business plan.")

    # Initialize generator
    generator = load_generator()

    # Load and display questionnaire
    questionnaire = load_questionnaire()

    # Create tabs for each section
    tabs = st.tabs([section_data["title"] for section_data in questionnaire.values()])

    # Dictionary to store all responses
    responses = {}

    # Render questions in each tab
    for tab, (section_id, section_data) in zip(tabs, questionnaire.items()):
        with tab:
            st.subheader(section_data["title"])
            for question in section_data["questions"]:
                try:
                    response = render_question(question)
                    if response is not None:  # Only store non-None responses
                        responses[question["id"]] = response
                except Exception as e:
                    st.error(f"Error rendering question {question['id']}: {str(e)}")

    # Add a progress indicator
    progress = st.progress(0)
    total_questions = sum(len(section["questions"]) for section in questionnaire.values())
    answered_questions = len([r for r in responses.values() if r])  # Count non-empty responses
    progress.progress(answered_questions / total_questions)

    st.write(f"Completed: {answered_questions}/{total_questions} questions")

    if st.button("Generate Business Plan"):
        if answered_questions < total_questions * 0.7:  # At least 70% completion required
            st.warning("Please fill out more questions to generate a comprehensive business plan.")
        else:
            prompt = format_prompt(responses)
            
            # Check cache first
            cached_plan = get_cached_plan(prompt)
            if cached_plan:
                st.write(cached_plan)
                return

            with st.spinner("Generating your business plan..."):
                try:
                    response = generator(prompt)
                    # Cache the result
                    cache_plan(prompt, response)
                    st.write(response)
                except Exception as e:
                    st.error(f"Error generating business plan: {str(e)}")

if __name__ == "__main__":
    main()