import openai
import json
import argparse
import random
from tqdm import tqdm

def load_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def save_results(results, output_path):
    with open(output_path, 'w') as file:
        json.dump(results, file, indent=4)

def get_openai_api_key():
    return "your_openai_api_key_here"

def chatgpt_score_response(response, dimension):
    """ Utilizes ChatGPT to score responses based on personality traits """
    openai.api_key = get_openai_api_key()
    prompt = f"Rate the following response on a scale of 1-5 for the trait {dimension}: {response}"
    response = openai.Completion.create(
        engine="text-davinci-002",  # Update model if needed
        prompt=prompt,
        max_tokens=10
    )
    score = extract_score(response.choices[0].text.strip())
    return score

def extract_score(text):
    """ Extracts a numerical score from text output """
    try:
        return int(text.split()[-1])
    except ValueError:
        return 3  # Default to a neutral score if parsing fails

def hit_at_k(average_scores, persona_thresholds, k):
    """ Check if scored traits are within k of their thresholds """
    hits = {}
    for dimension, score in average_scores.items():
        target_score = persona_thresholds.get(dimension, 0)
        if abs(target_score - score) <= k:
            hits[dimension] = True
        else:
            hits[dimension] = False
    return hits

def conduct_interview(persona, questionnaire):
    results = []
    scores = {}
    persona_thresholds = persona.get('profile', {}).get('expected_scores', {})
    for question in questionnaire['questions']:
        response = "Example response"  # Simulated response for this example
        score = chatgpt_score_response(response, question['dimension'])
        results.append({"question": question['rewritten_en'], "response": response, "score": score})
        dimension = question['dimension']
        scores.setdefault(dimension, []).append(score)
    average_scores = {dim: sum(scores[dim]) / len(scores[dim]) for dim in scores}
    hit_k_results = hit_at_k(average_scores, persona_thresholds, k=1)  # k value can be adjusted as needed
    return results, average_scores, hit_k_results

def run_experiments(personas, questionnaire, num_personas=None):
    if num_personas:
        personas = random.sample(personas, num_personas)
    overall_results = {}
    for persona in tqdm(personas):
        results, average_scores, hit_k_results = conduct_interview(persona, questionnaire)
        overall_results[persona['profile']['name']] = {
            "results": results,
            "average_scores": average_scores,
            "hit@k": hit_k_results
        }
    return overall_results

def main():
    parser = argparse.ArgumentParser(description="Run personality assessments with ChatGPT scoring.")
    parser.add_argument('--persona_file', type=str, default='./data/big5-1024-persona.json', help="File containing the personas.")
    parser.add_argument('--questionnaire_file', type=str, default='./data/BFI.json', help="Questionnaire file for the BFI.")
    parser.add_argument('--output_file', type=str, default='assessment_results.json', help="File to save the assessment results.")
    parser.add_argument('--num_personas', type=int, help="Number of personas to assess, randomly chosen if not all.")
    args = parser.parse_args()

    personas = load_data(args.persona_file)
    questionnaire = load_data(args.questionnaire_file)

    results = run_experiments(personas, questionnaire, args.num_personas)

    save_results(results, args.output_file)
    print(f"Assessment completed. Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
