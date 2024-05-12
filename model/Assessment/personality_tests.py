import argparse
import json
import os
import random
from tqdm import tqdm
from utils import logger, get_response_json

random.seed(42)

# Parser setup for command-line options
parser = argparse.ArgumentParser(description='Assess BFI personality traits of a character using the Big-5 1024 dataset')
parser.add_argument('--character', type=str, default='default_character', help='Character name or code')
parser.add_argument('--persona_index', type=int, default=0, help='Index of persona within the Big-5 1024 dataset')
parser.add_argument('--agent_llm', type=str, default='latest-api', help='LLM to use for generating responses, e.g., "gpt-4"')
args = parser.parse_args()

# Configuration and paths
config_path = 'config.json'
questionnaire_path = '../data/BFI.json'
personas_path = '../data/Big-5_1024_Personas.json'

# Load configuration
with open(config_path, 'r') as file:
    config = json.load(file)

def load_questionnaire():
    with open(questionnaire_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def load_persona(index):
    with open(personas_path, 'r', encoding='utf-8') as file:
        personas = json.load(file)
    try:
        return personas[index]
    except IndexError:
        logger.error(f"Index out of bounds for personas: {index}")
        exit(1)

def interview(character_agent, questionnaire):
    results = []
    for question in tqdm(questionnaire):
        response = character_agent.chat(question['rewritten_en'])
        results.append({
            'question_id': question['id'],
            'question': question['rewritten_en'],
            'response': response
        })
    return results

class ChatAgent:
    def __init__(self, persona, llm):
        self.persona = persona
        self.llm = llm

    def chat(self, prompt):
        # Enhanced response simulation to reflect persona traits
        return f"Response as {self.persona['name']}: {prompt} (simulated)"

def main():
    persona = load_persona(args.persona_index)
    character_agent = ChatAgent(persona=persona, llm=args.agent_llm)
    questionnaire = load_questionnaire()
    
    # Conducting the interview
    print("Starting the interview with persona:", persona['name'])
    results = interview(character_agent, questionnaire['questions'])
    
    # Output the results
    for result in results:
        print(f"Q: {result['question']}")
        print(f"A: {result['response']}\n")

if __name__ == '__main__':
    main()
